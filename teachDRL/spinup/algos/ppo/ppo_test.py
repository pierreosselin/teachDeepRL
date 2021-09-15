import numpy as np
import tensorflow as tf
import gym
import time
import teachDRL.spinup.algos.ppo.core as core
from teachDRL.spinup.utils.logx import EpochLogger
from teachDRL.spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from teachDRL.spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from tqdm import tqdm
from PIL import Image
from moviepy.editor import ImageSequenceClip
import os
from torchvision.utils import save_image
import torch
from pathlib import Path
import wandb

os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""
Proximal Policy Optimization (by clipping), 
with early stopping based on approximate KL
"""

def images_to_gif(l_images, epoch, name, path_gif):
    """
    Output gif from list of images
    """
    new_l = [np.repeat(el.astype(np.uint8)[:, :, np.newaxis], 3, axis=2) for el in l_images]
    new_l = [el[:,:,np.newaxis]/3 for el in l_images]
    clip = ImageSequenceClip(new_l, fps=20, ismask=True)
    
    clip.write_gif(os.path.join(path_gif, f'{name}_epoch_number_{epoch}.gif'), fps=20)



def ppo(env_fn, actor_critic=core.mlp_actor_critic, config=None, ac_kwargs=dict(), logger_kwargs=dict(),
        Teacher=None, path_gif=None, path_sampled_maze=None, path_metrics=None, path_maze_visu=None, gpu_name = "/device:CPU:0"):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:
            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``.
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    #allocate configuration elements
    gamma = config["student"]["gamma"]
    seed = config["seed"]
    epochs = config["student"]["epochs"]
    max_ep_len = config["student"]["max_ep_len"]
    steps_per_epoch = config["student"]["steps_per_ep"]
    clip_ratio = config["student"]["clip_ratio"]
    pi_lr = config["student"]["pi_lr"]
    vf_lr = config["student"]["vf_lr"]
    train_pi_iters = config["student"]["train_pi_iters"]
    train_v_iters = config["student"]["train_v_iters"]
    lam = config["student"]["lam"]
    target_kl = config["student"]["target_kl"]
    save_freq = config["student"]["save_freq"]
    test_freq = config["student"]["test_freq"]
    epochs_per_task = config["student"]["epochs_per_task"]


    ## Load maze for gif visualization
    maze_visu = np.load(path_maze_visu)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    ## Initialize Wandb
    wandb.init(project="test convergence", entity="pierre77")


    seed += 10000 * proc_id()
    #tf.set_random_seed(seed)
    #np.random.seed(seed)

    env, test_env, visu_env = env_fn(), env_fn(), env_fn()

    if Teacher: Teacher.set_env_params(env)


    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    with tf.device(gpu_name):
        x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    # Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})


    def visualize_test(epoch, maze_visu):
        o, r, d, ep_ret, ep_len = visu_env.reset(), 0, False, 0, 0
        o = visu_env.env.set_environment_maze(maze_visu)
        
        images_gif = []
        o_new = o.reshape(17,17)
        o_new[visu_env.env.y, visu_env.env.x] = 3
        images_gif.append(o_new)

        while not(d or (ep_len == max_ep_len)):
            o, r, d, _ = visu_env.step(sess.run(pi, feed_dict={x_ph: np.expand_dims(o, axis=0)})[0])
            ep_ret += r
            ep_len += 1
            o_new = o.reshape(17,17)
            o_new[visu_env.env.y, visu_env.env.x] = 3
            images_gif.append(o_new)
        images_to_gif(images_gif, epoch, "test_maze_visualization" + Teacher.teacher, path_gif)


    def test_agent(n, sess, pi):
        print("Test Set Evaluation...")
        list_rewards = []
        for j in tqdm(range(n)):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            if Teacher:
                o = Teacher.set_test_env_params(test_env)
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(sess.run(pi, feed_dict={x_ph: np.expand_dims(o, axis=0)})[0])
                ep_ret += r
                ep_len += 1

            list_rewards.append(ep_ret)
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        if Teacher: Teacher.record_test_episode(np.mean(list_rewards), 0)


    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in tqdm(range(train_pi_iters), desc="Gradient pi"):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in tqdm(range(train_v_iters), desc="Gradient v"):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))
        wandb.log({"pi_loss": pi_l_old, })

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(random=True), 0, False, 0, 0

    # Resample if not solvable
    while not env.is_solvable():
        print("Maze not solvable, resampling...")
        o, r, d, ep_ret, ep_len = env.reset(random=True), 0, False, 0, 0


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in tqdm(range(local_steps_per_epoch)):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: np.expand_dims(o, axis=0)})
            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)
            
            o, r, d, _ = env.step(a[0])
            
            ep_ret += r
            ep_len += 1
            
            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: np.expand_dims(o, axis=0)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                if Teacher:
                    Teacher.record_train_episode(ep_ret, ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Perform PPO update!
        update()
        
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    ### Check peformance on maze : Save a gif
    first_maze = env.env.maze
    visualize_test(1, first_maze)

    Teacher.set_env_params(env)
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    while not env.is_solvable():
        print("Maze not solvable, resampling...")
        Teacher.set_env_params(env)
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    print("new maze")
    print(env.env.maze)

    # Second training on different environment
    for epoch in range(epochs):
        for t in tqdm(range(local_steps_per_epoch)):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: np.expand_dims(o, axis=0)})
            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)
            
            o, r, d, _ = env.step(a[0])
            
            ep_ret += r
            ep_len += 1
            
            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else sess.run(v, feed_dict={x_ph: np.expand_dims(o, axis=0)})
                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                if Teacher:
                    Teacher.record_train_episode(ep_ret, ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()
    
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    ### Save the resulting policy on second environment
    visualize_test(2, env.env.maze)

    ### Check performance on first maze as well
    visualize_test(3, first_maze)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from teachDRL.spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)