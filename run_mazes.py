import argparse
from teachDRL.spinup.utils.run_utils import setup_logger_kwargs
from teachDRL.spinup.algos.ppo.ppo import ppo
from teachDRL.spinup.algos.ppo import core
import gym
from gym.wrappers.time_limit import TimeLimit
from teachDRL.gym_flowers.envs.maze_env import *
from teachDRL.teachers.teacher_controller import TeacherController
from collections import OrderedDict
import os
import numpy as np
from torchvision.utils import save_image
import torch
import yaml


# Argument definition
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='config')

args = parser.parse_args()

config = yaml.safe_load(open(f'teachDRL/config/{arg.config}.yaml'))

logger_kwargs = setup_logger_kwargs(config["exp_name"], config["seed"])

# Bind this run to specific GPU if there is one
if config["args.gpu_id"] != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["args.gpu_id"])

# Set up Student's DeepNN architecture if provided
ac_kwargs = dict()
if config["student"]["hid"] != -1:
    ac_kwargs['hidden_sizes'] = [config["student"]["hid"]] * config["student"]["l"]

# Get architecture of the actor critic
if config["student"]["actor"] == "convolutional":
    actor_critic = core.convolutional_actor_critic
elif config["student"]["actor"] == "mlp":
    actor_critic = core.mlp_actor_critic


# Set bounds for environment's parameter space format:[min, max, nb_dimensions] (if no nb_dimensions, assumes only 1)
## To modify with the model
param_env_bounds = OrderedDict()

param_env_bounds['Z'] = [-100, 100, 100]


# Set Teacher hyperparameters
params = {}
if args.teacher == 'ALP-GMM':
    if args.gmm_fitness_fun is not None:
        params['gmm_fitness_fun'] = args.gmm_fitness_fun
    if args.min_k is not None and args.max_k is not None:
        params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
    if args.weighted_gmm is True:
        params['weighted_gmm'] = args.weighted_gmm
    if args.nb_em_init is not None:
        params['nb_em_init'] = args.nb_em_init
    if args.fit_rate is not None:
        params['fit_rate'] = args.fit_rate
    if args.alp_max_size is not None:
        params['alp_max_size'] = args.alp_max_size
elif args.teacher == 'Covar-GMM':
    if args.absolute_lp is True:
        params['absolute_lp'] = args.absolute_lp
elif args.teacher == "RIAC":
    if args.max_region_size is not None:
        params['max_region_size'] = args.max_region_size
    if args.alp_window_size is not None:
        params['alp_window_size'] = args.alp_window_size
elif args.teacher == "Oracle":
    if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
        params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
    elif 'poly_shape' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 12
        print('hih')
    elif 'stump_seq' in param_env_bounds:
        params['window_step_vector'] = [0.1] * 10
    else:
        print('Oracle not defined for this parameter space')
        exit(1)

env_config = {}
env_config['device'] = "cuda"
env_config['maze_model_path'] = os.path.join(os.path.abspath(os.getcwd()), f'teachDRL/models/{config["model"]}.pth')
env_f = lambda: TimeLimit(MazeEnv(env_config), max_episode_steps=1000)
env_init = {}

# Initialize teacher
Teacher = TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds,
                            seed=args.seed, teacher_params=params)

# Launch Student training

ppo(env_f, actor_critic=actor_critic, ac_kwargs=ac_kwargs, gamma=config["student"]["gamma"], seed=config["seed"], epochs=config["students"]["epochs"],
    logger_kwargs=logger_kwargs, max_ep_len=config["student"]["max_ep_len"], steps_per_epoch=config["student"]["steps_per_ep"], Teacher=Teacher, path_gif = os.path.join(os.path.abspath(os.getcwd()), 'teachDRL/data/test_set/'))





