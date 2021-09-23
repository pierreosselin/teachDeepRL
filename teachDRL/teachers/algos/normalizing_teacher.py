from gym.spaces import Box
from teachDRL.teachers.utils.dataset import BufferedDataset
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model
from tqdm import tqdm


tf.keras.backend.set_floatx('float32')
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

@tf.function
def nll(distribution, data):
    """
    Computes the negative log liklihood loss for a given distribution and given data.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param data: Data or a batch from data.
    :return: Negative Log Likelihodd loss.
    """
    return -tf.reduce_mean(distribution.log_prob(data))


class Made(tfk.layers.Layer):
    """
    Implementation of a Masked Autoencoder for Distribution Estimation (MADE) [Germain et al. (2015)].
    The existing TensorFlow bijector "AutoregressiveNetwork" is used. The output is reshaped to output one shift vector
    and one log_scale vector.
    :param params: Python integer specifying the number of parameters to output per input.
    :param event_shape: Python list-like of positive integers (or a single int), specifying the shape of the input to this layer, which is also the event_shape of the distribution parameterized by this layer. Currently only rank-1 shapes are supported. That is, event_shape must be a single integer. If not specified, the event shape is inferred when this layer is first called or built.
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: An activation function. See tf.keras.layers.Dense. Default: None.
    :param use_bias: Whether or not the dense layers constructed in this layer should have a bias term. See tf.keras.layers.Dense. Default: True.
    :param kernel_regularizer: Regularizer function applied to the Dense kernel weight matrices. Default: None.
    :param bias_regularizer: Regularizer function applied to the Dense bias weight vectors. Default: None.
    """

    def __init__(self, params, event_shape=None, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.event_shape = event_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, event_shape=event_shape, hidden_units=hidden_units,
                                                 activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


@tf.function
def train_density_estimation(distribution, optimizer, batch):
    """
    Train function for density estimation normalizing flows.
    :param distribution: TensorFlow distribution, e.g. tf.TransformedDistribution.
    :param optimizer: TensorFlow keras optimizer, e.g. tf.keras.optimizers.Adam(..)
    :param batch: Batch of the train data.
    :return: loss.
    """
    with tf.GradientTape() as tape:
        tape.watch(distribution.trainable_variables)
        loss = -tf.reduce_mean(distribution.log_prob(batch))  # negative log likelihood
    gradients = tape.gradient(loss, distribution.trainable_variables)
    optimizer.apply_gradients(zip(gradients, distribution.trainable_variables))

    return loss

class NN(Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: Activation of the hidden units
    """
    def __init__(self, input_shape, n_hidden=[512, 512], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation))
        self.layer_list = layer_list
        self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
        self.t_layer = Dense(input_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t


class RealNVP(tfb.Bijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    This implementation only works for 1D arrays.
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    """

    def __init__(self, input_shape, n_hidden=[512, 512], forward_min_event_ndims=1, validate_args: bool = False, name="real_nvp"):
        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        assert input_shape % 2 == 0
        input_shape = input_shape // 2
        nn_layer = NN(input_shape, n_hidden)
        x = tf.keras.Input(input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name="nn")
        
    def _bijector_fn(self, x):
        log_s, t = self.nn(x)
        return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
    
    def _inverse_log_det_jacobian(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)




def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]

# Absolute Learning Progress (ALP) computer object
# It uses a buffered kd-tree to efficiently implement a k-nearest-neighbor algorithm
class EmpiricalALPComputer():
    def __init__(self, task_size, max_size=None, buffer_size=500):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    def compute_alp(self, task, reward):
        alp = 0
        if len(self.alp_knn) > 5:
            # Compute absolute learning progress for new task
            
            # 1 - Retrieve closest previous task
            dist, idx = self.alp_knn.nn_y(task)
            
            # 2 - Retrieve corresponding reward
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])

            # 3 - Compute alp as absolute difference in reward
            lp = reward - closest_previous_task_reward
            alp = np.abs(lp)

        # Add to database
        self.alp_knn.add_xy(reward, task)
        return alp

# Absolute Learning Progress - Gaussian Mixture Model
# mins / maxs are vectors defining task space boundaries (ex: mins=[0,0,0] maxs=[1,1,1])
class NF():
    def __init__(self, mins, maxs, seed=None, params=dict()):
        
        ### Think about whether or not remove its
        self.seed = seed

        # initialize flow

        #if not seed:
        #    self.seed = np.random.randint(42,424242)
        #np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        print("Hello here is the vector length")
        print(len(mins))

        self.dim_input = len(mins)

        # Restart new fit by initializing with last fit
        self.warm_start = False if "warm_start" not in params else params["warm_start"]
        # Number of episodes between two fit of the NF
        self.fit_rate = 250 if "fit_rate" not in params else params['fit_rate']
        self.nb_random = self.fit_rate  # Number of bootstrapping episodes

        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = 0.2 if "random_task_ratio" not in params else params["random_task_ratio"]
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)

        # Maximal number of episodes to account for when computing ALP
        alp_max_size = None if "alp_max_size" not in params else params["alp_max_size"]
        alp_buffer_size = 500 if "alp_buffer_size" not in params else params["alp_buffer_size"]

        # Init ALP computer
        self.alp_computer = EmpiricalALPComputer(len(mins), max_size=alp_max_size, buffer_size=alp_buffer_size)

        self.tasks = []
        self.alps = []
        self.tasks_alps = []

        # Init NF
        hidden_shape = [200, 200]  # hidden shape for MADE network of MAF
        layers = 12  # number of layers of the flow

        base_dist = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution

        bijectors = []
        for i in range(0, layers):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
            bijectors.append(tfb.Permute(permutation=list(reversed(range(self.dim_input)))))  # data permutation after layers of MAF
            
        self.bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

        self.maf = tfd.TransformedDistribution(
            distribution=tfd.Sample(base_dist, sample_shape=[self.dim_input]),
            bijector=self.bijector
        )

        self.samples = self.maf.sample()


        base_lr = 1e-3
        end_lr = 1e-4
        self.max_epochs = int(5e3)  # maximum number of epochs of the training
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(base_lr, self.max_epochs, end_lr, power=0.5)
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer



        print("Initialization FInished")

        # Boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_alps': [], 'episodes': []}


    def update(self, task, reward):
        self.tasks.append(task)

        # Compute corresponding ALP 
        self.alps.append(self.alp_computer.compute_alp(task, reward))

        # Concatenate task vector with ALP dimension
        self.tasks_alps.append(np.array(task.tolist() + [self.alps[-1]]))

        if len(self.tasks) >= self.nb_random:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # 1 - Retrieve last <fit_rate> (task, reward) pairs
                cur_tasks = self.tasks[-self.fit_rate:]
                cur_alps = np.array(self.alps[-self.fit_rate:])

                # 2 - Fit normalizing flow
                global_step = []
                train_losses = []
                min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)


                # start training
                for i in tqdm(range(self.max_epochs)):

                    ### Resampling dataset according to ALP
                    if np.sum(cur_alps) > 0:
                        batches = np.random.choice(cur_tasks, self.fit_rate, p = cur_alps/np.sum(cur_alps))
                    else:
                        batches = np.array(cur_tasks)


                    for batch in batches:
                        train_loss = train_density_estimation(self.maf, self.opt, batch)

                    if i % int(100) == 0:
                        #val_loss = nll(maf, val_data)
                        global_step.append(i)
                        train_losses.append(train_loss)
                        #val_losses.append(val_loss)
                        print(f"{i}, train_loss: {train_loss}")

                        if train_loss < min_train_loss:
                            min_train_loss = train_loss
                            min_train_epoch = i
                        """
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            min_val_epoch = i
                            checkpoint.write(file_prefix=checkpoint_prefix)  # overwrite best val model

                        elif i - min_val_epoch > delta_stop:  # no decrease in min_val_loss for "delta_stop epochs"
                            break
                        """


                # book-keeping
                self.bk['tasks_alps'] = self.tasks_alps
                self.bk['episodes'].append(len(self.tasks))

    def sample_task(self):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Random task sampling
            new_task = self.random_task_generator.sample()
            print("Sample random task")
        else:
            print("Sample GMM task")
            # ALP-based task sampling

            # 1 - Sample Gaussian proportionally to its mean ALP
            new_task = self.maf.sample(1)
            # 3 - Sample task in Gaussian, without forgetting to remove ALP dimension
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        return new_task

    def dump(self, dump_dict):
        dump_dict.update(self.bk)
        return dump_dict