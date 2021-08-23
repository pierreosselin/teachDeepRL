import torch
import random
import os
import numpy as np
from torch.autograd import Variable

from gym import Env, spaces

import sys
sys.path.insert(1, os.getcwd()+'/../gan')
import gan_utils

WALL = 0
FREE_SPACE = 1
GOAL = 2

class WGanGPConfig:
    n_epochs = 200
    batch_size = 64
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    latent_dim = 100
    img_size = 32
    channels = 1
    n_critic = 5
    clip_value = 0.01
    sample_interval = 400
    model_dir = '../../data/models'
    maze_type = 'pacman'


cfg = WGanGPConfig()

class MazeEnv(Env):
    def __init__(self, 
                 env_config):
        super(MazeEnv, self).__init__()

        if 'device' not in env_config.keys():
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = env_config['device']

        self.generator = gan_utils.Generator_Conv(cfg)
        if self.device == "cuda":
            self.generator.cuda()

        self.generator.load_state_dict(torch.load(str(env_config['maze_model_path']), map_location=self.device))


        self.action_space = spaces.Discrete(4) # 4 actions
        self.obs_radius = env_config['obs_radius']
        self.observation_space = spaces.Box(low=0, 
                                            high=2, 
                                            shape=(self.obs_radius*2 + 1, self.obs_radius*2 + 1), 
                                            dtype=np.uint8)
        
        self.Z = Variable(torch.tensor(np.random.normal(0, 1, (1, self.generator.latent_dim)), dtype=torch.float)).cuda()
        self.reset()

    def step(self, action):
        print(action)

        if action == 0:
            # up
            x_t = self.x
            y_t = self.y - 1
        elif action == 1:
            # down
            x_t = self.x
            y_t = self.y + 1
        elif action == 2:
            # left
            x_t = self.x - 1
            y_t = self.y 
        elif action == 3:
            # right
            x_t = self.x + 1
            y_t = self.y 
        
        if x_t >= 0 and x_t < self.maze.shape[1] and y_t >= 0 and y_t < self.maze.shape[0]:
            if self.maze[y_t, x_t] != WALL:
                self.x = x_t
                self.y = y_t

        return self._get_obs(), self._get_reward(), self._is_solved(), {}

    def _is_solved(self):
        if self.x == self.goal_x and self.goal_y:
            return True
        else:
            return False
    
    def _get_reward(self):
        if self.x == self.goal_x and self.y == self.goal_y:
            return 1
        else:
            return 0

    """
    def _get_obs(self):
        return self.padded_maze[self.y : self.y + 2 * self.obs_radius + 1,
                                self.x : self.x + 2 * self.obs_radius + 1].detach().cpu().numpy()
    """

    def _get_obs(self):
        return self.maze

    def reset(self, random = False):
        if random:
            self.maze = self.generator.generate_random()[0][0]
        else:
            print(self.Z)
            print(self.Z.shape)
            self.maze = self.generator.forward(self.Z)
        self.x = 0
        self.y = self.maze.shape[0] - 1

        self.goal_x, self.goal_y = self._sample_goal()

        return self._get_obs()

    def set_environment(self, **param_dict):
        self.Z = Variable(torch.tensor(param_dict["Z"].reshape(1,-1))).cuda()

    def _sample_goal(self):
        goal_y = self.maze.shape[0] - 1
        goal_x = 0

        self.maze[goal_y, goal_x] = GOAL

        return (goal_x, goal_y)