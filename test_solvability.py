import tensorflow as tf
import numpy as np
from teachDRL.gym_flowers.envs.maze_env import *
from gym.wrappers.time_limit import TimeLimit
from torchvision.utils import save_image
import torch

env_config = {}
env_config['device'] = "cuda"
env_config['maze_model_path'] = os.path.join(os.path.abspath(os.getcwd()), f'teachDRL/models/generator_aldous-pacman_4.pth')
env_f = lambda: TimeLimit(MazeEnv(env_config), max_episode_steps=1000)

env = env_f()

o, r, d, ep_ret, ep_len = env.reset(random=True), 0, False, 0, 0
save_image(torch.tensor(o), 'test_solvability.png', normalize=True)
print(env.is_solvable())