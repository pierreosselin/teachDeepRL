import torch
import os
from torch.utils.data import Dataset

import sys
sys.path.insert(1, os.getcwd()+'/../maze_generators')

import pacman
import aldousbroder

class MazeDataset(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, maze='pacman', size=(32, 32), length=100000, transform=None):
        if maze=='pacman':
            self.maze_generator = pacman.PacmanMazeGenerator(size=size)
        elif maze =='aldous-broder':
            self.maze_generator = aldousbroder.AldousBroderMazeGenerator(size=size)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.tensor(self.maze_generator.generate_maze()).unsqueeze(0)

if __name__=='__main__':
    d = MazeDataset()
    print(d.__getitem__(0))