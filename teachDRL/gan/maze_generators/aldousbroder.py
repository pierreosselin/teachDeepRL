# https://github.com/theJollySin/mazelib/blob/master/mazelib/generate/AldousBroder.py

import abc
import numpy as np
from numpy.random import shuffle
from random import choice, randrange
from pacman import make_grid
import cv2

class MazeGenAlgo:
    __metaclass__ = abc.ABCMeta

    def __init__(self, size=(17, 17)):
        h, w = size
        assert (w >= 3 and h >= 3), 'Mazes cannot be smaller than 3x3.'
        self.h = int((h+1)/2)
        self.w = int((w+1)/2)
        self.H = h 
        self.W = w 

    @abc.abstractmethod
    def generate(self):
        return None

    """ All of the methods below this are helper methods,
    common to many maze-generating algorithms.
    """

    def _find_neighbors(self, r, c, grid, is_wall=False):
        """ Find all the grid neighbors of the current position; visited, or not.
        Args:
            r (int): row of cell of interest
            c (int): column of cell of interest
            grid (np.array): 2D maze grid
            is_wall (bool): Are we looking for neighbors that are walls, or open cells?
        Returns:
            list: all neighboring cells that match our request
        """
        ns = []

        if r > 1 and grid[r - 2][c] == is_wall:
            ns.append((r - 2, c))
        if r < self.H - 2 and grid[r + 2][c] == is_wall:
            ns.append((r + 2, c))
        if c > 1 and grid[r][c - 2] == is_wall:
            ns.append((r, c - 2))
        if c < self.W - 2 and grid[r][c + 2] == is_wall:
            ns.append((r, c + 2))

        shuffle(ns)
        return ns

class AldousBroderMazeGenerator(MazeGenAlgo):
    """
    1. Choose a random cell.
    2. Choose a random neighbor of the current cell and visit it. If the neighbor has not
        yet been visited, add the traveled edge to the spanning tree.
    3. Repeat step 2 until all cells have been visited.
    """

    def __init__(self, size):
        super(AldousBroderMazeGenerator, self).__init__(size)

    def generate_maze(self):
        """ highest-level method that implements the maze-generating algorithm
        Returns:
            np.array: returned matrix
        """
        # create empty grid, with walls
        grid = np.empty((self.H, self.W), dtype=np.int8)
        grid.fill(0)

        crow = randrange(0, self.H, 2)
        ccol = randrange(0, self.W, 2)
        grid[crow][ccol] = 1 #Free space
        num_visited = 1

        while num_visited < self.h * self.w:
            # find neighbors
            neighbors = self._find_neighbors(crow, ccol, grid, 0)

            # how many neighbors have already been visited?
            if len(neighbors) == 0:
                # mark random neighbor as current
                (crow, ccol) = choice(self._find_neighbors(crow, ccol, grid, 1))
                continue

            # loop through neighbors
            for nrow, ncol in neighbors:
                if grid[nrow][ncol] == 0:
                    # open up wall to new neighbor
                    grid[(nrow + crow) // 2][(ncol + ccol) // 2] = 1
                    # mark neighbor as visited
                    grid[nrow][ncol] = 1
                    # bump the number visited
                    num_visited += 1
                    # current becomes new neighbor
                    crow = nrow
                    ccol = ncol
                    # break loop
                    break

        return grid

if __name__ == '__main__':
    p = AldousBroderMazeGenerator(size=(32, 32))
    imgs = [p.generate_maze() for i in range(16)]
    imgs_grid = make_grid(imgs)
    cv2.imwrite(f'/Users/suny/Desktop/Uni/DPhil/Projects/RL/mazegan/data/pacman-mazes/mazes_aldous_broder.png', imgs_grid*255)