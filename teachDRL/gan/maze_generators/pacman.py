import numpy as np 
import cv2
import itertools
import pacman_blocks

class PacmanMazeGenerator():
    def __init__(self, size=(17,17)):
        self.size_out = size
        self.size = (size[0] + 2, size[1] + 2) # h w
        self.block = pacman_blocks.PacmanBlock()
        self.map_dict = {0:1, 1:1, 2:0}

    def generate_maze(self):
        board = np.zeros(self.size)
        pointer = (0, 0)
        self.is_done = False
        
        while not self.is_done:
            board, pointer = self._add_block(board, pointer)

        board_out = np.vectorize(self.map_dict.get)(board)[:self.size_out[0], :self.size_out[1]]
        # ensure solvable from lower left corner
        board_out[-1, 0] = 1
        board_out[-2, 0] = 1
        board_out[-1, 1] = 1

        return board_out

    def _add_block(self, board, pointer):
        max_size = self._max_size(board, pointer)
        block = self.block.get_block(max_size)
        while block is None and not self.is_done:
            pointer = self._move_pointer(board, pointer)
            max_size = self._max_size(board, pointer)
            block = self.block.get_block(max_size)
        
        if self.is_done:
            return board, pointer
        
        h0, w0 = pointer
        h, w = block.shape
        board[h0 : h0 + h, w0 : w0 + w] = block
        pointer = (h0, w0 + w)
        if pointer[1] >= board.shape[1]-1:
            pointer = (h0+1, 0)

        return board, pointer

    def _move_pointer(self, board, pointer):
        while not self.is_done:
            if pointer == board.shape:
                self.is_done = True 
                return pointer

            h, w = pointer
            if pointer[1] >= board.shape[1]-1:
                pointer = (h+1, 0)
            else: 
                pointer = (h, w+1)
            
            if (pointer[0]+1, pointer[1]+1) == board.shape:
                self.is_done = True 
                return pointer

            if board[pointer] == 0:
                return pointer

    def _max_size(self, board, pointer):
        h_cnt = 0
        w_cnt = 0
        for i in range(pointer[0], board.shape[0]):
            if board[i, pointer[1]] == 0:
                h_cnt += 1
            else:
                break
        for j in range(pointer[1], board.shape[1]):
            if board[pointer[0], j] == 0:
                w_cnt += 1
            else:
                break
        return (h_cnt, w_cnt)

def make_grid(imgs, w=4, h=4, n=16, margin=1):
    w = w
    h = h
    n = n

    if len(imgs) != n:
        raise ValueError('Number of images ({}) does not match '
                         'matrix size {}x{}'.format(w, h, len(imgs)))

    img_h, img_w = imgs[0].shape

    m_x = 0
    m_y = 0
    if margin is not None:
        m_x = int(margin)
        m_y = m_x

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),),
                         np.uint8)

    imgmatrix.fill(0)    

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w,] = img

    return imgmatrix

if __name__ == '__main__':
    p = PacmanMazeGenerator(size=(17, 17))
    imgs = [p.generate_maze() for i in range(16)]
    imgs_grid = make_grid(imgs)
    cv2.imwrite(f'/Users/suny/Desktop/Uni/DPhil/Projects/RL/mazegan/data/pacman-mazes/mazes.png', imgs_grid*255)

