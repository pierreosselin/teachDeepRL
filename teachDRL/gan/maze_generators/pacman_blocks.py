import numpy as np
import random

rectangle3x3 = np.array([[1, 1, 1],
                         [1, 2, 1],
                         [1, 1, 1]])
rectangle2x5 = np.array([[1, 1, 1, 1],
                         [1, 2, 2, 1],
                         [1, 1, 1, 1]])
rectangle5x2 = np.transpose(rectangle2x5)

rectangle3x5 = np.array([[1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1]])
rectangle5x3 = np.transpose(rectangle3x5)

rectangle4x5 = np.array([[1, 1, 1, 1, 1, 1],
                         [1, 2, 2, 2, 2, 1],
                         [1, 1, 1, 1, 1, 1]])
rectangle5x4 = np.transpose(rectangle4x5)

L_up_right = np.array([[1, 1, 1, 1, 1],
                       [1, 2, 1, 1, 1],
                       [1, 2, 2, 2, 1],
                       [1, 1, 1, 1, 1]])
L_up_left =  np.flip(L_up_right, 1)
L_down_right =  np.flip(L_up_right, 0)
L_down_left =  np.flip(L_up_left, 0)

L_transpose_up_right = np.transpose(L_up_right)
L_transpose_up_left = np.transpose(L_up_right)
L_transpose_down_right = np.transpose(L_down_right)
L_transpose_down_left = np.transpose(L_down_right)

L_long_up_right = np.array([[1, 1, 1, 1, 1, 1],
                            [1, 2, 1, 1, 1, 1],
                            [1, 2, 2, 2, 2, 1],
                            [1, 1, 1, 1, 1, 1]])
L_long_up_left =  np.flip(L_long_up_right, 1)
L_long_down_right =  np.flip(L_long_up_right, 0)
L_long_down_left =  np.flip(L_long_up_left, 0)

L_long_transpose_up_right = np.transpose(L_long_up_right)
L_long_transpose_up_left = np.transpose(L_long_up_right)
L_long_transpose_down_right = np.transpose(L_long_down_right)
L_long_transpose_down_left = np.transpose(L_long_down_right)

T_up = np.array([[1, 1, 1, 1, 1],
                 [1, 1, 2, 1, 1],
                 [1, 2, 2, 2, 1],
                 [1, 1, 1, 1, 1]])
T_left = np.transpose(T_up)
T_down = np.flip(T_up, 0)
T_right = np.transpose(T_down)

T_long_up_right = np.array([[1, 1, 1, 1, 1, 1],
                            [1, 1, 2, 1, 1, 1],
                            [1, 2, 2, 2, 2, 1],
                            [1, 1, 1, 1, 1, 1]])
T_long_up_left =  np.flip(T_long_up_right, 1)
T_long_down_right =  np.flip(T_long_up_right, 0)
T_long_down_left =  np.flip(T_long_up_left, 0)

T_long_transpose_up_right = np.transpose(T_long_up_right)
T_long_transpose_up_left = np.transpose(T_long_up_right)
T_long_transpose_down_right = np.transpose(T_long_down_right)
T_long_transpose_down_left = np.transpose(T_long_down_right)

T_short_up_right = np.array([[1, 1, 1, 1],
                            [1, 2, 1, 1],
                            [1, 2, 2, 1],
                            [1, 1, 1, 1]])
T_short_up_left =  np.flip(T_short_up_right, 1)
T_short_down_right =  np.flip(T_short_up_right, 0)
T_short_down_left =  np.flip(T_short_up_left, 0)

T_short_transpose_up_right = np.transpose(T_short_up_right)
T_short_transpose_up_left = np.transpose(T_short_up_right)
T_short_transpose_down_right = np.transpose(T_short_down_right)
T_short_transpose_down_left = np.transpose(T_short_down_right)

S_up = np.array([[1, 1, 1, 1, 1],
                 [1, 2, 2, 1, 1],
                 [1, 1, 2, 2, 1],
                 [1, 1, 1, 1, 1]])
S_left = np.transpose(S_up)
S_down = np.flip(S_up, 0)
S_right = np.transpose(S_down)

S_long_up_right = np.array([[1, 1, 1, 1, 1, 1],
                            [1, 1, 2, 1, 1, 1],
                            [1, 2, 2, 2, 2, 1],
                            [1, 1, 1, 1, 1, 1]])
S_long_up_left =  np.flip(S_long_up_right, 1)
S_long_down_right =  np.flip(S_long_up_right, 0)
S_long_down_left =  np.flip(S_long_up_left, 0)

S_long_transpose_up_right = np.transpose(S_long_up_right)
S_long_transpose_up_left = np.transpose(S_long_up_right)
S_long_transpose_down_right = np.transpose(S_long_down_right)
S_long_transpose_down_left = np.transpose(S_long_down_right)




all_blocks = [#rectangle3x3, rectangle2x5, rectangle5x2, rectangle5x3, rectangle5x3, rectangle5x4, rectangle5x4,
            L_up_right, L_up_left, L_down_right, L_down_left,
            L_transpose_up_right, L_transpose_up_left, L_transpose_down_right, L_transpose_down_left,
            L_long_up_right, L_long_up_left, L_long_down_right, L_long_down_left,
            L_long_transpose_up_right, L_long_transpose_up_left, L_long_transpose_down_right, L_long_transpose_down_left,
            T_up, T_left, T_down, T_right, S_up, S_left, S_down, S_right,
            T_long_up_right, T_long_up_left, T_long_down_right, T_long_down_left,
            T_long_transpose_up_right, T_long_transpose_up_left, T_long_transpose_down_right, T_long_transpose_down_left,
            T_short_up_right, T_short_up_left, T_short_down_right, T_short_down_left,
            T_short_transpose_up_right, T_short_transpose_up_left, T_short_transpose_down_right, T_short_transpose_down_left,
            S_long_up_right, S_long_up_left, S_long_down_right, S_long_down_left,
            S_long_transpose_up_right, S_long_transpose_up_left, S_long_transpose_down_right, S_long_transpose_down_left,]

class PacmanBlock():
    def __init__(self, dense=True):
        self.blocks = all_blocks
        if dense:
            self.blocks = [block for block in self.blocks]
        self.num_blocks = len(self.blocks)

    def get_block(self, shape_max = None):
        if shape_max is not None:
            blocks = [block for block in self.blocks if block.shape[0] <= shape_max[0] and block.shape[1] <= shape_max[1]]
        else:
            blocks = self.blocks
        if len(blocks) == 0:
            return None
        block = random.choice(blocks)
        block = random.choice([block[:-1, :], block[:, 1:]])
        return block

if __name__ == '__main__':
    p = PacmanBlock()
    print(p.get_block())
    print(p.get_block(shape_max = (3,3)))

