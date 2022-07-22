from gym import Env
from gym.spaces import Discrete, Box
import numpy as np


class MatrixEnv(Env):
    def __init__(self):
        # Actions determined by the pivot index
        self.action_space = Discrete(10)
        # Array representing the matrix
        self.observation_space = Box(low=-100, high=100, shape=(10, 10), dtype=np.float32)
        # Set initial matrix state
        self.state = initial_mat
        # Set initial index
        self.index = 0
        # Set number of steps
        self.n_steps = 10

    def step(self, action):
        # matrix_update(mat, pivot) updates mat by pivoting on pivot and reducing rows with Gaussian elimination
        self.state = matrix_update(self.state, action)
        self.n_steps -= 1

        if action < self.index:
            reward = -100
        else:
            # reward_fn(mat, pivot) takes a mat and returns the number of (approx) zero entries in the pivot row
            reward = reward_fn(self.state, action)

        self.index += 1

        if self.n_steps <= 0:
            done = True
        else:
            done = False

        return self.state, reward, done