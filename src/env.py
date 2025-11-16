import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.mazeGenerator import generateMaze

class Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.grid, self.start, self.goal = generateMaze()
        self.pos = None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.grid.shape[0]-1, self.grid.shape[1]-1]),
            shape=(2,),
            dtype=np.int32,
        )

        self.fig = None
        self.ax = None
        self.img = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        return np.array(self.pos, dtype=np.int32), {}

    def step(self, action):
        row, col = self.pos
        d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][int(action)]
        new_row, new_col = row + d_row, col + d_col

        reward = -1.0
        terminated = False

        if (0 <= new_row < self.grid.shape[0] and
            0 <= new_col < self.grid.shape[1] and
            self.grid[new_row, new_col] == 0):
            self.pos = (new_row, new_col)
        else:
            reward = -10.0 

        if self.pos == self.goal:
            reward = 100.0
            terminated = True

        return np.array(self.pos, dtype=np.int32), reward, terminated, {}

    def render(self):
        display_grid = self.grid.astype(float)
        gr, gc = self.goal
        display_grid[gr, gc] = 0  

        if self.pos is not None:
            pr, pc = self.pos
            display_grid[pr, pc] = 0.5 

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.img = self.ax.imshow(display_grid, cmap=plt.cm.binary, interpolation='nearest')
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            self.img.set_data(display_grid)

        plt.pause(0.1)