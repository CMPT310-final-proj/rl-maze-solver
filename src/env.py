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


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        return np.array(self.pos, dtype=np.int32), {}

    def step(self, action):
        r, c = self.pos
        dr, dc = [(-1,0), (1,0), (0,-1), (0,1)][int(action)]
        nr, nc = r + dr, c + dc

        reward = -1.0
        terminated = False

        if (0 <= nr < self.grid.shape[0] and
            0 <= nc < self.grid.shape[1] and
            self.grid[nr, nc] == 0):
            self.pos = (nr, nc)
        else:
            reward = -10.0 

        if self.pos == self.goal:
            reward = 100.0
            terminated = True

        return np.array(self.pos, dtype=np.int32), reward, terminated, {}

    def render(self):
        return