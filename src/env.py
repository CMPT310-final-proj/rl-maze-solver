import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.mazeGenerator import generateMaze
import random as ran 


class Env(gym.Env):
    def __init__(self):
        super().__init__()

        self.grid, self.start, self.goal = generateMaze()
        self.position = None
        self.has_treasure = False

        free_cells = [
            (row, col)
            for row in range(self.grid.shape[0])
            for col in range(self.grid.shape[1])
            if self.grid[row, col] == 0
        ]

        # making sure treasure doesn't end up in the entrance/exit
        self._treasure_candidates = [
            cell for cell in free_cells
            if cell not in (self.start, self.goal)
        ]

        # treasure location
        self.treasure = ran.choice(self._treasure_candidates)


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
        self.position = self.start

        # reset treasure state each episode
        self.has_treasure = False

        return np.array(self.position, dtype=np.int32), {}

    def step(self, action):
        row, col = self.position
        d_row, d_col = [(-1,0), (1,0), (0,-1), (0,1)][int(action)]
        new_row, new_col = row + d_row, col + d_col

        reward = -1.0
        terminated = False

        # movement
        if (0 <= new_row < self.grid.shape[0] and
            0 <= new_col < self.grid.shape[1] and
            self.grid[new_row, new_col] == 0):
            self.position = (new_row, new_col)
        else:
            # reward for hitting walls
            reward = -10.0 

        # reward for taking treasure
        if (not self.has_treasure) and (self.position == self.treasure):
            self.has_treasure = True
            reward += 50.0

        # reward for reaching goal
        if self.position == self.goal:
            reward = 100.0
            terminated = True

        return np.array(self.position, dtype=np.int32), reward, terminated, {}

    def render(self):
        height, width = self.grid.shape
        img = np.zeros((height, width, 3), dtype=float)

        # free space = grey
        img[self.grid == 0] = [0.8, 0.8, 0.8]

        # walls = black
        img[self.grid == 1] = [0, 0, 0]

        # treasure = yellow
        if not self.has_treasure:
            treasure_row, treasure_col = self.treasure
            img[treasure_row, treasure_col] = [1, 1, 0]

        # goal = green
        goal_row, goal_col = self.goal
        img[goal_row, goal_col] = [0, 1, 0]

        # agent = red
        if self.position is not None:
            agent_row, agent_col = self.position
            img[agent_row, agent_col] = [1, 0, 0]

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.img = self.ax.imshow(img)
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        else:
            self.img.set_data(img)

        plt.pause(0.1)
