import numpy as np

class MazeEnv:
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.position = start
        self.rows, self.cols = maze.shape

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        r, c = self.position
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        # Check boundaries / walls
        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze[nr][nc] == 0:
            self.position = (nr, nc)
            reward = -1
        else:
            reward = -5

        done = self.position == self.end
        if done:
            reward = 10

        return self.position, reward, done
    
    def reset(self):
        self.position = self.start
        return self.position
