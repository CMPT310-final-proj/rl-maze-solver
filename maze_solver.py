from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import numpy as np
import random


# Generate a maze using Prim's algorithm
def generate_maze(rows=5, cols=8):
    
    m = Maze()
    m.generator = Prims(rows, cols)
    m.generate()
    m.generate_entrances()
    m.grid[m.start] = 0
    m.grid[m.end] = 0
    return np.array(m.grid), m.start, m.end

class MazeEnv:
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.position = start
        self.rows, self.cols = maze.shape

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        # 0=Up, 1=Down, 2=Left, 3=Right
        r, c = self.position
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.maze[nr][nc] == 0:
            self.position = (nr, nc)
            reward = -1
        else:
            reward = -5

        done = self.position == self.end
        if done:
            reward = 10
        return self.position, reward, done

# A simple agent that chooses random directions
class RandomAgent:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
    def choose_action(self):
        return random.randint(0, self.n_actions - 1)


def run_simulation():
    # Generate maze and setup
    maze, start, end = generate_maze(10, 15)
    env = MazeEnv(maze, start, end)
    agent = RandomAgent()

    # Plot maze
    plt.ion()
    figure, maze_axes = plt.subplots()
    maze_axes.imshow(maze, cmap='gray')
    agent_marker, = maze_axes.plot(start[1], start[0], 'ro', markersize=8)
    maze_axes.plot(end[1], end[0], 'go', markersize=8)
    plt.title("Maze Solver")
    plt.pause(0.5)

    # Run agent
    total_reward = 0
    steps = 0
    for _ in range(200):
        action = agent.choose_action()
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        # Update visualization
        agent_marker.set_data([next_state[1]], [next_state[0]])
        plt.pause(0.1)

        if done:
            print(f"Goal reached in {steps} steps! Total reward: {total_reward}")
            break
    else:
        print(f"Agent did not reach goal. Total reward: {total_reward}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_simulation()