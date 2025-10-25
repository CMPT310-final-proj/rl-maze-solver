import numpy as np
import matplotlib.pyplot as plt
from maze_generator import generate_maze
from maze_env import MazeEnv
from agent_random import RandomAgent

# Create maze
maze, start, end = generate_maze(10, 15)
env = MazeEnv(maze, start, end)
agent = RandomAgent()

# Visualize
plt.ion()
fig, ax = plt.subplots()
ax.imshow(maze, cmap='gray')
agent_marker, = ax.plot(start[1], start[0], 'ro')

state = env.reset()
for step in range(100):
    action = agent.choose_action()
    next_state, reward, done = env.step(action)
    print(f"Step {step}: {state} -> {next_state}, Reward={reward}")
    agent_marker.set_data([next_state[1]], [next_state[0]])
    plt.pause(0.2)
    state = next_state
    if done:
        print("🎉 Goal reached!")
        break

plt.ioff()
plt.show()
