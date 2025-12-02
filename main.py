from src.env import Env
from src.q_learning import q_learning, run_greedy
from src.search import maze_bfs
import matplotlib.pyplot as plt
import numpy as np

######################
# RUN THIS FILE ONLY #
######################

EPISODES = 5000
MAX_STEPS = 300
ALPHA = 0.1
GAMMA = 0.95

def main():
    env = Env()
    Q, episode_rewards, episode_lengths, success_count = q_learning(env, EPISODES, MAX_STEPS, ALPHA, GAMMA)

    print("\nTraining summary:")
    print(f"Episodes:                   {len(episode_rewards)}")
    print(f"Solved:                     {success_count}")
    print(f"Success rate:               {success_count / len(episode_rewards):.2%}")
    print(f"Average episode reward:         {np.mean(episode_rewards):.2f}")
    print(f"Average steps to complete episode: {np.mean(episode_lengths):.1f}")

    print("\nLearned policy:")
    solved, steps = run_greedy(env, Q, MAX_STEPS)
    if solved:
        print(f"Solved in {steps} steps")
    else:
        print("Policy did not learn to reach the goal")

    # optional graphs to visualize metrics

    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Episode rewards")
    plt.grid(True)
    plt.show(block=False)

    plt.figure()
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode lengths")
    plt.grid(True)
    plt.show(block=False)

    grid = env.grid
    start = env.start
    goal = env.goal
    treasure = env.treasure

    bfs_path, bfs_expanded = maze_bfs(grid, start, treasure, goal)
    if bfs_path is not None:
        bfs_steps = len(bfs_path) - 1

        print("\nBFS Summary:")
        print(f"BFS steps: {bfs_steps}")
        print(f"Nodes expanded: {bfs_expanded}")
    else:
        print("BFS could not find a path.")

    if bfs_path is not None:
        extra = steps - bfs_steps
        print(f"\nAgent vs BFS comparison")
        print(f"BFS # of steps: {bfs_steps}")
        print(f"Agent # of steps: {steps}")
        print(f"Difference in steps: {extra}")
    
    print("\nClose the maze window to exit the program.")
    plt.show()

if __name__ == "__main__":
    main()