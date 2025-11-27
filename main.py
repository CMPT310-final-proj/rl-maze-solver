from src.env import Env
from src.q_learning import q_learning, run_greedy
import matplotlib.pyplot as plt
import numpy as np

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

    # optional graphs to visualize metrics

    # plt.figure()
    # plt.plot(episode_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Total reward")
    # plt.title("Episode rewards")
    # plt.grid(True)
    # plt.show()

    # plt.figure()
    # plt.plot(episode_lengths)
    # plt.xlabel("Episode")
    # plt.ylabel("Steps")
    # plt.title("Episode lengths")
    # plt.grid(True)
    # plt.show()

    print("\nLearned policy:")
    solved, steps = run_greedy(env, Q, MAX_STEPS)
    if solved:
        print(f"Solved in {steps} steps")
    else:
        print("Did not learn to reach the goal")

if __name__ == "__main__":
    main()