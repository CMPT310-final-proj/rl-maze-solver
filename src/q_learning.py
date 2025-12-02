import numpy as np
import random as ran

def q_learning(env, episodes, max_steps, alpha, gamma):
    grid = env.grid
    n_rows, n_cols = grid.shape
    n_actions = env.action_space.n

                # row, col, has_treasure(0/1), action
    Q = np.zeros((n_rows, n_cols, 2, n_actions))

    # exploration rate
    def epsilon(ep):
        return max(0.1, 1.0 - ep / episodes)

    # for holding metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for ep in range(episodes):
        observation, _ = env.reset()
        row, col, has_treasure = observation
        has_treasure = int(has_treasure) 
        eps = epsilon(ep)

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            if ran.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[row, col, has_treasure]))

            new_observation, reward, terminated, _ = env.step(action)
            new_row, new_col, new_has_treasure = new_observation
            new_has_treasure = int(new_has_treasure)

            # q learning update
            if terminated:
                target = reward
            else:
                V = np.max(Q[new_row, new_col, new_has_treasure])
                target = reward + gamma * V

            Q[row, col, has_treasure, action] += alpha * (
                target - Q[row, col, has_treasure, action]
            )

            # move to new state
            row, col, has_treasure = new_row, new_col, new_has_treasure

            # metrics update
            total_reward += reward
            steps += 1

            if terminated:
                success_count += 1
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    # return q and metrics
    return Q, episode_rewards, episode_lengths, success_count

# function that renders the agent's learned policy
def run_greedy(env, Q, max_steps):
    observation, _ = env.reset()
    row, col, has_treasure = observation
    has_treasure = int(has_treasure)

    for t in range(max_steps):
        action = int(np.argmax(Q[row, col, has_treasure]))
        observation, reward, terminated, _ = env.step(action)
        row, col, has_treasure = observation
        has_treasure = int(has_treasure)

        env.render()

        if terminated:
            env.ax.text(
                0.5, 0.5,
                f"SOLVED IN {t + 1} STEPS!",
                transform=env.ax.transAxes,
                fontsize=20,
                fontweight='bold',
                color='red',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
            )
            env.fig.canvas.draw()
            env.fig.canvas.flush_events()
            return True, t + 1

    # return these when q learning didn't converge to an optimal policy
    return False, max_steps