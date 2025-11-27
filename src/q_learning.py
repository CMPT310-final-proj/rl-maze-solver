import numpy as np
import random as ran

def train_q_learning(env, episodes=600, max_steps=300, alpha=0.1, gamma=0.95):
    grid = env.grid
    n_rows, n_cols = grid.shape
    n_actions = env.action_space.n

    Q = np.zeros((n_rows, n_cols, n_actions))

    def epsilon(ep):
        return max(0.1, 1.0 - ep / episodes)

    for ep in range(episodes):
        position, _ = env.reset()
        row, col = position
        eps = epsilon(ep)

        for step in range(max_steps):
            if ran.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[row, col]))

            new_position, reward, terminated, _ = env.step(action)
            new_row, new_col = new_position

            if terminated:
                target = reward
            else:
                V = np.max(Q[new_row, new_col])
                target = reward + gamma * V

            Q[row, col, action] += alpha * (target - Q[row, col, action])

            row, col = new_row, new_col

            if terminated:
                break

    return Q

# function that renders the agents learn policy
def run_greedy(env, Q, max_steps=300):
    position, _ = env.reset()
    row, col = position

    for t in range(max_steps):
        action = int(np.argmax(Q[row, col]))
        position, reward, terminated, _ = env.step(action)
        row, col = position
        env.render()

        if terminated:
            print(f"Solved in {t+1} steps!")
            break