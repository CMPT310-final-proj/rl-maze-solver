import numpy as np
import random as ran

def q_learning(env, episodes, max_steps, alpha, gamma):
    grid = env.grid
    n_rows, n_cols = grid.shape
    n_actions = env.action_space.n

    Q = np.zeros((n_rows, n_cols, 2, n_actions))

    def epsilon(ep):
        return max(0.1, 1.0 - ep / episodes)

    for ep in range(episodes):
        observation, _ = env.reset()
        row, col, has_treasure = observation
        eps = epsilon(ep)

        for step in range(max_steps):
            if ran.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[row, col, has_treasure]))

            new_observation, reward, terminated, _ = env.step(action)
            new_row, new_col, new_has_treasure = new_observation

            if terminated:
                target = reward
            else:
                V = np.max(Q[new_row, new_col, new_has_treasure])
                target = reward + gamma * V

            Q[row, col, has_treasure, action] += alpha * (target - Q[row, col, has_treasure, action])

            row, col, has_treasure = new_row, new_col, new_has_treasure

            if terminated:
                break

    return Q

# function that renders the agents learned policy
def run_greedy(env, Q, max_steps=300):
    observation, _ = env.reset()
    row, col, has_treasure = observation

    for t in range(max_steps):
        action = int(np.argmax(Q[row, col, has_treasure]))
        observation, reward, terminated, _ = env.step(action)
        row, col, has_treasure = observation
        env.render()

        if terminated:
            print(f"Solved in {t+1} steps!")
            break