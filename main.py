from src.env import Env
from src.q_learning import train_q_learning, run_greedy
import matplotlib.pyplot as plt

EPISODES = 1000
MAX_STEPS = 300
ALPHA = 0.1
GAMMA = 0.95

def main():
    env = Env()
    Q = train_q_learning(env, EPISODES, MAX_STEPS, ALPHA, GAMMA)

    # show the learned policy
    run_greedy(env, Q)

if __name__ == "__main__":
    main()
