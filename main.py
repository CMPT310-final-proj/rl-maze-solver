from src.env import Env
from src.q_learning import train_q_learning, run_greedy

def main():
    env = Env()
    Q = train_q_learning(env)

    run_greedy(env, Q)

main()