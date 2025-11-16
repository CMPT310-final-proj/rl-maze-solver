from src.env import Env

def main():
    env = Env()
    obs, info = env.reset()
    print("Initial:", obs, "Goal:", env.goal)
    for a in range(4):
        o, r, done, info = env.step(a)
        print(f"Action {a} -> Obs={o}, Reward={r}, Done={done}")
        env.render()

main()