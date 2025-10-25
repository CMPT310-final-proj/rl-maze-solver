import random

class RandomAgent:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions

    def choose_action(self):
        return random.randint(0, self.n_actions - 1)
