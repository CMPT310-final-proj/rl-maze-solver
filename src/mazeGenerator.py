from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import random as ran
import numpy as np

def generateMaze():
    m = Maze()
    m.generator = Prims(ran.randint(15,17), ran.randint(15,17))
    m.generate()
    m.generate_entrances()
    m.grid[m.start] = 0
    m.grid[m.end] = 0
    return np.array(m.grid), m.start, m.end
