from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import random as ran
import numpy as np

def generateMaze():
    m = Maze()
    m.generator = Prims(ran.randint(9,11), ran.randint(9,11))
    m.generate()
    m.generate_entrances()
    m.grid[m.start] = 0
    m.grid[m.end] = 0
    return np.array(m.grid), m.start, m.end

def showMaze(grid):
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()