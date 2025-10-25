from mazelib import Maze
from mazelib.generate.Prims import Prims
import matplotlib.pyplot as plt
import random as ran
import numpy as np


m = Maze()
m.generator = Prims(10,15)
m.generate()
m.generate_entrances()
m.grid[m.start] = 0
m.grid[m.end] = 0

def showMaze(grid):
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation='nearest')
    plt.xticks([]), plt.yticks([])
    plt.show()

showMaze(m.grid)