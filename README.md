# rl-maze-solver

## Project Overview

This project implements a **Reinforcement Learning (RL)** agent that learns to solve **procedurally generated mazes**.

The environment includes:

- **Start position**
- **Goal**
- **Treasure** that must be collected before exiting
- **Random perfect mazes** generated using **Prim’s Algorithm**

The agent interacts with a **custom Gymnasium-style environment** and learns an optimal navigation strategy using **Tabular Q-Learning**.

After training, a **greedy policy** is used to solve the maze, and the final path is visualized with **start, goal, treasure, and agent markers**.

---

## Features

### Procedural Maze Generation
- Mazes are generated using a randomized version of **Prim’s Algorithm**
- Guaranteed:
  - Fully connected mazes  
  - No loops  
  - Valid start/goal placement  

---

### Custom RL Environment
Gymnasium-compatible environment with:

**State:**  
`(row, col, has_treasure)`

**Actions:**  
- Up  
- Down  
- Left  
- Right  

**Reward Structure:**
- `-1` per step  
- `-10` for hitting walls  
- `+50` for collecting treasure  
- `+100` for reaching the goal *with treasure*  
- `-5` for reaching the goal *without treasure*  

---

### Q-Learning Agent
- **Tabular Q-table**
- **ε-greedy exploration**
- **Linear epsilon decay**
- **Q-value updates per step**
- Tracks:
  - Episode length  
  - Total reward  

---

### Visualized Maze Solver
When the agent reaches the goal, the maze displays:

### ** SOLVED IN XX STEPS!**

With a legend showing:

- 🟦 **Start**  
- 🟩 **Goal**  
- 🟨 **Treasure**  
- 🟥 **Agent**

---

### Training Graphs
After the maze visualization, two plots appear:

- **Episode Rewards**
- **Episode Lengths**

These help diagnose learning efficiency and convergence.

---

### BFS Baseline Comparison
A Breadth-First Search (BFS) baseline is used to compare optimality:

- BFS shortest-path **steps**
- Agent’s greedy-policy **steps**
- **Nodes expanded**
- **Difference in steps (Agent – BFS)**

---

## Project Structure
```
rl-maze-solver/
│
├── main.py # Run this file
├── src/
│ ├── env.py
│ ├── mazeGenerator.py 
│ ├── q_learning.py 
│ └── search.py 
│ 
│
├── README.md
```
## How to Run

### Install dependencies

```bash
pip install matplotlib numpy gymnasium
```

```bash 
python main.py
```
OR
```bash
py main.py
```
## Output Windows

You will see:

- Maze solver with live agent movement
- Reward curve
- Episode length graph

Close the maze window at the end to exit the program.