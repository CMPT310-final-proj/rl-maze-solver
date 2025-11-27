from collections import deque

def bfs(grid, start, goal):
    height, width = grid.shape

    queue = deque([start])
    parent = {start: None}
    nodes_expanded = 0

    directions = [
        (-1, 0),  # up
        (1, 0),   # down
        (0, -1),  # left
        (0, 1),   # right
    ]

    while queue:
        current_row, current_col = queue.popleft()
        nodes_expanded += 1
        # reached the goal
        if (current_row, current_col) == goal:
            break
        # explore neighbors
        for d_row, d_col in directions:
            next_row = current_row + d_row
            next_col = current_col + d_col
            # check bounds and that the cell is free
            if 0 <= next_row < height and 0 <= next_col < width:
                if grid[next_row, next_col] == 0:
                    if (next_row, next_col) not in parent:  # not visited
                        parent[(next_row, next_col)] = (current_row, current_col)
                        queue.append((next_row, next_col))

    # if we never reached the goal
    if goal not in parent:
        return None, nodes_expanded

    # reconstruct the shortest path
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return path, nodes_expanded


def maze_bfs(grid, start, treasure, goal):
    path_to_treasure, nodes1 = bfs(grid, start, treasure) # start to treasure
    path_to_goal, nodes2 = bfs(grid, treasure, goal) # from goal to treasure

    if path_to_treasure is None or path_to_goal is None:
        return None, nodes1 + nodes2

    # combine paths without repeating the treasure node
    full_path = path_to_treasure + path_to_goal[1:]

    return full_path, nodes1 + nodes2
