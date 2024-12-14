import time
import heapq
from copy import deepcopy

def is_solvable(state):
    """Check if a given 8-puzzle state is solvable."""
    flat_state = [num for row in state for num in row if num != 0]
    inversions = sum(1 for i in range(len(flat_state)) for j in range(i + 1, len(flat_state)) if flat_state[i] > flat_state[j])
    return inversions % 2 == 0

def find_blank_tile(state):
    for i, row in enumerate(state):
        for j, tile in enumerate(row):
            if tile == 0:
                return i, j

def generate_successors(state):
    """Generate successors for a given state by moving the blank tile."""
    successors = []
    row, col = find_blank_tile(state)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(state) and 0 <= new_col < len(state[0]):
            new_state = deepcopy(state)
            new_state[row][col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[row][col]
            successors.append(new_state)

    return successors

def manhattan_distance(state, goal):
    """Calculate Manhattan distance as a heuristic."""
    distance = 0
    for i in range(len(state)):
        for j in range(len(state[0])):
            if state[i][j] != 0:
                flat_goal = [num for row in goal for num in row]
                target_x, target_y = divmod(flat_goal.index(state[i][j]), len(state))

                distance += abs(target_x - i) + abs(target_y - j)
    return distance

def misplaced_tiles(state, goal):
    """Calculate the number of misplaced tiles as a heuristic."""
    return sum(1 for i in range(len(state)) for j in range(len(state[0])) if state[i][j] != 0 and state[i][j] != goal[i][j])

def a_star(initial_state, goal_state, heuristic):
    """A* search algorithm."""
    start_time = time.time()
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  # (priority, state, path)
    explored = set()
    nodes_visited = 0

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        nodes_visited += 1

        if current_state == goal_state:
            return path, nodes_visited, time.time() - start_time

        explored.add(str(current_state))

        for successor in generate_successors(current_state):
            if str(successor) not in explored:
                cost = len(path) + 1
                priority = cost + heuristic(successor, goal_state)
                heapq.heappush(frontier, (priority, successor, path + [successor]))

    return None, nodes_visited, time.time() - start_time

def greedy_search(initial_state, goal_state, heuristic):
    """Greedy search algorithm."""
    start_time = time.time()
    frontier = []
    heapq.heappush(frontier, (0, initial_state, []))  # (priority, state, path)
    explored = set()
    nodes_visited = 0

    while frontier:
        _, current_state, path = heapq.heappop(frontier)
        nodes_visited += 1

        if current_state == goal_state:
            return path, nodes_visited, time.time() - start_time

        explored.add(str(current_state))

        for successor in generate_successors(current_state):
            if str(successor) not in explored:
                priority = heuristic(successor, goal_state)
                heapq.heappush(frontier, (priority, successor, path + [successor]))

    return None, nodes_visited, time.time() - start_time

def minimax(state, depth, maximizing, goal_state):
    """Minimax algorithm for the 8-puzzle."""
    if state == goal_state or depth == 0:
        return misplaced_tiles(state, goal_state), []

    if maximizing:
        max_eval = float('-inf')
        best_path = []
        for successor in generate_successors(state):
            eval_score, path = minimax(successor, depth - 1, False, goal_state)
            if eval_score > max_eval:
                max_eval = eval_score
                best_path = [successor] + path
        return max_eval, best_path
    else:
        min_eval = float('inf')
        best_path = []
        for successor in generate_successors(state):
            eval_score, path = minimax(successor, depth - 1, True, goal_state)
            if eval_score < min_eval:
                min_eval = eval_score
                best_path = [successor] + path
        return min_eval, best_path

def parse_input(puzzle_string):
    """Parse the input string into a 2D list."""
    if puzzle_string == 's':
        return [
            [4, 1, 3],
            [2, 0, 6],
            [7, 5, 8]
        ]
    else:
        numbers = list(map(int, puzzle_string.split()))
        size = int(len(numbers) ** 0.5)
        return [numbers[i * size:(i + 1) * size] for i in range(size)]

# Initial and goal states
puzzle_input = input("Enter the puzzle (e.g., '4 1 3 2 0 6 7 5 8') (Type s to use the standard puzzle): ")
initial_state = parse_input(puzzle_input)

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

if not is_solvable(initial_state):
    print("This puzzle is not solvable.")
else:
    print("\nRunning A* with Manhattan distance...")
    path, nodes, duration = a_star(initial_state, goal_state, manhattan_distance)
    print(f"Manhattan Distance - Path Length: {len(path)}, Nodes Visited: {nodes}, Time {duration}")

    print("\nRunning A* with Misplaced Tiles...")
    path, nodes, duration = a_star(initial_state, goal_state, misplaced_tiles)
    print(f"Misplaced Tiles - Path Length: {len(path)}, Nodes Visited: {nodes}, Time {duration}")

    print("\nRunning Greedy Search with Manhattan distance...")
    path, nodes, duration = greedy_search(initial_state, goal_state, manhattan_distance)
    print(f"Greedy Search (Manhattan) - Path Length: {len(path)}, Nodes Visited: {nodes}, Time {duration}")

    print("\nRunning Minimax (depth 12)...")
    score, path = minimax(initial_state, 12, True, goal_state)
    print(f"Minimax - Path Length: {len(path)}, Score: {score}")
