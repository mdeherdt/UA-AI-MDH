
from util import PriorityQueue


def reconstruct_path(came_from, goal_state):
    path = []
    s = goal_state
    while s in came_from:
        s, move = came_from[s]
        path.append(move)
    path.reverse()
    return path
def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def astar(problem, heuristic= nullHeuristic):
    "*** YOUR CODE HERE ***"

    start_state = problem.getStartState()
    states = PriorityQueue()

    # Track best known g-costs
    best_cost = {start_state: 0}

    # Store g in the tuple; priority = g + h
    states.push((start_state, [], 0), heuristic(start_state, problem))

    while not states.isEmpty():
        state, path, cost = states.pop()

        # Skip stale entries (we already found a cheaper path to this state)
        if cost > best_cost.get(state, float('inf')):
            continue

        if problem.isGoalState(state):
            print(state)
            return path

        for new_state, action, step_cost in problem.getSuccessors(state):
            new_cost = cost + step_cost
            if new_cost < best_cost.get(new_state, float('inf')):
                best_cost[new_state] = new_cost
                pred_cost = new_cost + heuristic(new_state, problem)
                states.push((new_state, path + [action], new_cost), pred_cost)

    return []  # no solution

def ucs(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    states = PriorityQueue()
    states.push((start_state, [], 0, set()), 0)
    while not states.isEmpty():
        state, path, total_cost, visited = states.pop()

        # Skip if we already expanded this state
        if state in visited:
            continue
        visited.add(state)

        # Goal test on the node we popped (handles start-goal too)
        if problem.isGoalState(state):
            return path

        # Expand
        for new_state, action, cost in problem.getSuccessors(state):
            if new_state not in visited:
                states.push((new_state, path + [action], cost + total_cost, visited), total_cost + cost)
    return []  # no solution