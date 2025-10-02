# State Space Exercise: BFS & DFS on Two Classic Puzzles

In this assignment you will **reuse your Pacman search algorithms** (BFS and DFS) to solve two famous riddles. The goal is to understand how to formalize puzzles as search problems and then solve them automatically.

---

## What you need to do

1. Copy your `depthFirstSearch` and `breadthFirstSearch` implementations from the Pacman assignment into `search.py`.  

2. Implement the **problem definitions** in the provided files:
   - `river_problem.py` — Farmer, Wolf, Goat, and Cabbage puzzle  
   - `missionaries_cannibals.py` — Missionaries & Cannibals puzzle  

   Each problem must define the Pacman search problem API:
   - `getStartState()`  
   - `isGoalState(state)`  
   - `getSuccessors(state)` → returns a list of `(successor, action, cost)`  
   - `_is_valid(state)`
3. Run `python run_search.py` to test your algorithms on both problems.  

---

## Exercise 1: River Riddle

A **farmer** is traveling with a **wolf**, a **goat**, and a **cabbage**. They come across a river with a small boat.

- The boat can carry the farmer **plus at most one** other (wolf, goat, or cabbage).  
- All four start on the **left bank** and must reach the **right bank**.  
- The farmer must always be in the boat (he can never send the boat alone).  

### Danger rules (invalid states)

- If the **goat** and **wolf** are left alone on one bank **without the farmer**, the wolf eats the goat → **Game Over**.  
- If the **goat** and **cabbage** are left alone on one bank **without the farmer**, the goat eats the cabbage → **Game Over**.  

### Task

- Define the **state representation** (e.g., `(farmer, goat, wolf, cabbage)` where each is `'L'` or `'R'`).  
- Implement `getSuccessors` so that each legal boat crossing generates a new valid state.  
- Use your BFS/DFS to find a solution sequence that brings everyone safely to the right bank.

### Run

```bash
python run_search.py
```

and check that BFS finds the shortest plan (7 steps).

---

## Exercise 2: Missionaries & Cannibals

Three **missionaries** and three **cannibals** are on the left bank of a river. They all need to cross to the right bank using a boat that can carry **1 or 2 people** (there always needs to be at least 1 person in the boat to travel (you can't just send the boat back without anyone in it).

### Danger rule

- On either bank, if missionaries are present, they must **not be outnumbered** by cannibals (otherwise they get eaten).  

### Task

- Define the **state representation**.
- Implement `getSuccessors` to generate all valid boat moves.
- Use BFS/DFS to find a safe sequence of crossings.

### Run

```bash
python run_search.py
```

and check that BFS finds the shortest plan (11 steps).
