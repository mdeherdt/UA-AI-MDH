from typing import List, Tuple

L, R = 'L', 'R'

class RiverProblem:
    """State = (F, G, W, C), each in {'L','R'}. (start is {L,L,L,L})
    Actions are strings: 'FG', 'FW', 'FC', 'F' (deterministic order).
    Unit-cost per action.
    """
    def __init__(self):
        self._start = (L, L, L, L)
        self._actions = ['FG', 'FW', 'FC', 'F']

    def getStartState(self):
        "Return the start state"
        "*** YOUR CODE HERE ***"

    def isGoalState(self, state) -> bool:
        """Return bool whether the provided state is the goal state"""
        "*** YOUR CODE HERE ***"

    def getSuccessors(self, state) -> List[Tuple[tuple, str, int]]:
        """Return the successors of a state in the form of:
            A list of:
              a tuple (state)
              a string (action)
              an integer (the cost (just 1))"""
        "*** YOUR CODE HERE ***"

    def _is_valid(self, s) -> bool:
        """Return whether a state is valid:
        - the goat-wolf or goat-cabbage are not left together without the farmer
        """
        "*** YOUR CODE HERE ***"
