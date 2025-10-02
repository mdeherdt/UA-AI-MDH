from typing import List, Tuple


class MissionariesCannibals:
    def __init__(self):
        """Implement a state space"""
        "*** YOUR CODE HERE ***"

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



    def _valid_state(self, s) -> bool:
        """Return whether a state is valid:
        - the missionaries are nowhere outnumbered by the cannibals
        - There are in total always 3 missionaries and 3 cannibals"""
        "*** YOUR CODE HERE ***"

