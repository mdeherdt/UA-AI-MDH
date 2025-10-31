# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        closestGhostDist = float('inf')
        for ghost_state in newGhostStates:
            dist = util.manhattanDistance(newPos, ghost_state.getPosition())
            closestGhostDist = min(closestGhostDist, dist)


        if closestGhostDist <= 1:
            return float('-inf')


        foodList = newFood.asList()

        if not foodList:
            return float('inf')

        closestFoodDist = float('inf')
        for food_pos in foodList:
            dist = util.manhattanDistance(newPos, food_pos)
            closestFoodDist = min(closestFoodDist, dist)


        score = successorGameState.getScore()

        score += 1.0 / closestFoodDist

        score -= 1.0 / closestGhostDist

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        best_action = None
        max_value = float('-inf')
        pacman_agent_index = 0

        legal_actions = gameState.getLegalActions(pacman_agent_index)

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(pacman_agent_index, action)

            # Start de recursie voor de *eerste geest* (agent 1) op diepte 0
            # De score wordt bepaald door de 'min' speler (het eerste spook)
            score = self._minimax(successor_state, 1, 0)

            if score > max_value:
                max_value = score
                best_action = action

        return best_action

    def _minimax(self, state: GameState, agentIndex: int, depth: int):

        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legal_actions = state.getLegalActions(agentIndex)

        if not legal_actions:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            max_value = float('-inf')
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = self._minimax(successor, 1, depth)
                max_value = max(max_value, value)
            return max_value


        else:  # MIN-speler (Spook)
            min_value = float('inf')
            next_agent_index = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth

            if next_agent_index == 0:
                next_depth = depth + 1

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = self._minimax(successor, next_agent_index, next_depth)
                min_value = min(min_value, value)

            return min_value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        best_action = None
        max_value = float('-inf')
        pacman_agent_index = 0

        legal_actions = gameState.getLegalActions(pacman_agent_index)

        alfa = float('-inf')
        beta = float('inf')

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(pacman_agent_index, action)

            score = self._alpha_beta_search(successor_state, 1, 0, alfa, beta)

            if score > max_value:
                max_value = score
                best_action = action

            alfa = max(alfa, max_value)


        return best_action

    def _alpha_beta_search(self, state: GameState, agentIndex: int, depth: int, alpha: float, beta: float):

        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legal_actions = state.getLegalActions(agentIndex)

        if not legal_actions:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            max_value = float('-inf')

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)

                value = self._alpha_beta_search(successor, 1, depth, alpha, beta)

                max_value = max(max_value, value)

                alpha = max(alpha, max_value)

                if beta < alpha:
                    break


            return max_value

        else:
            min_value = float('inf')
            next_agent_index = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth

            if next_agent_index == 0:
                next_depth = depth + 1

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)

                value = self._alpha_beta_search(successor, next_agent_index, next_depth, alpha, beta)

                min_value = min(min_value, value)

                beta = min(beta, min_value)

                if beta < alpha:
                    break

            return min_value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        best_action = None
        max_value = float('-inf')
        pacman_agent_index = 0

        legal_actions = gameState.getLegalActions(pacman_agent_index)

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(pacman_agent_index, action)
            score = self._expectimax(successor_state, 1, 0)
            if score > max_value:
                max_value = score
                best_action = action
        return best_action

    def _expectimax(self, state: GameState, agentIndex: int, depth: int):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        num_agents = state.getNumAgents()
        legal_actions = state.getLegalActions(agentIndex)

        if not legal_actions:
            return self.evaluationFunction(state)

        if agentIndex == 0:
            max_value = float('-inf')
            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = self._expectimax(successor, 1, depth)
                max_value = max(max_value, value)
            return max_value
        else:
            total_value = 0
            next_agent_index = (agentIndex + 1) % state.getNumAgents()
            next_depth = depth

            if next_agent_index == 0:
                next_depth = depth + 1

            for action in legal_actions:
                successor = state.generateSuccessor(agentIndex, action)
                value = self._expectimax(successor, next_agent_index, next_depth)
                total_value += value
            expected_value = total_value / len(legal_actions)

            return expected_value


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    """
    pac = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    value = float(currentGameState.getScore())

    value -= 4.0 * len(food)

    if food:
        d_food = min(manhattanDistance(pac, f) for f in food)
        value += 3.0 / (d_food + 1.0)

    danger_close = False
    for g in ghosts:
        d = manhattanDistance(pac, g.getPosition())
        if g.scaredTimer > 0:
            value += 10.0 / (d + 1.0)
            if d == 0:
                value += 50.0  # Hextra bonus: spook gegeten
        else:
            if d <= 1:
                value -= 1000.0
                danger_close = True
            else:
                value -= 5.0 / d

    if capsules:
        d_cap = min(manhattanDistance(pac, c) for c in capsules)
        if danger_close:
            value += 15.0 / (d_cap + 1.0)

    return value

# Abbreviation
better = betterEvaluationFunction
