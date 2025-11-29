# bustersAgents.py
# ----------------
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


import util
from util import raiseNotDefined
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics:
    "Plaatshouder voor graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basis inferentiemodule voor gebruik met het toetsenbord.
    """
    def initializeUniformly(self, gameState):
        "Begin met een uniforme verdeling over spookposities."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        noisyDistance = observation
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if noisyDistance != None and \
                    busters.getObservationProbability(noisyDistance, trueDistance) > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "Een agent die zijn overtuigingen over spookposities bijhoudt en weergeeft."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        try:
            inferenceType = util.lookup(inference, globals())
        except Exception:
            inferenceType = util.lookup('inference.' + inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initialiseert overtuigingen en inferentiemodules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Verwijdert de spooktoestanden uit de gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Werkt overtuigingen bij en kiest vervolgens een actie op basis van bijgewerkte overtuigingen."
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.observeEnable:
                inf.observe(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "Standaard stopt een BustersAgent gewoon. Dit moet worden overschreven."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "Een agent bestuurd door het toetsenbord die overtuigingen over spookposities weergeeft."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions

class GreedyBustersAgent(BustersAgent):
    "Een agent die op de dichtstbijzijnde spook afstormt."

    def registerInitialState(self, gameState: busters.GameState):
        "Berekent vooraf de afstand tussen elk tweetal punten."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ########### ########### ###########
    ########### QUESTION 8  ###########
    ########### ########### ###########

    def chooseAction(self, gameState: busters.GameState):
        """
        Berekent eerst de meest waarschijnlijke positie van elke spook die
        nog niet is gevangen, en kiest vervolgens een actie die
        Pacman het dichtst bij de dichtstbijzijnde spook brengt (volgens mazeDistance!).
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]

        # Vind de meest waarschijnlijke positie van elke spook
        mostLikelyPositions = [dist.argMax() for dist in livingGhostPositionDistributions]

        # Als er geen levende spoken zijn, retourneer een willekeurige legale actie
        if not mostLikelyPositions:
            return random.choice(legal)

        # Vind de dichtstbijzijnde spook
        closestGhostDistance = float('inf')
        closestGhostPosition = None

        for ghostPos in mostLikelyPositions:
            distance = self.distancer.getDistance(pacmanPosition, ghostPos)
            if distance < closestGhostDistance:
                closestGhostDistance = distance
                closestGhostPosition = ghostPos

        # Kies de actie die de afstand tot de dichtstbijzijnde spook minimaliseert
        bestAction = None
        minDistance = float('inf')

        for action in legal:
            successorPosition = Actions.getSuccessor(pacmanPosition, action)
            distance = self.distancer.getDistance(successorPosition, closestGhostPosition)
            if distance < minDistance:
                minDistance = distance
                bestAction = action

        return bestAction
