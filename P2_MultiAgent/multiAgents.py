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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (foodGrid) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth*gameState.getNumAgents(), self.index)[1]

    def minimax(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), 'STOP')

        nextDepth = depth - 1
        nextIndex = (agentIndex + 1) % gameState.getNumAgents()

        if agentIndex == 0:
            bestValue = -float('inf')
            bestAction = 'STOP'
            for action in gameState.getLegalActions(agentIndex):
                nextValue, nextAction = self.minimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex)
                if nextValue > bestValue:
                    bestValue = nextValue
                    bestAction = action
        else:
            bestValue = float('inf')
            bestAction = 'STOP'
            for action in gameState.getLegalActions(agentIndex):
                nextValue, nextAction = self.minimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex)
                if nextValue < bestValue:
                    bestValue = nextValue
                    bestAction = action

        return bestValue, bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta(gameState, self.depth*gameState.getNumAgents(), self.index)[1]

    def alphaBeta(self, gameState, depth, agentIndex, alpha=-float('inf'), beta=float('inf')):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), 'STOP')

        nextDepth = depth - 1
        nextIndex = (agentIndex + 1) % gameState.getNumAgents()

        if agentIndex == 0:
            bestValue = -float('inf')
            bestAction = 'STOP'
            for action in gameState.getLegalActions(agentIndex):
                nextValue, nextAction = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex, alpha, beta)
                if nextValue > bestValue:
                    bestValue = nextValue
                    bestAction = action
                if nextValue > alpha:
                    alpha = nextValue
                if beta < alpha:
                    break
        else:
            bestValue = float('inf')
            bestAction = 'STOP'
            for action in gameState.getLegalActions(agentIndex):
                nextValue, nextAction = self.alphaBeta(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex, alpha, beta)
                if nextValue < bestValue:
                    bestValue = nextValue
                    bestAction = action
                if nextValue < beta:
                    beta = nextValue
                if beta < alpha:
                    break

        return bestValue, bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax(gameState, self.depth * gameState.getNumAgents(), self.index)[1]

    def expectimax(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), 'STOP')

        nextDepth = depth - 1
        nextIndex = (agentIndex + 1) % gameState.getNumAgents()

        if agentIndex == 0:
            bestValue = -float('inf')
            bestAction = 'STOP'
            for action in gameState.getLegalActions(agentIndex):
                nextValue, nextAction = self.expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex)
                if nextValue > bestValue:
                    bestValue = nextValue
                    bestAction = action
            return bestValue, bestAction
        else:
            expectedValue = 0
            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                nextValue, nextAction = self.expectimax(gameState.generateSuccessor(agentIndex, action), nextDepth, nextIndex)
                expectedValue += nextValue
            expectedValue = expectedValue/float(len(legalActions))
            return expectedValue, ""


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacmanPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()

    if currentGameState.isWin():
        return float("inf")
    elif currentGameState.isLose():
        return -float("inf")

    minFoodDistance = float("inf")
    if foodList:
        minFoodDistance = min(map(lambda foodPos: manhattanDistance(pacmanPosition, foodPos), foodList))

    ghosts = currentGameState.getGhostStates()

    # Filter threatening ghosts i.e. ghosts not currently scared
    threateningGhosts = filter(lambda ghost: ghost.scaredTimer == 0, ghosts)

    # Filter scared ghosts that are close enough to reach within scared timer, otherwise ignore them
    scaredGhosts = filter(lambda ghost: manhattanDistance(ghost.getPosition(), pacmanPosition) < ghost.scaredTimer,
                          ghosts)

    minThreateningGhostDistance = float("inf")
    minScaredGhostDistance = float("inf")

    if threateningGhosts:
        minThreateningGhostDistance = min(
            map(lambda ghost: manhattanDistance(ghost.getPosition(), pacmanPosition), threateningGhosts))

    if scaredGhosts:
        minScaredGhostDistance = min(
            map(lambda ghost: manhattanDistance(ghost.getPosition(), pacmanPosition), scaredGhosts))

    foodCount = currentGameState.getNumFood()
    capsules = currentGameState.getCapsules()
    capsuleCount = len(capsules)

    minCapsuleDistance = float("inf")
    if capsules:
        minCapsuleDistance = min(map(lambda capsule: manhattanDistance(capsule, pacmanPosition), capsules))

    gameScore = currentGameState.getScore()
    foodCountScore = foodCount * -10
    foodDistanceScore = 1.0 / minFoodDistance * 3
    capsuleCountScore = capsuleCount * -25
    capsuleDistanceScore = 1.0 / minCapsuleDistance * 5
    scaredGhostScore = 1.0 / minScaredGhostDistance * 8
    if minThreateningGhostDistance < 2:
        threatGhostScore = minThreateningGhostDistance * -20
    else:
        threatGhostScore = 1.0 / minThreateningGhostDistance * -1

    "*** YOUR CODE HERE ***"
    return gameScore + foodCountScore + foodDistanceScore + capsuleCountScore + capsuleDistanceScore + threatGhostScore + scaredGhostScore


# Abbreviation
better = betterEvaluationFunction