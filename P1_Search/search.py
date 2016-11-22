# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    This class contains information regarding to a node on the path we're searching.
    Contains:
        state - (x,y) tuple coordinate
        path - list of directions ('Start', 'North', 'South', 'East', 'West')
        pathCost - sum of stepCosts for each action (direction) taken

    """
    def __init__(self, state, path, pathCost = 0):
        self.state = state
        self.path = path
        self.pathCost = pathCost

start = "Start"

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def minCostHeuristic(node, problem=None):
    return node.pathCost

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def preparePathForResponse(path):
    if path:
        path.remove(start)
    else:
        path = []

    return path


def recursiveDFSHelper(problem, currentNode, visited = {}):
    # Not necessary, but recursion is always fun
    currentState = currentNode.state

    visited[currentState] = True;

    if problem.isGoalState(currentState):
        return [currentNode.direction]

    for successorTuple in problem.getSuccessors(currentState):
        successorNode = Node(successorTuple[0], successorTuple[1])
        if visited.get(successorNode.state):
            continue

        path = recursiveDFSHelper(problem, successorNode, visited)
        if path:
            path = [currentNode.direction] + path
            return path
    return None

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    startState = problem.getStartState()
    #print "Start:", startState

    startNode = Node(startState, [start])

    path = searchHelper(problem, util.Stack(), startNode)

    return preparePathForResponse(path)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    startState = problem.getStartState()

    startNode = Node(startState, [start])
    queue = util.Queue()

    path = searchHelper(problem, queue, startNode)

    if path:
        path.remove(start)
    else:
        path = []

    return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    startState = problem.getStartState()

    startNode = Node(startState, [start])
    queue = util.PriorityQueueWithFunction(minCostHeuristic, problem)

    path = searchHelper(problem, queue, startNode)

    return preparePathForResponse(path)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startState = problem.getStartState()

    startNode = Node(startState, [start])

    aStarCostFunction = lambda node, problem: node.pathCost + heuristic(node.state, problem)

    queue = util.PriorityQueueWithFunction(aStarCostFunction, problem)

    path = searchHelper(problem, queue, startNode)

    return preparePathForResponse(path)


def searchHelper(problem, queue, startNode):
    visited = []
    path = []

    queue.push(startNode)

    while(not queue.isEmpty()):
        currentNode = queue.pop()
        currentState = currentNode.state

        if currentState in visited:
            continue
        visited.append(currentState)

        if problem.isGoalState(currentState):
            return currentNode.path

        for successorTuple in problem.getSuccessors(currentState):
            successorState = successorTuple[0]
            successorDirection = successorTuple[1]
            successorStepCost = successorTuple[2]
            successorPath = currentNode.path[:]
            successorPath.append(successorDirection)
            successorPathCost = currentNode.pathCost + successorStepCost

            successorNode = Node(successorState, successorPath, successorPathCost)

            if successorState in visited:
                continue

            queue.push(successorNode)

    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
