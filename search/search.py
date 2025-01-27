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


#######################################################
#                 Cost functions                      #
#######################################################


def backCost(item):
    """ item <state, actions, backwardCost> """
    return item[2]


def forwardCost(item, problem, heuristic):
    """Find forward cost which is heuristic"""

    # heuristic None means null so return Zero
    if heuristic is None:
        return 0

    return heuristic(item[0], problem)


def totalCost(item, problem, heuristic):
    """sum of forward cost and backward cost"""
    return backCost(item) + forwardCost(item, problem, heuristic)


#######################################################
#                  Search Algoritms                   #
#######################################################


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def searchAlgorithm(problem, fringe):
    # get start state
    startState = problem.getStartState()
    # make visited states set as graph contains cycles
    visitedStates = []

    # Initialize fringe with <start state, no actions, backward cost zero>
    fringe.push((startState, [], 0))

    while not fringe.isEmpty():
        # cheepest node or deepest node or shallowest node
        currentState, actions, currentCost = fringe.pop()

        # if state is visited then do not search
        if currentState in visitedStates:
            continue

        # add state to visited state
        visitedStates.append(currentState)

        # if state is goal state then return actions
        if problem.isGoalState(currentState):
            return actions

        # check all neighbours
        for neighbour, action, cost in problem.getSuccessors(currentState):
            if neighbour in visitedStates:
                continue
            newCost = currentCost + cost
            newAction = actions + [action]
            fringe.push((neighbour, newAction, newCost))

    #  if not path found then return empty set
    # return []
    # raise path not found for debug purpose
    raise ValueError("path not found")


#######################################################
#                 Uninformed Search                   #
#######################################################


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
    "*** YOUR CODE HERE ***"
    return searchAlgorithm(problem, util.Stack())


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return searchAlgorithm(problem, util.Queue())


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return searchAlgorithm(problem, util.PriorityQueueWithFunction(backCost))


#######################################################
#                  Informed Search                    #
#######################################################


def greedySearch(problem, heuristic=None):
    """Search the node that has the least heuristic cost."""
    "*** YOUR CODE HERE ***"
    return searchAlgorithm(problem, util.PriorityQueueWithFunction(
        lambda x: forwardCost(x, problem=problem, heuristic=heuristic)))


def aStarSearch(problem, heuristic=None):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    return searchAlgorithm(problem, util.PriorityQueueWithFunction(
        lambda x: totalCost(x, problem=problem, heuristic=heuristic)))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch


#######################################################
# below changes are made so that autograder can run   #
#######################################################


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
