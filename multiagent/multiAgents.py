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
from copy import deepcopy


################################################
#            Evaluation Functions              #
################################################


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


################################################
#                   Agents                     #
################################################


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    """
    Pacman Agent States:
    Safe: Any ghost are not near and it is also staring state
    Fear: One or more ghost are near pacman
    """
    SAFESTATE = 1
    FEARSTATE = 2
    FEARDISTANCE = 2


    def closestFoodPosition(self, gameState):
      """
      Find closest food position according to current game state
      """
      currentPosition = gameState.getPacmanPosition()
      mfd = float('inf') # min food distance
      mfp = None  # min food position
      for foodPos in gameState.getFood().asList():
        distance = manhattanDistance(currentPosition, foodPos)
        if distance < mfd:
          mfd = distance
          mfp = foodPos
      return mfp

    def findPathToClosestFood(self, gameState):
      """
      Find path to closest food
      """
      # Find closest food
      foodPos = self.closestFoodPosition(gameState)
      # Make queue as fringe for BFS
      fringe = util.Queue()
      # Make current state as start state for problem
      startState = gameState
      # VisiteStates list for graph search
      visitedStates = []
      # Initialize fringe with statr state and empty actions as queue
      fringe.push((startState, util.Queue()))

      while not fringe.isEmpty():
        currentState, actions = fringe.pop()

        # If pacmanPosition of current state is visited then do not search
        if currentState.getPacmanPosition() in visitedStates:
          continue

        # Add state to visited search
        visitedStates.append(currentState.getPacmanPosition())

        # If state if goal state then return actions
        if currentState.getPacmanPosition() == foodPos:
          return actions

        # TODO: We can avoid calculation of getting foodPos by checking if there
        # is food at there perticular postion, below code should work but somehow
        # it isn't working
        # if currentState.getPacmanPosition() in currentState.getFood().asList():
          # return actions

        # check for all neighbours
        legalMoves = currentState.getLegalActions()
        # legalMoves.remove(Directions.STOP)
        for move in legalMoves:
          newState = currentState.generatePacmanSuccessor(move)
          newActions = deepcopy(actions)
          newActions.push(move)
          fringe.push((newState, newActions))

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Define state of pacman as safe initially
        self.pacmanState = self.SAFESTATE

        # First check ghost positions by proximate distance 2
        for ghostPos in gameState.getGhostPositions():
          distance = manhattanDistance(gameState.getPacmanPosition(), ghostPos)
          # If found then make ghost state to fear
          if distance <= self.FEARDISTANCE:
            self.pacmanState = self.FEARSTATE
            break

        # If ghost is safe then just follow path to closest food
        if self.pacmanState == self.SAFESTATE:
          # If path is already calculated then do not calculate agin just use it :)
          if hasattr(self, 'actionQueue') and not self.actionQueue.isEmpty():
            return self.actionQueue.pop()
          self.actionQueue = self.findPathToClosestFood(gameState)
          return self.actionQueue.pop()

        if self.pacmanState == self.FEARSTATE:
          # Forget past first
          if hasattr(self, 'actionQueue'):
            del self.actionQueue

          # Find out best possible way
          scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
          bestScore = max(scores)
          bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
          chosenIndex = random.choice(bestIndices) # Pick randomly among the best
          "Add more of your code here if you want to"
          return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

        "*** YOUR CODE HERE ***"
        # Calculate nearest food dot distance
        foodPositions = newFood.asList()
        mfd = float('inf') # minimum food distance
        for food in foodPositions:
          distance = manhattanDistance(newPos, food)
          mfd = min(distance, mfd)
        if mfd == float('inf'):
          mfd = 0

        # Check distance from ghost and how many ghosts are far less than 1 position
        dg = 0 # number of ghost that endangers
        for ghost in successorGameState.getGhostPositions():
          distance = manhattanDistance(newPos, ghost)
          if distance <= self.FEARDISTANCE:
            dg += 1

        # Lesser the min food distance better the state
        # Lesser the number of ghost that endangers better the state
        return successorGameState.getScore() - mfd - 10 * dg

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
        # Find number of agents first
        self.noOfAgents = gameState.getNumAgents()

        # Find effective depth
        # Because we have to expand all agents to given depth times
        effectiveDepth = self.noOfAgents * self.depth

        # Call value function and get direction from second element of return value
        _, direction = self.value(gameState, effectiveDepth, self.index)

        # Return obtain direction as best move
        return direction

    def value(self, gameState, depth, agentIndex):
        """
        Find value of node
        Params:
        @gameState: gameState
        @depth: which depth level are we
        @agentIndex: agentIndex
        @action: last taken action, if None then it's first step

        @ret:
        Return tuple (best value, for best value required direction)
        """
        # If we reached last step return  evaluation function value
        if depth == 0:
            # Return value is followed as (best value, direction for best value) tuple
            return (self.evaluationFunction(gameState), None)

        # Decide type ot agent, MIN or MAX
        if agentIndex == 0:
            agentType = 'MAX'
        else:
            agentType = 'MIN'

        # Call functions accordingly
        if agentType == 'MAX':
            return self.maxValue(gameState, depth, agentIndex)

        if agentType == 'MIN':
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth, agentIndex):
        # Find out available legal actions for agent
        legalActions = gameState.getLegalActions(agentIndex)

        # If legal action is empty then it is terminal state
        # So, call value function as depth zero and agentIndex doesn't matter
        if not legalActions:
            return self.value(gameState, 0, agentIndex)

        # Prepate nextAgentIndex
        nextAgentIndex = (agentIndex + 1) % self.noOfAgents

        # Get scores of successor for particular agent for particular state
        scores = [self.value(gameState.generateSuccessor(agentIndex, action), depth-1, nextAgentIndex)[0]
                    for action in legalActions]

        # Find max score from available scores
        maxScore = max(scores)

        # Find indices for best score and return random any of them
        bestIndices = [index for index in range(len(scores)) if scores[index] == maxScore]
        chosenIndex = random.choice(bestIndices)
        return (maxScore, legalActions[chosenIndex])

    def minValue(self, gameState, depth, agentIndex):
        # Find out available legal actions for agent
        legalActions = gameState.getLegalActions(agentIndex)

        # If legal action is empty then it is terminal state
        # So, call value function as depth zero and agentIndex doesn't matter
        if not legalActions:
            return self.value(gameState, 0, agentIndex)

        # Prepate nextAgentIndex
        nextAgentIndex = (agentIndex + 1) % self.noOfAgents

        # Get scores of successor for particular agent for particular state
        scores = [self.value(gameState.generateSuccessor(agentIndex, action), depth-1, nextAgentIndex)[0]
                    for action in legalActions]

        # Find min score from available scores
        minScore = min(scores)

        # Find indices for best score and return random any of them
        bestIndices = [index for index in range(len(scores)) if scores[index] == minScore]
        chosenIndex = random.choice(bestIndices)
        return (minScore, legalActions[chosenIndex])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Find number of agents first
        self.noOfAgents = gameState.getNumAgents()

        # Find effective depth
        # Because we have to expand all agents to given depth times
        effectiveDepth = self.noOfAgents * self.depth

        # Call value function and get direction from second element of return value
        # Intialize alpha with minus infinity and beta with plus infinity
        _, direction = self.value(gameState, effectiveDepth, self.index, float('-inf'), float('inf'))

        # Return obtain direction as best move
        return direction

    def value(self, gameState, depth, agentIndex, alpha, beta):
        # If we reached last step return  evaluation function value
        if depth == 0:
            # Return value is followed as (best value, direction for best value) tuple
            return (self.evaluationFunction(gameState), None)

        # Decide type ot agent, MIN or MAX
        if agentIndex == 0:
            agentType = 'MAX'
        else:
            agentType = 'MIN'

        # Call functions accordingly
        if agentType == 'MAX':
            return self.maxValue(gameState, depth, agentIndex, alpha, beta)

        if agentType == 'MIN':
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, gameState, depth, agentIndex, alpha, beta):
        # Find out available legal actions for agent
        legalActions = gameState.getLegalActions(agentIndex)

        # If legal action is empty then it is terminal state
        # So, call value function as depth zero and agentIndex doesn't matter
        if not legalActions:
            return self.value(gameState, 0, agentIndex, alpha, beta)

        # Prepate nextAgentIndex
        nextAgentIndex = (agentIndex + 1) % self.noOfAgents

        # Get scores of successor for particular agent for particular state
        maxValue = float('-inf')
        retAction = None
        for action in legalActions:
            value = self.value(gameState.generateSuccessor(agentIndex, action),
                                depth-1, nextAgentIndex, alpha, beta)[0]

            if value >= maxValue:
                maxValue = value
                retAction = action

            if value > beta:
                return value, retAction
            alpha = max(alpha, value)

        return maxValue, retAction

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        # Find out available legal actions for agent
        legalActions = gameState.getLegalActions(agentIndex)

        # If legal action is empty then it is terminal state
        # So, call value function as depth zero and agentIndex doesn't matter
        if not legalActions:
            return self.value(gameState, 0, agentIndex, alpha, beta)

        # Prepate nextAgentIndex
        nextAgentIndex = (agentIndex + 1) % self.noOfAgents

        # Find indices for best score and return random any of them
        minValue = float('inf')
        retAction = None
        for action in legalActions:
            value = self.value(gameState.generateSuccessor(agentIndex, action),
                                depth-1, nextAgentIndex, alpha, beta)[0]

            if value <= minValue:
                minValue = value
                retAction = action

            if value < alpha:
                return value, retAction
            beta = min(beta, value)

        return minValue, retAction

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
