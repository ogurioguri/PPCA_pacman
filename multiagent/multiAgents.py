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
        newFoodList = newFood.asList()
        foodDistances = [manhattanDistance(newPos, food) for food in newFoodList]
        if foodDistances:
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 0
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances) if ghostDistances else float('inf')
        scaredCountdown = min(newScaredTimes) if newScaredTimes else 0
        if scaredCountdown == 0:
            if minGhostDistance > 0:
                ghostPenalty = 1.0 / minGhostDistance
            else:
                ghostPenalty = float('inf')
        else:
            ghostPenalty =  -(5.0 / (minGhostDistance + 1) * (scaredCountdown-2))
        evaluation = successorGameState.getScore() + 2*(1.0 / (minFoodDistance + 1)) - ghostPenalty
        return evaluation

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
    #    if(depth == 0 or state.isLose() or state.isWin()):
     #       return self.evaluationFunction(state)
    #    if(index == 0):
      #      return self.maxValue(state,depth,index)
       # else:
        #    return self.minValue(state,depth,index)
    
   # def maxValue(self, state, depth,index):
      #  v = float('-inf')
      #  actions = state.getLegalActions(0)
     #   for action in actions:
       #     successorState = state.generateSuccessor(0,action)
      #      v = max(v, self.minimax(successorState, depth, 1))
       # return v

    #def minValue(self, state, depth, agentIndex):
        #v = float('inf')
       # nextAgentIndex = agentIndex + 1
       # if nextAgentIndex >= state.getNumAgents():
          #  nextAgentIndex = 0
           # depth -= 1
        #for action in state.getLegalActions(agentIndex):
            #successorState = state.generateSuccessor(agentIndex, action)
            #v = min(v, self.minimax(successorState, depth, nextAgentIndex))
        #return v
    def minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        
        if agentIndex == 0: 
            return self.maxValue(state, depth)
        else: 
            return self.minValue(state, depth, agentIndex)

    def maxValue(self, state, depth):
        v = float('-inf')
        bestAction = None
        actions = state.getLegalActions(0)  
        for action in actions:
            successorState = state.generateSuccessor(0, action) 
            score, _ = self.minimax(successorState, depth , 1)
            if score > v:
                v = score
                bestAction = action
        return v, bestAction
    
    def minValue(self, state, depth, agentIndex):
        v = float('inf')
        bestAction = None
        actions = state.getLegalActions(agentIndex)
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents() 
        newDepth = depth - 1 if nextAgentIndex == 0 else depth

        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successorState, newDepth, nextAgentIndex)
            if score < v:
                v = score
                bestAction = action
        return v, bestAction
    
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
        "*** YOUR CODE HERE ***"
        _, bestAction = self.minimax(gameState, self.depth, self.index)
        return bestAction
        util.raiseNotDefined()
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimax(self, state, depth, agentIndex,a,b):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        if agentIndex == 0: 
            return self.maxValue(state, depth,a,b)
        else: 
            return self.minValue(state, depth, agentIndex,a,b)

    def maxValue(self, state, depth,a,b):
        v = float('-inf')
        bestAction = None
        actions = state.getLegalActions(0)  
        for action in actions:
            successorState = state.generateSuccessor(0, action) 
            score, _ = self.minimax(successorState, depth,1,a,b)
            if score > v:
                v = score
                bestAction = action
            a = max(a,v)
            if(a >b):
                break
        return v, bestAction
    
    def minValue(self, state, depth, agentIndex,a,b):
        v = float('inf')
        bestAction = None
        actions = state.getLegalActions(agentIndex)
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents() 
        newDepth = depth - 1 if nextAgentIndex == 0 else depth

        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successorState, newDepth, nextAgentIndex,a,b)
            if score < v:
                v = score
                bestAction = action
            b = min(b,v)
            if(a>b):
                break
        return v, bestAction

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.minimax(gameState, self.depth, self.index,float('-inf'),float('inf'))
        return bestAction
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def minimax(self, state, depth, agentIndex):
        if depth == 0 or state.isWin() or state.isLose():
            return self.evaluationFunction(state), None
        
        if agentIndex == 0: 
            return self.maxValue(state, depth)
        else: 
            return self.ExpectValue(state, depth, agentIndex)

    def maxValue(self, state, depth):
        v = float('-inf')
        bestAction = None
        actions = state.getLegalActions(0)  
        for action in actions:
            successorState = state.generateSuccessor(0, action) 
            score, _ = self.minimax(successorState, depth , 1)
            if score > v:
                v = score
                bestAction = action
        return v, bestAction
    
    def ExpectValue(self, state, depth, agentIndex):
        v = float('inf')
        bestAction = None
        actions = state.getLegalActions(agentIndex)
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents() 
        newDepth = depth - 1 if nextAgentIndex == 0 else depth
        all_score = []
        for action in actions:
            successorState = state.generateSuccessor(agentIndex, action)
            score, _ = self.minimax(successorState, newDepth, nextAgentIndex)
            all_score.append(score)
        v = sum(all_score) / len(actions)
        return v, None

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.minimax(gameState, self.depth, self.index)
        return bestAction
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newFoodList = newFood.asList()
    foodDistances = [manhattanDistance(newPos, food) for food in newFoodList]
    if foodDistances:
        minFoodDistance = min(foodDistances)
    else:
        minFoodDistance = 0
    ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    minGhostDistance = min(ghostDistances) if ghostDistances else float('inf')
    scaredCountdown = min(newScaredTimes) if newScaredTimes else 0
    if scaredCountdown == 0:
        if minGhostDistance > 0:
            ghostPenalty = 1.0 / minGhostDistance
        else:
            ghostPenalty = float('inf')
    else:
        ghostPenalty =  -(10.0 / (minGhostDistance + 1) * (scaredCountdown-2))
    stopPenalty = 0
    if currentGameState.getPacmanState().getDirection() == 'Stop':
        stopPenalty = 10 
    evaluation = currentGameState.getScore() + (3.0 / (minFoodDistance + 1)) - ghostPenalty - stopPenalty
    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
