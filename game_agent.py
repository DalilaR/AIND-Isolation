"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

"""Check the length of the intersection between legal moves for player and opponent.
Remove the result from the player number of legal moves"""
def heuristic4(game,player, opponent):
    opp_moves_set = set(game.get_legal_moves(opponent))
    my_moves_set = set(game.get_legal_moves(player))
    l_inter = len(opp_moves_set.intersection(my_moves_set))
    my_len = len(my_moves_set)
    score = (my_len - l_inter)
    return score
   
"""Heuristic 2 borrow from Statistics,  more precisely, the F Statistics.  The F statistics measure differences between "clusters" or groups.  It is the ratio of the mean variance between centroids ( means of each group) and the mean of the variances within groups"""
def heuristic2(game,player,opponent):
    my_moves = game.get_legal_moves(player)
    my_len = len(my_moves)
    opp_moves = game.get_legal_moves(opponent)
    opp_len = len(opp_moves)
    my_location = game.get_player_location(player)
    opponent_location = game.get_player_location(opponent)
    mean_location = [(my_location[0]+opponent_location[0])/2, (my_location[1]+opponent_location[1])/2]
    sb = (my_location[0] - mean_location[0])*(my_location[0] - mean_location[0])+(my_location[1] - mean_location[1])*(my_location[1] - mean_location[1])+(opponent_location[0] - mean_location[0])*(opponent_location[0] - mean_location[0])+(opponent_location[1] - mean_location[1])*(opponent_location[1] - mean_location[1])
    if (my_len+opp_len-2) > 0:
        sw = 5 * (my_len+opp_len)/(my_len+opp_len-2) 
        score = sb/sw
    else:
        score = math.sqrt((my_location[0]-opponent_location[0])**2+(my_location[1]-opponent_location[1])**2)
    return score
"""Check how many legal moves player and opponent shares.  As the maximum of number of shared legal moves can be at most 2, I
decided to set score to 12 if the number of shared legal moves is 2, 10 if shared legal moves is 10 and the number of legal moves for player for all other instances"""
def heuristic3(game,player, opponent):
    my_moves_set = set(game.get_legal_moves(player))
    opp_moves_set = set(game.get_legal_moves(opponent))
    l_inter = len(opp_moves_set.intersection(my_moves_set))
    my_len = len(my_moves_set)
    opp_len = len(opp_moves_set)
    if l_inter == 2:
        score = 12
    elif l_inter == 1:
        score = 10
    else:
        score = my_len
    return score
def heuristic1(game,player, opponent):
    weighted_positions = { (0,0): 2, (0,1):3, (0,2): 4,(0,3):4,(0,4): 4,(0,5):3,(0,6):2,
                           (1,0): 3, (1,1):4, (1,2): 6,(1,3):6,(1,4): 6,(1,5):4,(1,6):3,
                           (2,0): 4, (2,1):6, (2,2): 8,(2,3):8,(2,4): 8,(2,5):6,(2,6):4,
                           (3,0): 4, (3,1):6, (3,2): 8,(3,3):8,(3,4): 8,(3,5):6,(3,6):4,
                           (4,0): 4, (4,1):6, (4,2): 8,(4,3):8,(4,4): 8,(4,5):6,(4,6):4,
                           (5,0): 3, (5,1):4, (5,2): 6,(5,3):6,(5,4): 6,(5,5):4,(5,6):3,
                           (6,0): 2, (6,1):3, (6,2): 4,(6,3):4,(6,4): 4,(6,5):3,(6,6):2,
                            }
    weighted_positions2 = {(0,0): 4, (0,1):8, (0,2): 16, (0,3):16, (0,4):16,(0,5):8,(0,6):4,
                           (1,0): 8, (1,1):16, (1,2): 12,(1,3):12,(1,4): 12,(1,5):16,(1,6):8,
                           (2,0): 16, (2,1):12, (2,2): 9,(2,3):9,(2,4): 9,(2,5):12,(2,6):16,
                           (3,0): 16, (3,1):12, (3,2): 9,(3,3):9,(3,4): 9,(3,5):12,(3,6):16,
                           (4,0): 16, (4,1):12, (4,2): 9,(4,3):9,(4,4): 9,(4,5):12,(4,6):16,
                           (5,0): 8, (5,1):16, (5,2): 12,(5,3):12,(5,4): 12,(5,5):16,(5,6):8,
                           (6,0): 4, (6,1):8, (6,2): 16,(6,3):16,(6,4): 16,(6,5):8,(6,6):4,
                            }
    
    my_location = game.get_player_location(player)
    opponent_location = game.get_player_location(opponent)
    #score =  20 - weighted_positions[my_location]
    my_moves = game.get_legal_moves(player)
    my_len = len(my_moves)
    if my_len > 0:
        score = (10-weighted_positions[my_location])/ my_len
    else: 
        score = (10-weighted_positions[my_location])
    return score
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    """ For the score, I know that at most a player has 8 potential legal moves at any them; therefore,
    I decided to use get_legal_moves to find all possible moves, and then count them.
    A location is a strong as the number of possible exits it has, which is similar to the isolation problem worked on in the lecture.  
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    if len(game.get_legal_moves(player)) == 0:
        return float("-inf")
    opponent = game.get_opponent(player)
    score = heuristic1(game,player, opponent)
    
    return float(score )

    raise NotImplementedError
    

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=8, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        check_legal_move = game.get_legal_moves()
        number_legal_moves = len(check_legal_move)
        #Stating that a game is a losing game is equivalent to stating that the player has no legal 
        # moves.  Still, I check for both conditions
        if game.is_loser(self):
            return (-1,-1)
        if number_legal_moves == 0:
            return (-1, -1)
        score = float("-inf")
        position = (-1, -1)
        depth = 0
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            """As we need to loop until we find a winning position or we get to timeout.  
            I have decided to create an infinity loop when iterative is true.
            As minimax and alphabeta return best position and score, given a starting position, 
            it doesn't make sense to go through the legal moves within the iterative loop.  
            Furthermore, as we are interested only on winning position or timeout, 
            it makes sense to return position when score is equal to "inf", which 
            means a winning position.
            If time runs out, a position has to be returned.
            """
            while self.iterative == True:
                depth += 1
                if self.method == "minimax":
                    score, position = self.minimax(game, depth,True) 
                elif self.method == "alphabeta":
                    score, position = self.alphabeta(game, depth)
                if score == float("inf"):
                    return position 
            if not self.iterative:
                depth = self.search_depth
                if self.method == "minimax":
                    score, position = self.minimax(game, depth,True) 
                elif self.method == "alphabeta":
                    score, position = self.alphabeta(game, depth)
            pass 
        except Timeout:
            return position
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return position
        raise NotImplementedError
    
                   
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

    def minimax(self, game, depth ,maximizing_player=True ):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        check_legal_move = game.get_legal_moves()
        if game.is_loser(self):
                bestValue = float('-inf')
                current_location = (-1,-1)
                return bestValue,current_location
        if game.is_winner(self):
                bestValue = float('inf')
                current_location = game.get_player_location(self)
                return  bestValue,current_location
        '''if len (check_legal_move) == 0:
                bestValue = custom_score(game,self)
                current_location = game.get_player_location(self)
                return bestValue,current_location'''
        if depth == 0:
                bestValue = self.score(game,self)
                return bestValue,(-1,-1)
        if maximizing_player:
            bestValue = float('-inf')
            current_location = game.get_player_location(self)
            for v in check_legal_move:
                v_game = game.forecast_move(v)
                value, location = self.minimax(v_game,depth-1,False)
                if value > bestValue :
                    bestValue = value
                    current_location = v
        else:
            bestValue = float('inf')
            current_location = game.get_player_location(self)
            for v in check_legal_move:  
                v_game = game.forecast_move(v)
                value, location = self.minimax(v_game,depth-1,True)
                if value < bestValue :
                    bestValue = value
                    current_location = v
        return bestValue, current_location  
        
        
   
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        
        #print("current location: ", game.get_player_location(self))
        check_legal_move = game.get_legal_moves()
        if game.is_loser(self):
                bestValue = float('-inf')
                current_location = (-1,-1)
                return bestValue,current_location
        if game.is_winner(self):
                bestValue = float('inf')
                current_location = game.get_player_location(self)
                return  bestValue,current_location
        # This is not needed, as if there are no legal moves it means game.is_loser 
        '''if len (check_legal_move) == 0:
                bestValue = custom_score(game,self)
                current_location = game.get_player_location(self)
                #print("LEAF",game.get_player_location(self),self.score(game,self))
                return bestValue,current_location'''
        if depth == 0:
                #print("REACHED DEPTH ",game.get_player_location(self))
                bestValue = self.score(game,self)
                #print("Value ", bestValue)
                return bestValue,(-1,-1)
        if maximizing_player:
            #print("Max")
            bestValue = float('-inf')
            current_location = game.get_player_location(self)
            for v in check_legal_move:
                v_game = game.forecast_move(v)
                value, _ = self.alphabeta(v_game,depth-1,alpha, beta,False)
                #print("Value at max ", value, "location ",current_location," V value ",v)
                if value > bestValue :
                    bestValue = value
                    current_location = v
                #alpha = max(alpha, bestValue)
                if beta <= bestValue:
                       return bestValue, current_location
                alpha = max(alpha, bestValue)
            #print("Best: ",bestValue," Current_location: ",current_location)
            return bestValue, current_location
        else:
            #print("Min")
            bestValue = float('inf')
            current_location = game.get_player_location(self)
            for v in check_legal_move:  
                v_game = game.forecast_move(v)
                value, _ = self.alphabeta(v_game,depth-1,alpha, beta, True)
                if value < bestValue :
                    bestValue = value
                    current_location = v   
                #print("Best: ",bestValue," Current_location: ",current_location)
                if bestValue <= alpha:
                      return bestValue, current_location  
                beta = min(beta,bestValue)
            return bestValue, current_location
        #raise NotImplementedError
        