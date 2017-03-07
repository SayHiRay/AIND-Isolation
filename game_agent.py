"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score_3(game, player):
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
    # print(len(game.get_legal_moves(player=player)))
    # print(len(game.get_legal_moves(player=(game.get_opponent(player)))))
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    r1, c1 = game.get_player_location(player=player)
    r2, c2 = game.get_player_location(player=game.get_opponent(player))

    if (game.move_count) < game.width * game.height // 3:
        return float(1 * len(game.get_legal_moves(player=player)) - 1 * len(
            game.get_legal_moves(player=(game.get_opponent(player)))))

    else:
        return float(1 * len(game.get_legal_moves(player=player)) - 2 * len(
            game.get_legal_moves(player=(game.get_opponent(player)))))


def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]

    if game.get_player_location(player=player):
        r, c = game.get_player_location(player=player)
        player_move_num = len([(r + dr, c + dc) for dr, dc in directions if 0 <= r+dr < game.height and 0 <= c+dc < game.width])
    else:
        player_move_num = 1

    if game.get_player_location(player=game.get_opponent(player)):
        r, c = game.get_player_location(player=game.get_opponent(player))
        oppo_move_num = len([(r + dr, c + dc) for dr, dc in directions if 0 <= r+dr < game.height and 0 <= c+dc < game.width])
    else:
        oppo_move_num = 1

    return float(1 * len(game.get_legal_moves(player=player))/player_move_num - 2 * len(game.get_legal_moves(player=(game.get_opponent(player))))/oppo_move_num)


def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    return float(1 * len(game.get_legal_moves(player=player)) - 1 * len(game.get_legal_moves(player=(game.get_opponent(player)))))


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

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=15.):
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

        # constructing a book of opening moves
        opening_book = {}
        opening_book[''] = []
        if game.height <= 3:
            opening_book[''].append(0)
        else:
            opening_book[''].append(0)
        if game.width <= 3:
            opening_book[''].append(0)
        else:
            opening_book[''].append(0)

        # search opening_book for opening moves
        if len(legal_moves) == 0:  # returning immediately if there are no legal moves
            return -1, -1
        elif game.move_count == 0:  # first player, first move
            if '' in opening_book:
                return tuple(opening_book[''])

        best_move = legal_moves[random.randrange(0, len(legal_moves))]  # initialize best_move as one of the legal moves

        try:
            if self.iterative:
                # print("ID")
                # print("move count:", game.move_count)
                search_depth = 1  # initialize the depth for iterative deepening
                max_depth = len(game.get_blank_spaces())
                while search_depth <= max_depth:
                    # print("depth: ", search_depth)
                    best_score, best_move = self.minimax(game, search_depth) if self.method == "minimax" \
                        else self.alphabeta(game, search_depth)
                    search_depth += 1  # deepen the search depth
            else:
                # print("NOID")
                # print("move count:", game.move_count)
                # print("depth: ", self.search_depth)
                best_score, best_move = self.minimax(game, self.search_depth) if self.method == "minimax"\
                    else self.alphabeta(game, self.search_depth)
        except:
            return best_move
        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
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

        def minimax_max_value(game, remaining_ply_num, maximizing_player):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if (not game.get_legal_moves(player=game.active_player)) or remaining_ply_num == 0:
                return self.score(game, game.active_player) if maximizing_player \
                    else self.score(game, game.inactive_player)
            v = float('-inf')

            for move in game.get_legal_moves(player=game.active_player):
                v = max(v, minimax_min_value(game.forecast_move(move), remaining_ply_num - 1, maximizing_player))
            return v

        def minimax_min_value(game, remaining_ply_num, maximizing_player):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if (not game.get_legal_moves(player=game.active_player)) or remaining_ply_num == 0:
                return self.score(game, game.inactive_player) if maximizing_player \
                    else self.score(game, game.active_player)
            v = float('inf')

            for move in game.get_legal_moves(player=game.active_player):
                v = min(v, minimax_max_value(game.forecast_move(move), remaining_ply_num - 1, maximizing_player))
            return v

        if maximizing_player:
            best_score = float('-inf')
            best_move = None
            for move in game.get_legal_moves(player=game.active_player):
                v = minimax_min_value(game.forecast_move(move), depth - 1, maximizing_player)
                if v > best_score:
                    best_score = v
                    best_move = move
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for move in game.get_legal_moves(player=game.active_player):
                v = minimax_max_value(game.forecast_move(move), depth - 1, maximizing_player)
                if v < best_score:
                    best_score = v
                    best_move = move
            return best_score, best_move

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

        def alphabeta_max_value(game, alpha, beta, remaining_ply_num, maximizing_player):
            # print("alphabeta_max_value  remaining_ply_num: ", remaining_ply_num)
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if (not game.get_legal_moves(player=game.active_player)) or remaining_ply_num == 0:
                return self.score(game, game.active_player) if maximizing_player \
                    else self.score(game, game.inactive_player)
            v = float('-inf')

            for move in game.get_legal_moves(player=game.active_player):
                v = max(v, alphabeta_min_value(game.forecast_move(move),
                                                    alpha, beta, remaining_ply_num - 1, maximizing_player))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def alphabeta_min_value(game, alpha, beta, remaining_ply_num, maximizing_player):
            # print("alphabeta_min_value  remaining_ply_num: ", remaining_ply_num)
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            if (not game.get_legal_moves(player=game.active_player)) or remaining_ply_num == 0:
                return self.score(game, game.inactive_player) if maximizing_player \
                    else self.score(game, game.active_player)
            v = float('inf')

            for move in game.get_legal_moves(player=game.active_player):
                v = min(v, alphabeta_max_value(game.forecast_move(move),
                                                    alpha, beta, remaining_ply_num - 1, maximizing_player))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        if maximizing_player:
            best_score = float('-inf')
            best_move = None
            for move in game.get_legal_moves(player=game.active_player):
                v = alphabeta_min_value(game.forecast_move(move), best_score, beta, depth - 1, maximizing_player)
                if v > best_score:
                    best_score = v
                    best_move = move
            return best_score, best_move
        else:
            best_score = float('inf')
            best_move = None
            for move in game.get_legal_moves(player=game.active_player):
                v = alphabeta_max_value(game.forecast_move(move), alpha, best_score, depth - 1, maximizing_player)
                if v < best_score:
                    best_score = v
                    best_move = move
            return best_score, best_move
