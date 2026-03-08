"""
search.py
---------
Minimax search with Alpha-Beta pruning — the core AI decision engine.

HOW THE AI CHOOSES A MOVE
--------------------------
Chess has too many possible positions to search exhaustively, so the AI
uses Minimax: it builds a game tree to a fixed depth and assumes both
players always play the best available move.

  - White is the MAXIMISING player  (wants the highest evaluation score)
  - Black is the MINIMISING player  (wants the lowest evaluation score)

At each node the AI:
  1. Tries every legal move.
  2. Recursively evaluates the resulting position.
  3. Keeps the move that gives the best outcome for the current player.

When the tree reaches the depth limit (or the game is over) it calls
evaluate() from evaluation.py to score the leaf position heuristically.

ALPHA-BETA PRUNING
------------------
Alpha-Beta is an optimisation that cuts branches that cannot affect the
final decision:
  - alpha  = best score the maximising player can already guarantee
  - beta   = best score the minimising player can already guarantee

If beta ≤ alpha at any node, the opponent would never allow play to reach
this branch, so we stop searching it (a "cutoff").  This prunes large
parts of the tree without changing the result — the same move is chosen
as pure Minimax, just faster.

In the best case Alpha-Beta reduces the branching factor from ~35 to ~6,
allowing roughly twice the search depth in the same time.
"""
import chess
import math
import time
from evaluation import evaluate


def minimax(board, depth, alpha, beta, is_maximising, counter):
    """
    Minimax search with Alpha-Beta pruning.

    Args:
        board:          chess.Board — current game state (modified in-place
                        using push/pop to avoid copying the board each call)
        depth:          int — remaining plies to search (0 = leaf node)
        alpha:          float — best score the MAX player can guarantee so far
        beta:           float — best score the MIN player can guarantee so far
        is_maximising:  bool — True when it is White's turn (MAX node)
        counter:        dict with key 'nodes' — incremented each call so
                        the caller can report how many nodes were searched

    Returns:
        (score, move) tuple where:
          score — the minimax value of this node (centipawns)
          move  — the best chess.Move found at this node (None at leaves)
    """
    # Count every node visited (used for benchmarking)
    counter['nodes'] += 1

    # Base case: depth limit reached, or the game is already over.
    # Return the static evaluation of the current position.
    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    if is_maximising:
        # WHITE's turn — try to maximise the score
        best_score = -math.inf
        best_move = None
        for move in board.legal_moves:
            board.push(move)                                      # make move
            score, _ = minimax(board, depth - 1, alpha, beta, False, counter)
            board.pop()                                           # unmake move

            if score > best_score:
                best_score = score
                best_move = move

            # Update alpha: the best White can guarantee from here
            alpha = max(alpha, best_score)

            # Beta cutoff: Black already has a better option elsewhere,
            # so White will never reach this node — stop searching.
            if beta <= alpha:
                break
        return best_score, best_move

    else:
        # BLACK's turn — try to minimise the score
        best_score = math.inf
        best_move = None
        for move in board.legal_moves:
            board.push(move)                                      # make move
            score, _ = minimax(board, depth - 1, alpha, beta, True, counter)
            board.pop()                                           # unmake move

            if score < best_score:
                best_score = score
                best_move = move

            # Update beta: the best Black can guarantee from here
            beta = min(beta, best_score)

            # Alpha cutoff: White already has a better option elsewhere,
            # so Black will never reach this node — stop searching.
            if beta <= alpha:
                break
        return best_score, best_move


def minimax_no_pruning(board, depth, is_maximising, counter):
    """
    Pure Minimax WITHOUT Alpha-Beta pruning.

    Identical logic to minimax() above but visits every node in the tree
    without any cutoffs.  Used only in benchmark.py to compare how many
    nodes Alpha-Beta saves versus exhaustive Minimax.

    Args / Returns: same as minimax() except no alpha/beta parameters.
    """
    counter['nodes'] += 1

    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None
    if is_maximising:
        best_score = -math.inf
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax_no_pruning(board, depth - 1, False, counter)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    else:
        best_score = math.inf
        for move in board.legal_moves:
            board.push(move)
            score, _ = minimax_no_pruning(board, depth - 1, True, counter)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
        return best_score, best_move


def get_best_move_no_pruning(board, depth):
    """
    Public wrapper for pure Minimax (no pruning).

    Sets up the counter and timer, then calls minimax_no_pruning().
    Used by benchmark.py for performance comparison.

    Returns:
        (move, nodes_searched, elapsed_seconds)
    """
    counter = {'nodes': 0}
    is_max = (board.turn == chess.WHITE)   # White maximises, Black minimises
    start = time.time()
    score, move = minimax_no_pruning(board, depth, is_max, counter)
    elapsed = time.time() - start
    return move, counter['nodes'], elapsed


def get_best_move(board, depth):
    """
    Public wrapper for Minimax with Alpha-Beta pruning.

    Sets up the counter, initialises alpha to -∞ and beta to +∞ (no
    information known yet), then calls minimax().  This is the function
    called by the game loop on every AI turn.

    Returns:
        (move, nodes_searched, elapsed_seconds)
    """
    counter = {'nodes': 0}
    is_max = (board.turn == chess.WHITE)   # White maximises, Black minimises
    start = time.time()
    # Start with the widest possible window: alpha = -∞, beta = +∞
    score, move = minimax(board, depth, -math.inf, math.inf, is_max, counter)
    elapsed = time.time() - start
    return move, counter['nodes'], elapsed


if __name__ == "__main__":
    # Quick smoke test: find the best first move from the starting position
    board = chess.Board()
    move, nodes, t = get_best_move(board, depth=3)
    print(f"Best move: {move}")
    print(f"Nodes searched: {nodes}")
    print(f"Time: {t:.3f}s")