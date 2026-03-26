"""
search.py
---------
Minimax search with Alpha-Beta pruning - the core AI decision engine.

HOW THE AI CHOOSES A MOVE
-------------------------
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

If beta <= alpha at any node, the opponent would never allow play to reach
this branch, so we stop searching it (a "cutoff"). This prunes large parts
of the tree without changing the result - the same move is chosen as pure
Minimax, just faster.
"""
from dataclasses import dataclass
import math
import time

import chess

from .evaluation import evaluate


@dataclass(frozen=True)
class SearchConfig:
    """
    Tunable settings for the chess agent.

    depth:
        Search depth in plies.
    use_alpha_beta:
        When False, use plain Minimax instead.
    move_ordering:
        When True, search forcing moves first to improve Alpha-Beta pruning.
    """
    depth: int
    use_alpha_beta: bool = True
    move_ordering: bool = False


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move | None
    score: float
    nodes: int
    elapsed_seconds: float


def _move_order_score(board, move):
    """
    Score moves so tactical moves are searched before quiet ones.

    This is intentionally simple: promotions, captures, and checks are given
    priority because they tend to create better cutoffs.
    """
    score = 0
    piece_type_at = board.piece_type_at

    if move.promotion:
        score += 10_000 + (move.promotion * 100)

    if board.is_capture(move):
        captured_type = piece_type_at(move.to_square)
        attacker_type = piece_type_at(move.from_square)
        captured_value = 0 if captured_type is None else captured_type * 100
        attacker_value = 0 if attacker_type is None else attacker_type * 10
        score += 5_000 + captured_value - attacker_value

    if board.gives_check(move):
        score += 1_000

    return score


def _get_candidate_moves(board, move_ordering):
    moves = list(board.legal_moves)
    if move_ordering:
        score_move = lambda move: _move_order_score(board, move)
        moves.sort(key=score_move, reverse=True)
    return moves


def minimax(board, depth, alpha, beta, is_maximising, counter, move_ordering=False):
    """
    Minimax search with Alpha-Beta pruning.

    Args:
        board:          chess.Board - current game state (modified in-place
                        using push/pop to avoid copying the board each call)
        depth:          int - remaining plies to search (0 = leaf node)
        alpha:          float - best score the MAX player can guarantee so far
        beta:           float - best score the MIN player can guarantee so far
        is_maximising:  bool - True when it is White's turn (MAX node)
        counter:        dict with key 'nodes' - incremented each call so the
                        caller can report how many nodes were searched
        move_ordering:  bool - whether to sort candidate moves before search

    Returns:
        (score, move) tuple where:
          score - the minimax value of this node (centipawns)
          move  - the best chess.Move found at this node (None at leaves)
    """
    counter[0] += 1

    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    if is_maximising:
        best_score = -math.inf
        best_move = None
        for move in _get_candidate_moves(board, move_ordering):
            board.push(move)
            score, _ = minimax(
                board, depth - 1, alpha, beta, False, counter, move_ordering
            )
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score, best_move

    best_score = math.inf
    best_move = None
    for move in _get_candidate_moves(board, move_ordering):
        board.push(move)
        score, _ = minimax(
            board, depth - 1, alpha, beta, True, counter, move_ordering
        )
        board.pop()

        if score < best_score:
            best_score = score
            best_move = move

        beta = min(beta, best_score)
        if beta <= alpha:
            break
    return best_score, best_move


def minimax_no_pruning(board, depth, is_maximising, counter, move_ordering=False):
    """
    Pure Minimax without Alpha-Beta pruning.

    Args / Returns: same as minimax() except there are no alpha/beta bounds.
    """
    counter[0] += 1

    if depth == 0 or board.is_game_over():
        return evaluate(board), None

    best_move = None
    if is_maximising:
        best_score = -math.inf
        for move in _get_candidate_moves(board, move_ordering):
            board.push(move)
            score, _ = minimax_no_pruning(
                board, depth - 1, False, counter, move_ordering
            )
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    best_score = math.inf
    for move in _get_candidate_moves(board, move_ordering):
        board.push(move)
        score, _ = minimax_no_pruning(
            board, depth - 1, True, counter, move_ordering
        )
        board.pop()
        if score < best_score:
            best_score = score
            best_move = move
    return best_score, best_move


def search_position(board, config):
    """
    Search a position using the supplied SearchConfig.

    Returns:
        SearchResult containing the chosen move plus timing/statistics.
    """
    counter = [0]
    is_max = board.turn == chess.WHITE
    start = time.perf_counter()

    if config.use_alpha_beta:
        score, move = minimax(
            board,
            config.depth,
            -math.inf,
            math.inf,
            is_max,
            counter,
            config.move_ordering,
        )
    else:
        score, move = minimax_no_pruning(
            board,
            config.depth,
            is_max,
            counter,
            config.move_ordering,
        )

    elapsed = time.perf_counter() - start
    return SearchResult(move, score, counter[0], elapsed)


def get_best_move_no_pruning(board, depth, move_ordering=False):
    """
    Public wrapper for pure Minimax (no pruning).

    Returns:
        (move, nodes_searched, elapsed_seconds)
    """
    result = search_position(
        board,
        SearchConfig(depth=depth, use_alpha_beta=False, move_ordering=move_ordering),
    )
    return result.move, result.nodes, result.elapsed_seconds


def get_best_move(board, depth, move_ordering=False):
    """
    Public wrapper for Minimax with Alpha-Beta pruning.

    Returns:
        (move, nodes_searched, elapsed_seconds)
    """
    result = search_position(
        board,
        SearchConfig(depth=depth, use_alpha_beta=True, move_ordering=move_ordering),
    )
    return result.move, result.nodes, result.elapsed_seconds


if __name__ == "__main__":
    board = chess.Board()
    result = search_position(board, SearchConfig(depth=3, move_ordering=True))
    print(f"Best move: {result.move}")
    print(f"Score: {result.score}")
    print(f"Nodes searched: {result.nodes}")
    print(f"Time: {result.elapsed_seconds:.3f}s")
