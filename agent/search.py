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
import time

import chess

from .evaluation import BLACK_SCORES, PIECE_VALUES, WHITE_SCORES, evaluate

# Integer bounds used instead of _INF.  Any legal evaluation fits inside
# these limits and integer comparisons are faster than float in CPython.
_INF = 200_000
_WHITE = chess.WHITE
_PAWN = chess.PAWN
_KING = chess.KING
_CHECK_BONUS = 800
_CASTLE_BONUS = 600


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


def _get_candidate_moves(board, move_ordering):
    """
    Return the legal moves for the current position, optionally sorted by quality.

    When move_ordering is False, the unsorted generator from board.legal_moves is
    returned directly — no list allocation, no sorting cost.

    When move_ordering is True, moves are materialised into a list and sorted
    by a composite heuristic score (highest first) so that Alpha-Beta pruning
    sees the most promising moves earliest.  A beta-cutoff on the first move
    skips all remaining moves entirely, which is the best-case pruning scenario.

    Scoring terms (summed, highest wins):
      1. Promotions       (+10 000 + promotion-piece bonus)
                          Searched first — a promotion almost always changes the game.
      2. Captures         (+5 000 + victim value - attacker value / 10)
                          MVV-LVA heuristic: prefer capturing a high-value piece with
                          a low-value one (e.g. pawn takes queen scores highest).
      3. Positional gain  (piece-square table score at destination - source)
                          Rewards moves that improve the piece's position even when
                          no capture or special move is involved.
      4. Castling         (+600)
                          Castling improves king safety and connects the rooks.
      5. Checks           (+800)
                          Forcing moves restrict the opponent's replies and often
                          lead to immediate material gain or checkmate.
      6. Promotion target (+opponent table bonus for the promoted piece)
                          Extra positional credit for where the promoted piece lands.
    """
    if not move_ordering:
        return board.legal_moves

    mover_is_white = board.turn == _WHITE
    is_capture = board.is_capture
    gives_check = board.gives_check
    piece_type_at = board.piece_type_at
    score_table = WHITE_SCORES if mover_is_white else BLACK_SCORES
    opponent_table = BLACK_SCORES if mover_is_white else WHITE_SCORES

    moves = list(board.legal_moves)
    moves.sort(
        key=lambda move: (
            (10_000 + (move.promotion * 100)) if move.promotion else 0
        ) + (
            (
                5_000
                + (
                    PIECE_VALUES[_PAWN]
                    if board.is_en_passant(move)
                    else PIECE_VALUES.get(piece_type_at(move.to_square) or 0, 0)
                )
                - PIECE_VALUES.get(piece_type_at(move.from_square) or 0, 0) // 10
            )
            if is_capture(move)
            else 0
        ) + (
            score_table[move.promotion or piece_type_at(move.from_square)][move.to_square]
            - score_table[piece_type_at(move.from_square)][move.from_square]
        ) + (
            _CASTLE_BONUS
            if piece_type_at(move.from_square) == _KING and board.is_castling(move)
            else 0
        ) + (
            _CHECK_BONUS if gives_check(move) else 0
        ) + (
            opponent_table[move.promotion][move.to_square]
            if move.promotion
            else 0
        ),
        reverse=True,
    )
    return moves


def minimax(board, depth, alpha, beta, is_maximising, counter, move_ordering=False,
            _evaluate=evaluate, _get_moves=_get_candidate_moves):
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
        return _evaluate(board), None

    push = board.push
    pop = board.pop

    if is_maximising:
        best_score = -_INF
        best_move = None
        for move in _get_moves(board, move_ordering):
            push(move)
            score, _ = minimax(
                board, depth - 1, alpha, beta, False, counter, move_ordering
            )
            pop()

            if score > best_score:
                best_score = score
                best_move = move

            if best_score > alpha:
                alpha = best_score
            if beta <= alpha:
                break
        return best_score, best_move

    best_score = _INF
    best_move = None
    for move in _get_moves(board, move_ordering):
        push(move)
        score, _ = minimax(
            board, depth - 1, alpha, beta, True, counter, move_ordering
        )
        pop()

        if score < best_score:
            best_score = score
            best_move = move

        if best_score < beta:
            beta = best_score
        if beta <= alpha:
            break
    return best_score, best_move


def minimax_no_pruning(board, depth, is_maximising, counter, move_ordering=False,
                       _evaluate=evaluate, _get_moves=_get_candidate_moves):
    """
    Pure Minimax without Alpha-Beta pruning.

    Args / Returns: same as minimax() except there are no alpha/beta bounds.
    """
    counter[0] += 1

    if depth == 0 or board.is_game_over():
        return _evaluate(board), None

    push = board.push
    pop = board.pop
    best_move = None

    if is_maximising:
        best_score = -_INF
        for move in _get_moves(board, move_ordering):
            push(move)
            score, _ = minimax_no_pruning(
                board, depth - 1, False, counter, move_ordering
            )
            pop()
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    best_score = _INF
    for move in _get_moves(board, move_ordering):
        push(move)
        score, _ = minimax_no_pruning(
            board, depth - 1, True, counter, move_ordering
        )
        pop()
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
            -_INF,
            _INF,
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
