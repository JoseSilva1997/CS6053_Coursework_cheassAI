"""
game.py
-------
Single-game logic for the agent-vs-engine benchmark.

Responsibilities:
  - Build the list of opening positions to test (build_opening_suite).
  - Determine the game outcome from the agent's perspective (win/draw/loss/truncated).
  - Play one complete game between the agent and a UCI engine (play_game).

The game loop alternates between calling the agent's search (search_position)
and the engine's play() method until the game ends or the ply limit is reached.
"""
import chess
import chess.engine

from agent.search import search_position
from config.openings import OPENING_POSITIONS
from .models import AgentPreset, GameRecord


def build_opening_suite(count):
    """
    Return a list of (name, FEN) pairs to use as starting positions.

    Always includes the standard starting position first, then `count`
    curated openings from config/openings.py.

    If `count` exceeds the number of curated openings, the list wraps
    around from the start again.
    """
    openings = [("Starting Position", chess.STARTING_FEN)]
    curated_count = max(0, count)
    num_curated_openings = len(OPENING_POSITIONS)

    for i in range(curated_count):
        openings.append(OPENING_POSITIONS[i % num_curated_openings])

    return openings


def _is_game_over(board):
    """
    Game-over check that includes draw conditions without generating all legal moves.

    The built-in claim_draw path calls can_claim_threefold_repetition() which
    generates *every* legal move, pushes it, checks for repetition, and pops.
    That hidden move-generation pass runs every ply and dominates benchmark time.

    Instead we check:
      - is_game_over()     : checkmate / stalemate / fivefold / seventy-five-move / insufficient
      - is_repetition(3)   : current position already appeared 3 times (O(move_stack))
      - halfmove_clock≥100 : fifty-move rule (current position, not "any next move")
    """
    return board.is_game_over() or board.is_repetition(3) or board.halfmove_clock >= 100


def _outcome_from_agent_perspective(board, agent_color, truncated):
    """
    Translate the board outcome into a (label, score) pair from the agent's point of view.

    Returns:
        ("truncated", 0.5) if the ply limit was hit.
        ("draw",      0.5) if the game ended in a draw.
        ("win",       1.0) if the agent won.
        ("loss",      0.0) if the agent lost.
    """
    if truncated:
        return "truncated", 0.5

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw", 0.5
    if outcome.winner == agent_color:
        return "win", 1.0
    return "loss", 0.0


def play_game(engine, preset: AgentPreset, opening_name, opening_fen,
              agent_color, engine_limit, max_plies) -> GameRecord:
    """
    Play one game between the agent and a UCI engine from a given opening.

    The agent uses preset.config (depth, alpha-beta, move ordering).
    The engine uses the pre-configured engine_limit (time per move).
    The game is truncated after max_plies half-moves to avoid infinite draws.

    Args:
        engine:       chess.engine.SimpleEngine — the opponent engine.
        preset:       AgentPreset — which search configuration the agent uses.
        opening_name: str — human-readable label for the starting position.
        opening_fen:  str — FEN string for the starting position.
        agent_color:  chess.WHITE or chess.BLACK.
        engine_limit: chess.engine.Limit — time budget for engine moves.
        max_plies:    int — truncation limit.

    Returns:
        GameRecord with outcome statistics for this game.
    """
    board = chess.Board(opening_fen)
    agent_move_times = []
    agent_nodes = []
    engine_move_times = []
    game_over = False
    truncated = False

    while not game_over and board.ply() < max_plies:
        if board.turn == agent_color:
            result = search_position(board, preset.config)
            if result.move is None:
                break
            board.push(result.move)
            agent_move_times.append(result.elapsed_seconds)
            agent_nodes.append(result.nodes)
        else:
            # Send only the current FEN to the engine so it doesn't have to
            # replay the full move history (which grows every ply).
            fen_board = chess.Board(board.fen())
            engine_result = engine.play(fen_board, engine_limit)
            if engine_result.move is None:
                break
            board.push(engine_result.move)
            if engine_result.info and "time" in engine_result.info:
                engine_move_times.append(engine_result.info["time"])

        game_over = _is_game_over(board)

    if not game_over and board.ply() >= max_plies:
        truncated = True

    outcome, score = _outcome_from_agent_perspective(board, agent_color, truncated)
    agent_avg_time = sum(agent_move_times) / len(agent_move_times) if agent_move_times else 0.0
    agent_avg_nodes = sum(agent_nodes) / len(agent_nodes) if agent_nodes else 0.0
    engine_avg_time = sum(engine_move_times) / len(engine_move_times) if engine_move_times else 0.0

    return GameRecord(
        preset_name=preset.name,
        opening_name=opening_name,
        agent_color="white" if agent_color == chess.WHITE else "black",
        outcome=outcome,
        score=score,
        plies=board.ply(),
        agent_moves=len(agent_move_times),
        agent_avg_time_s=agent_avg_time,
        agent_avg_nodes=agent_avg_nodes,
        engine_avg_time_s=engine_avg_time,
    )
