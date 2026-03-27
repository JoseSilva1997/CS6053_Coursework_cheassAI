"""
vs_stockfish.py
---------------
Play this agent against a UCI engine (e.g. Stockfish) and compare multiple
search configurations across a set of opening positions.

Default parameters are read from config/settings.py — edit that file to
change depths, Elo, number of openings, etc. without using CLI flags.

Usage (simplest — uses settings.py defaults):
    python benchmarks/vs_stockfish.py

Usage (override engine path):
    python benchmarks/vs_stockfish.py --engine-path "C:\\path\\to\\stockfish.exe"

Usage (custom agent configs via CLI):
    python benchmarks/vs_stockfish.py ^
        --config ab_d3,3,true,false ^
        --config ab_d3_ordered,3,true,true
"""
import sys
from pathlib import Path

# Allow imports from the repo root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from dataclasses import dataclass
import math
from multiprocessing import Pool
import os
import time

import chess
import chess.engine

from config.openings import OPENING_POSITIONS
from config.settings import (
    AGENT_CONFIGS,
    ENGINE_ELO,
    ENGINE_TIME,
    MAX_WORKERS,
    NUM_OPENINGS,
    MAX_PLIES,
    VS_ENGINE_CSV,
    VS_ENGINE_PLOT,
)
from agent.search import SearchConfig, search_position


DEFAULT_ENGINE_PATTERNS = [
    "stockfish/stockfish*.exe",
    "stockfish/*.exe",
    "engines/stockfish*.exe",
    "engines/*.exe",
    "stockfish*.exe",
]


@dataclass(frozen=True)
class AgentPreset:
    name: str
    config: SearchConfig


@dataclass
class GameRecord:
    preset_name: str
    opening_name: str
    agent_color: str
    outcome: str
    score: float
    plies: int
    agent_moves: int
    agent_avg_time_s: float
    agent_avg_nodes: float
    engine_avg_time_s: float


def parse_bool(value):
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_config_spec(spec):
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Config must be 'name,depth,use_alpha_beta,move_ordering'"
        )

    name, depth, use_alpha_beta, move_ordering = parts
    return AgentPreset(
        name=name,
        config=SearchConfig(
            depth=int(depth),
            use_alpha_beta=parse_bool(use_alpha_beta),
            move_ordering=parse_bool(move_ordering),
        ),
    )


def build_default_presets():
    return [
        AgentPreset(name, SearchConfig(depth=depth, use_alpha_beta=ab, move_ordering=mo))
        for name, depth, ab, mo in AGENT_CONFIGS
    ]


def build_opening_suite(count):
    openings = [("Starting Position", chess.STARTING_FEN)]
    openings.extend(OPENING_POSITIONS[:count])
    return openings


def outcome_from_agent_perspective(board, agent_color, truncated):
    if truncated:
        return "truncated", 0.5

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw", 0.5
    if outcome.winner == agent_color:
        return "win", 1.0
    return "loss", 0.0


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


def play_game(engine, preset, opening_name, opening_fen, agent_color, engine_limit, max_plies):
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

    outcome, score = outcome_from_agent_perspective(board, agent_color, truncated)
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


def summarise_records(records, preset_name):
    subset = [record for record in records if record.preset_name == preset_name]
    games = len(subset)
    wins = sum(1 for record in subset if record.outcome == "win")
    draws = sum(1 for record in subset if record.outcome in {"draw", "truncated"})
    losses = sum(1 for record in subset if record.outcome == "loss")
    avg_score = sum(record.score for record in subset) / games if games else 0.0
    avg_plies = sum(record.plies for record in subset) / games if games else 0.0
    avg_time = (
        sum(record.agent_avg_time_s for record in subset) / games if games else 0.0
    )
    avg_nodes = (
        sum(record.agent_avg_nodes for record in subset) / games if games else 0.0
    )

    return {
        "preset": preset_name,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_pct": avg_score * 100.0,
        "avg_plies": avg_plies,
        "agent_avg_time_s": avg_time,
        "agent_avg_nodes": avg_nodes,
    }


def estimate_elo(opponent_elo, score_pct):
    """
    Estimate player Elo from score percentage against a known-Elo opponent.

    Derived from the Elo expected-score formula:
        E = 1 / (1 + 10^((opponent - player) / 400))

    Solving for player_elo:
        player_elo = opponent_elo - 400 * log10(1/score_pct - 1)

    Returns None if score_pct is 0% or 100% (formula is undefined at extremes).
    A note is printed instead recommending more games.
    """
    if score_pct <= 0.0 or score_pct >= 1.0:
        return None
    return round(opponent_elo - 400 * math.log10(1.0 / score_pct - 1))


def collect_summaries(records, presets):
    return [summarise_records(records, preset.name) for preset in presets]


def print_summary_table(summaries, engine_elo):
    print(f"{'Preset':<18} {'Games':<6} {'W':<4} {'D':<4} {'L':<4} {'Score %':<9} {'Est. Elo':<10} {'Avg Plies':<10} {'Avg Time(s)':<12} {'Avg Nodes'}")
    print("-" * 102)
    for summary in summaries:
        elo = estimate_elo(engine_elo, summary["score_pct"] / 100.0)
        elo_str = str(elo) if elo is not None else "n/a*"
        print(
            f"{summary['preset']:<18} "
            f"{summary['games']:<6} "
            f"{summary['wins']:<4} "
            f"{summary['draws']:<4} "
            f"{summary['losses']:<4} "
            f"{summary['score_pct']:<9.1f} "
            f"{elo_str:<10} "
            f"{summary['avg_plies']:<10.1f} "
            f"{summary['agent_avg_time_s']:<12.3f} "
            f"{summary['agent_avg_nodes']:.0f}"
        )
    if any(estimate_elo(engine_elo, s["score_pct"] / 100.0) is None for s in summaries):
        print("  * Elo undefined at 0% or 100% score — play more games for a reliable estimate.")


def plot_summaries(summaries, plot_output, show_plot):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    labels = [summary["preset"] for summary in summaries]
    score_pct = [summary["score_pct"] for summary in summaries]
    avg_nodes = [summary["agent_avg_nodes"] for summary in summaries]
    avg_time = [summary["agent_avg_time_s"] for summary in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(labels, score_pct, color="seagreen")
    axes[0].set_title("Score vs Engine")
    axes[0].set_ylabel("Score %")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, avg_nodes, color="steelblue")
    axes[1].set_title("Avg Nodes / Agent Move")
    axes[1].set_ylabel("Nodes")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, avg_time, color="darkorange")
    axes[2].set_title("Avg Time / Agent Move")
    axes[2].set_ylabel("Seconds")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(plot_output, dpi=150)
    print(f"Summary plot written to {plot_output}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def write_csv(records, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preset_name",
                "opening_name",
                "agent_color",
                "outcome",
                "score",
                "plies",
                "agent_moves",
                "agent_avg_time_s",
                "agent_avg_nodes",
                "engine_avg_time_s",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def discover_engine_path():
    repo_root = Path(__file__).resolve().parent.parent
    for pattern in DEFAULT_ENGINE_PATTERNS:
        matches = sorted(repo_root.glob(pattern))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Multiprocessing worker helpers
# ---------------------------------------------------------------------------

_worker_engine = None
_worker_limit = None


def _init_worker(engine_path, engine_elo, engine_time):
    """Initializer for each pool worker — spawns one Stockfish process."""
    global _worker_engine, _worker_limit

    _worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))

    try:
        _worker_engine.configure({"Threads": 1})
    except chess.engine.EngineError:
        pass

    elo_opt = _worker_engine.options.get("UCI_Elo")
    if elo_opt is not None:
        elo_min = elo_opt.min or 0
        elo_max = elo_opt.max or 9999
        clamped = max(elo_min, min(elo_max, engine_elo))
        try:
            _worker_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": clamped})
        except chess.engine.EngineError:
            pass

    _worker_limit = chess.engine.Limit(time=engine_time)


def _play_game_task(indexed_task):
    """Worker function — plays a single game using the process-local engine."""
    idx, (preset, opening_name, opening_fen, agent_color, max_plies) = indexed_task
    record = play_game(
        _worker_engine, preset, opening_name, opening_fen,
        agent_color, _worker_limit, max_plies,
    )
    return idx, record


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def _print_progress(done, total, start_time):
    elapsed = time.perf_counter() - start_time
    pct = done / total if total > 0 else 1.0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    if done > 0 and done < total:
        eta = elapsed * (total - done) / done
        eta_m, eta_s = divmod(int(eta), 60)
        eta_str = f"{eta_m}m {eta_s:02d}s" if eta_m else f"{eta_s}s"
    elif done >= total:
        eta_str = "0s"
    else:
        eta_str = "..."

    sys.stdout.write(
        f"\r  Playing games  [{bar}] {done}/{total}  "
        f"elapsed: {elapsed_str}  ETA: {eta_str}   "
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Validation (runs once in the main process before spawning workers)
# ---------------------------------------------------------------------------

def _validate_engine(engine_path, requested_elo):
    """Spawn a temporary engine to check UCI options and print warnings."""
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    clamped_elo = requested_elo
    try:
        elo_opt = engine.options.get("UCI_Elo")
        if elo_opt is not None:
            elo_min = elo_opt.min or 0
            elo_max = elo_opt.max or 9999
            clamped_elo = max(elo_min, min(elo_max, requested_elo))
            if clamped_elo != requested_elo:
                print(
                    f"Warning: requested Elo {requested_elo} is outside engine range "
                    f"[{elo_min}, {elo_max}]. Clamping to {clamped_elo}."
                )
        else:
            print("Warning: this engine does not support UCI_LimitStrength/UCI_Elo; engine runs at full strength.")
    finally:
        engine.quit()
    return clamped_elo


# ---------------------------------------------------------------------------
# Per-game result printing (after all games finish)
# ---------------------------------------------------------------------------

def _print_game_results(records, presets):
    for preset in presets:
        print(
            f"[{preset.name}] depth={preset.config.depth}, "
            f"alpha_beta={preset.config.use_alpha_beta}, "
            f"move_ordering={preset.config.move_ordering}"
        )
        for record in records:
            if record.preset_name != preset.name:
                continue
            print(
                f"  {record.opening_name:<28} "
                f"agent={record.agent_color:<5} "
                f"result={record.outcome:<9} "
                f"plies={record.plies:<3} "
                f"avg_nodes={record.agent_avg_nodes:.0f}"
            )
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark this chess agent against a UCI engine."
    )
    parser.add_argument(
        "--engine-path",
        help="Path to the UCI engine binary. If omitted, searches for a repo-local Stockfish executable.",
    )
    parser.add_argument(
        "--engine-elo",
        type=int,
        default=ENGINE_ELO,
        help=f"Target engine Elo when UCI_LimitStrength is supported. (default: {ENGINE_ELO})",
    )
    parser.add_argument(
        "--engine-time",
        type=float,
        default=ENGINE_TIME,
        help=f"Per-move engine think time in seconds. (default: {ENGINE_TIME})",
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=NUM_OPENINGS,
        help=f"Number of curated opening positions to include. (default: {NUM_OPENINGS})",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=MAX_PLIES,
        help=f"Declare a truncated draw after this many plies. (default: {MAX_PLIES})",
    )
    parser.add_argument(
        "--output",
        default=VS_ENGINE_CSV,
        help=f"CSV file for detailed game results. (default: {VS_ENGINE_CSV})",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Custom config as 'name,depth,use_alpha_beta,move_ordering'. Overrides settings.py if provided.",
    )
    parser.add_argument(
        "--plot-output",
        default=VS_ENGINE_PLOT,
        help=f"PNG file for the summary plot. (default: {VS_ENGINE_PLOT})",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the matplotlib window after saving the summary plot.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel worker processes. (default: {MAX_WORKERS or 'CPU count'})",
    )
    args = parser.parse_args()

    presets = [parse_config_spec(spec) for spec in args.config] if args.config else build_default_presets()
    openings = build_opening_suite(args.openings)
    output_path = Path(args.output)
    engine_path = Path(args.engine_path) if args.engine_path else discover_engine_path()

    if engine_path is None:
        raise SystemExit(
            "No engine executable found. Put Stockfish in stockfish/ or pass --engine-path."
        )

    # Build the flat list of game tasks.
    tasks = [
        (preset, opening_name, opening_fen, agent_color, args.max_plies)
        for preset in presets
        for opening_name, opening_fen in openings
        for agent_color in (chess.WHITE, chess.BLACK)
    ]

    num_workers = min(args.workers or os.cpu_count() or 1, len(tasks))

    # Validate engine options once (prints warnings) before spawning workers.
    clamped_elo = _validate_engine(engine_path, args.engine_elo)

    print("Benchmarking agent against external engine")
    print(f"Engine path:          {engine_path}")
    print(f"Engine target Elo:    {clamped_elo}")
    print(f"Engine time per move: {args.engine_time:.3f}s")
    print(f"Openings tested:      {len(openings)}")
    print(f"Configs tested:       {len(presets)}")
    print(f"Total games:          {len(tasks)}")
    print(f"Worker processes:     {num_workers}")
    print()

    # Map from task index -> record so we can reconstruct insertion order.
    results_by_index = {}
    start_time = time.perf_counter()

    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(str(engine_path), clamped_elo, args.engine_time),
    )

    indexed_tasks = list(enumerate(tasks))
    _print_progress(0, len(tasks), start_time)
    for idx, record in pool.imap_unordered(_play_game_task, indexed_tasks):
        results_by_index[idx] = record
        _print_progress(len(results_by_index), len(tasks), start_time)

    pool.terminate()
    pool.join()

    # Clear the progress line.
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Rebuild records in the original deterministic order.
    records = [results_by_index[i] for i in range(len(tasks))]

    total_elapsed = time.perf_counter() - start_time
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"All {len(records)} games finished in {mins}m {secs:02d}s.\n")

    _print_game_results(records, presets)

    write_csv(records, output_path)
    summaries = collect_summaries(records, presets)

    print_summary_table(summaries, clamped_elo)
    plot_summaries(summaries, Path(args.plot_output), args.show_plot)

    print()
    print(f"Detailed results written to {output_path}")


if __name__ == "__main__":
    main()
