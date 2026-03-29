"""
vs_stockfish.py
---------------
Entry point: benchmark the chess agent against a UCI engine (e.g. Stockfish).

Tests multiple search configurations across a set of opening positions and
prints a W/D/L summary table with estimated Elo ratings.

HOW IT WORKS
------------
For each (preset, opening, color) combination a game is played in parallel
using a pool of worker processes (one Stockfish subprocess per worker).
Results are collected in order, written to CSV, and plotted.

The three search techniques are defined in config/settings.py AGENT_CONFIGS:
  1. Pure Minimax            — exhaustive tree search
  2. Minimax + Alpha-Beta    — prunes branches that can't affect the result
  3. Alpha-Beta + Move Order — searches promising moves first to prune more

Default parameters live in config/settings.py — edit that file to change
depths, Elo, number of openings, etc. without touching this file.

Usage (uses settings.py defaults):
    python benchmarks/vs_stockfish.py

Usage (override engine path):
    python benchmarks/vs_stockfish.py --engine-path "C:\\path\\to\\stockfish.exe"

Usage (custom agent configs via CLI):
    python benchmarks/vs_stockfish.py ^
        --config ab_d3,3,true,false ^
        --config ab_d3_ordered,3,true,true
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Allow imports from the repo root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess

from agent.search import SearchConfig
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
from benchmarks.models import AgentPreset
from benchmarks.game import build_opening_suite
from benchmarks.worker import discover_engine_path, _validate_engine, _init_worker, _play_game_task
from benchmarks.reporting import (
    collect_summaries,
    print_summary_table,
    _print_game_results,
    _print_progress,
    write_csv,
    plot_summaries,
)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_bool(value):
    """Accept common truthy/falsy strings from CLI arguments."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_config_spec(spec):
    """
    Parse a 'name,depth,use_alpha_beta,move_ordering' CLI string into an AgentPreset.

    Example: 'ab_d3,3,true,false'
    """
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
            use_alpha_beta=_parse_bool(use_alpha_beta),
            move_ordering=_parse_bool(move_ordering),
        ),
    )


def _build_default_presets():
    """Convert the AGENT_CONFIGS list from settings.py into AgentPreset objects."""
    return [
        AgentPreset(name, SearchConfig(depth=depth, use_alpha_beta=ab, move_ordering=mo))
        for name, depth, ab, mo in AGENT_CONFIGS
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    presets = (
        [_parse_config_spec(spec) for spec in args.config]
        if args.config
        else _build_default_presets()
    )
    openings    = build_opening_suite(args.openings)
    output_path = Path(args.output)
    engine_path = Path(args.engine_path) if args.engine_path else discover_engine_path()

    if engine_path is None:
        raise SystemExit(
            "No engine executable found. Put Stockfish in stockfish/ or pass --engine-path."
        )

    # One task per (preset × opening × color).
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
    print("Press Ctrl+C to stop early\n")

    # Run games in parallel, collecting results as they complete.
    results_by_index = {}
    start_time = time.perf_counter()

    pool = Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(str(engine_path), clamped_elo, args.engine_time),
    )

    indexed_tasks  = list(enumerate(tasks))
    async_results  = [pool.apply_async(_play_game_task, (task,)) for task in indexed_tasks]
    pending        = set(range(len(async_results)))
    _print_progress(0, len(tasks), start_time)

    try:
        while pending:
            for i in list(pending):
                if async_results[i].ready():
                    idx, record = async_results[i].get()
                    results_by_index[idx] = record
                    pending.discard(i)
                    _print_progress(len(results_by_index), len(tasks), start_time)
            if pending:
                time.sleep(0.05)  # short sleep keeps the loop interruptible on Windows
    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 80 + "\r")
        print("Stopping...", flush=True)
        pool.terminate()
        pool.join()
        raise SystemExit(1)

    pool.terminate()
    pool.join()

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    # Reconstruct records in original deterministic order.
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
