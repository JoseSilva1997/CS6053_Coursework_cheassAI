"""
worker.py
---------
Multiprocessing support for the vs-engine benchmark.

Each worker process owns one Stockfish subprocess so that games run in
parallel without sharing engine state.  This module provides:

  discover_engine_path()  — find a Stockfish executable in the repo.
  _validate_engine()      — pre-flight check: warn if Elo is out of range.
  _init_worker()          — pool initializer that spawns one engine per process.
  _play_game_task()       — pool task that plays one game using the local engine.
"""
import signal
import sys
from pathlib import Path

import chess
import chess.engine

from .game import play_game


# Glob patterns searched (relative to repo root) when no --engine-path is given.
DEFAULT_ENGINE_PATTERNS = [
    "stockfish/stockfish*.exe",
    "stockfish/*.exe",
    "engines/stockfish*.exe",
    "engines/*.exe",
    "stockfish*.exe",
]

# Process-local engine and limit — set once by _init_worker, reused for every task.
_worker_engine = None
_worker_limit = None


def discover_engine_path():
    """
    Search the repo for a Stockfish executable using DEFAULT_ENGINE_PATTERNS.

    Returns the first match as a Path, or None if nothing is found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for pattern in DEFAULT_ENGINE_PATTERNS:
        matches = sorted(repo_root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _validate_engine(engine_path, requested_elo):
    """
    Spawn a temporary engine instance to check UCI_Elo support.

    Prints a warning if the requested Elo is outside the engine's supported
    range or if the engine does not support strength limiting at all.

    Returns the clamped Elo that will actually be used.
    """
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
            print(
                "Warning: this engine does not support UCI_LimitStrength/UCI_Elo; "
                "engine runs at full strength."
            )
    finally:
        engine.quit()
    return clamped_elo


def _init_worker(engine_path, engine_elo, engine_time):
    """
    Pool initializer — called once per worker process at startup.

    Spawns a Stockfish subprocess, configures its Elo and thread count,
    and stores the engine + time limit in process-local globals so that
    every game task can reuse the same engine without re-launching it.

    On Windows, Stockfish is placed in a new process group so that
    Ctrl+C (broadcast to the whole console group) doesn't kill it directly —
    pool.terminate() handles clean shutdown instead.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # let main process handle Ctrl+C
    global _worker_engine, _worker_limit

    popen_kwargs = {}
    if sys.platform == "win32":
        import subprocess as _subprocess
        popen_kwargs["creationflags"] = _subprocess.CREATE_NEW_PROCESS_GROUP

    _worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path), **popen_kwargs)

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
    """
    Pool task — plays one game and returns (original_index, GameRecord).

    The index is passed through so the main process can reconstruct results
    in their original deterministic order after parallel collection.
    """
    idx, (preset, opening_name, opening_fen, agent_color, max_plies) = indexed_task
    record = play_game(
        _worker_engine, preset, opening_name, opening_fen,
        agent_color, _worker_limit, max_plies,
    )
    return idx, record
