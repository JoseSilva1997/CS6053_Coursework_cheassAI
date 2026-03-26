"""
settings.py
-----------
Central configuration for all benchmark parameters.

Edit this file to change how benchmarks run — no CLI flags needed.
Both internal.py and vs_stockfish.py read their defaults from here.

AGENT_CONFIGS controls which search configurations are tested.
Format per entry: (name, depth, use_alpha_beta, move_ordering)
"""

# ---------------------------------------------------------------------------
# Agent configurations
# Each entry defines one version of the AI agent to test.
# ---------------------------------------------------------------------------
AGENT_CONFIGS = [
    # name              depth  alpha_beta  move_ordering
    ("mm_d2",           2,     False,      False),   # pure Minimax, depth 2
    ("ab_d2",           2,     True,       False),   # Alpha-Beta, depth 2
    ("ab_d3",           3,     True,       False),   # Alpha-Beta, depth 3
    ("ab_d3_ordered",   3,     True,       True),    # Alpha-Beta, depth 3 + move ordering
]

# ---------------------------------------------------------------------------
# Internal benchmark (Minimax vs Alpha-Beta efficiency comparison)
# ---------------------------------------------------------------------------
INTERNAL_DEPTHS = [1, 2, 3, 4]          # depth 5 excluded — pure Minimax is too slow
INTERNAL_PLOT_OUTPUT = "internal_benchmark.png"

# ---------------------------------------------------------------------------
# Stockfish benchmark (agent strength vs external engine)
# ---------------------------------------------------------------------------
ENGINE_ELO       = 1320    # target Elo for Stockfish — minimum supported by this build is 1320
ENGINE_TIME      = 0.05     # seconds per engine move
NUM_OPENINGS     = 6        # number of opening positions to test (from config/openings.py)
MAX_PLIES        = 160      # truncate game after this many half-moves (avoids infinite draws)
VS_ENGINE_CSV    = "engine_benchmark_results.csv"
VS_ENGINE_PLOT   = "engine_benchmark_summary.png"
