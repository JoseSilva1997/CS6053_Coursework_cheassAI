"""
internal.py
-----------
Performance comparison: pure Minimax vs Minimax with Alpha-Beta Pruning.

WHY BENCHMARK?
--------------
Alpha-Beta pruning is claimed to dramatically reduce the number of nodes
the AI must evaluate compared to pure Minimax, without changing the move
that is chosen.  This script verifies both claims:
  1. Both algorithms pick the SAME best move (correctness check).
  2. Alpha-Beta explores far fewer nodes (efficiency check).

THREE TEST POSITIONS are used to average out position-specific variation:
  - Starting position (symmetric, many legal moves)
  - Italian Opening after 4 moves (typical early middlegame)
  - Queen's Gambit Declined (rich pawn structure)

Depths to test and output file are controlled by config/settings.py.

OUTPUT
------
  - Table printed to the terminal: depth, nodes, times, same move?
  - PNG chart (path set in settings.py): two side-by-side plots
      Left:  bar chart — nodes explored at each depth
      Right: line chart — time per move at each depth

Usage:
    python benchmarks/internal.py
"""
import sys
from pathlib import Path

# Allow imports from the repo root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import matplotlib.pyplot as plt

from agent.search import get_best_move, get_best_move_no_pruning
from config.openings import OPENING_POSITIONS
from config.settings import INTERNAL_DEPTHS, INTERNAL_PLOT_OUTPUT

# ---------------------------------------------------------------------------
# Test positions
# Three positions give a more representative average than a single position.
# ---------------------------------------------------------------------------
POSITIONS = [
    chess.Board(),                            # starting position
    chess.Board(OPENING_POSITIONS[0][1]),     # Italian Game
    chess.Board(OPENING_POSITIONS[1][1]),     # Queen's Gambit Declined
]

# Accumulators for plot data
mm_nodes  = []   # average nodes explored by pure Minimax
ab_nodes  = []   # average nodes explored by Alpha-Beta
mm_times  = []   # average time (seconds) for Minimax
ab_times  = []   # average time (seconds) for Alpha-Beta

print(f"{'Depth':<8} {'Minimax Nodes':<18} {'AlphaBeta Nodes':<18} {'MM Time(s)':<14} {'AB Time(s)':<12} {'Same Move?'}")
print("-" * 82)

for depth in INTERNAL_DEPTHS:
    total_mm_nodes = 0
    total_ab_nodes = 0
    total_mm_time  = 0
    total_ab_time  = 0
    all_same = True   # set to False if the two algorithms disagree on any position

    for board in POSITIONS:
        b = board.copy()   # copy so we don't modify the test position

        # Run pure Minimax (no pruning) — returns (move, nodes_visited, time)
        mm_move, mn, mt = get_best_move_no_pruning(b, depth)

        # Run Alpha-Beta Minimax
        ab_move, an, at = get_best_move(b, depth)

        total_mm_nodes += mn
        total_ab_nodes += an
        total_mm_time  += mt
        total_ab_time  += at

        # Both algorithms should always choose the same move
        if mm_move != ab_move:
            all_same = False

    # Average across the three test positions
    n = len(POSITIONS)
    avg_mm_nodes = total_mm_nodes // n
    avg_ab_nodes = total_ab_nodes // n
    avg_mm_time  = total_mm_time  / n
    avg_ab_time  = total_ab_time  / n

    mm_nodes.append(avg_mm_nodes)
    ab_nodes.append(avg_ab_nodes)
    mm_times.append(avg_mm_time)
    ab_times.append(avg_ab_time)

    print(f"{depth:<8} {avg_mm_nodes:<18} {avg_ab_nodes:<18} {avg_mm_time:<14.3f} {avg_ab_time:<12.3f} {'Yes' if all_same else 'No'}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: bar chart comparing nodes explored
ax1.bar([d - 0.2 for d in INTERNAL_DEPTHS], mm_nodes, width=0.4, label="Minimax",    color="steelblue")
ax1.bar([d + 0.2 for d in INTERNAL_DEPTHS], ab_nodes, width=0.4, label="Alpha-Beta", color="darkorange")
ax1.set_xlabel("Search Depth")
ax1.set_ylabel("Nodes Explored (avg)")
ax1.set_title("Nodes Explored: Minimax vs Alpha-Beta")
ax1.set_xticks(INTERNAL_DEPTHS)
ax1.legend()

# Right plot: line chart comparing wall-clock time per move
ax2.plot(INTERNAL_DEPTHS, mm_times, marker='o', label="Minimax",    color="steelblue")
ax2.plot(INTERNAL_DEPTHS, ab_times, marker='o', label="Alpha-Beta", color="darkorange")
ax2.set_xlabel("Search Depth")
ax2.set_ylabel("Time per Move (s, avg)")
ax2.set_title("Time per Move: Minimax vs Alpha-Beta")
ax2.set_xticks(INTERNAL_DEPTHS)
ax2.legend()

plt.tight_layout()
plt.savefig(INTERNAL_PLOT_OUTPUT, dpi=150)
plt.show()
print(f"\nPlot saved as {INTERNAL_PLOT_OUTPUT}")
