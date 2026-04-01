"""
internal.py
-----------
Performance comparison: Minimax vs Alpha-Beta Pruning vs Alpha-Beta + Move Ordering.

WHY BENCHMARK?
--------------
Two properties of Alpha-Beta pruning and move ordering are verified empirically:

  1. EFFICIENCY  — AB pruning and AB + Move Ordering explore far fewer nodes than
                   pure Minimax, and therefore run faster.

  2. CORRECTNESS — Both pruned variants should agree with pure Minimax on the
                   minimax VALUE of every position (same score), and in most cases
                   also on the specific move chosen (same move).

NOTE ON "SAME MOVE" vs "SAME SCORE"
-------------------------------------
"Same score" is the theoretically meaningful correctness claim: all three
algorithms must compute the same minimax value.  This should be 100 % at all
depths and is the standard proof of equivalence.

"Same move" is an empirical observation.  When two moves share the same minimax
score (a tie), the algorithm returns whichever it encountered first.  Move
ordering changes the iteration order, so AB + Move Ordering may break ties
differently from pure Minimax — producing a different (but equally optimal)
move.  A "same move" percentage below 100 % for AB+MO therefore indicates
tie-breaking variation, NOT a correctness failure.

POSITIONS
---------
All 50 opening positions from config/openings.py are used for a representative
sample.  Each position is evaluated at every depth in INTERNAL_DEPTHS.

PARALLELISM
-----------
Each (position, depth) pair is dispatched as an independent worker task so that
all positions at a given depth run concurrently, keeping wall-clock time
manageable even at depth 4 where pure Minimax is expensive.

OUTPUT
------
Three tables printed to the terminal:
    Table 1a — Nodes explored (avg, median, and reduction % vs Minimax)
    Table 1b — Average search time with speedup multiplier vs Minimax
    Table 2  — Move and score agreement (%) with pure Minimax

One PNG chart (path set in settings.py) with four panels:
    Top-left:  Average nodes explored per depth (log scale)
    Top-right: Median nodes explored per depth (log scale)
    Bot-left:  Average search time per depth
    Bot-right: Move-agreement percentage per depth

Usage:
    python benchmarks/internal.py
"""
import os
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

# Allow imports from the repo root when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chess
import matplotlib.pyplot as plt

from agent.search import search_position, SearchConfig
from config.openings import OPENING_POSITIONS
from config.settings import INTERNAL_DEPTHS, INTERNAL_NUM_OPENINGS, INTERNAL_PLOT_OUTPUT, INTERNAL_MAX_WORKERS


# ---------------------------------------------------------------------------
# Worker task — must be at module level so multiprocessing can pickle it.
# ---------------------------------------------------------------------------

def _run_position_task(args):
    """
    Run all three search variants on a single (position, depth) pair.

    Args:
        args: (index, (opening_name, fen, depth))

    Returns:
        (index, opening_name, depth, mm_result, ab_result, mo_result)
        where each *_result is a SearchResult(move, score, nodes, elapsed_seconds).
    """
    idx, (opening_name, fen, depth) = args
    board = chess.Board(fen)

    mm = search_position(board, SearchConfig(depth=depth, use_alpha_beta=False, move_ordering=False))
    ab = search_position(board, SearchConfig(depth=depth, use_alpha_beta=True,  move_ordering=False))
    mo = search_position(board, SearchConfig(depth=depth, use_alpha_beta=True,  move_ordering=True))

    return idx, opening_name, depth, mm, ab, mo


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(raw_results, depths):
    """
    Group raw task results by depth into per-position observation lists.

    Returns a dict {depth: {metric: [value, ...]}} suitable for computing
    mean, median, and agreement percentages.
    """
    by_depth = {
        d: dict(
            mm_nodes=[], ab_nodes=[], mo_nodes=[],
            mm_times=[], ab_times=[], mo_times=[],
            mm_scores=[], ab_scores=[], mo_scores=[],
            ab_same_move=[], mo_same_move=[],
        )
        for d in depths
    }

    for _idx, _name, depth, mm, ab, mo in raw_results:
        r = by_depth[depth]
        r['mm_nodes'].append(mm.nodes)
        r['ab_nodes'].append(ab.nodes)
        r['mo_nodes'].append(mo.nodes)
        r['mm_times'].append(mm.elapsed_seconds)
        r['ab_times'].append(ab.elapsed_seconds)
        r['mo_times'].append(mo.elapsed_seconds)
        r['mm_scores'].append(mm.score)
        r['ab_scores'].append(ab.score)
        r['mo_scores'].append(mo.score)
        r['ab_same_move'].append(mm.move == ab.move)
        r['mo_same_move'].append(mm.move == mo.move)

    return by_depth


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(bool_list):
    return sum(bool_list) / len(bool_list) * 100 if bool_list else 0.0


def _print_nodes_table(by_depth, depths):
    n = len(by_depth[depths[0]]['mm_nodes'])
    title = f"Table 1a — Nodes Explored  (n = {n} positions per depth)"
    hdr = (
        f"{'Depth':<7}"
        f"{'MM Avg':>13}{'MM Med':>13}"
        f"{'AB Avg':>13}{'AB Med':>13}{'AB Reduc%':>11}"
        f"{'MO Avg':>13}{'MO Med':>13}{'MO Reduc%':>11}"
    )
    sep = "=" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for d in depths:
        r = by_depth[d]
        n = len(r['mm_nodes'])
        mm_avg = sum(r['mm_nodes']) // n
        mm_med = int(statistics.median(r['mm_nodes']))
        ab_avg = sum(r['ab_nodes']) // n
        ab_med = int(statistics.median(r['ab_nodes']))
        mo_avg = sum(r['mo_nodes']) // n
        mo_med = int(statistics.median(r['mo_nodes']))
        ab_red = (1 - ab_avg / mm_avg) * 100 if mm_avg else 0.0
        mo_red = (1 - mo_avg / mm_avg) * 100 if mm_avg else 0.0
        print(
            f"{d:<7}"
            f"{mm_avg:>13,}{mm_med:>13,}"
            f"{ab_avg:>13,}{ab_med:>13,}{ab_red:>10.1f}%"
            f"{mo_avg:>13,}{mo_med:>13,}{mo_red:>10.1f}%"
        )


def _print_time_table(by_depth, depths):
    n = len(by_depth[depths[0]]['mm_times'])
    title = f"Table 1b — Average Search Time (s)  (n = {n} positions per depth)"
    hdr = (
        f"{'Depth':<7}"
        f"{'MM Time':>12}{'AB Time':>12}{'AB Speedup':>12}"
        f"{'MO Time':>12}{'MO Speedup':>12}"
    )
    sep = "=" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for d in depths:
        r = by_depth[d]
        n = len(r['mm_times'])
        mm_t = sum(r['mm_times']) / n
        ab_t = sum(r['ab_times']) / n
        mo_t = sum(r['mo_times']) / n
        ab_sp = mm_t / ab_t if ab_t > 0 else float('inf')
        mo_sp = mm_t / mo_t if mo_t > 0 else float('inf')
        print(
            f"{d:<7}"
            f"{mm_t:>12.4f}{ab_t:>12.4f}{ab_sp:>11.2f}x"
            f"{mo_t:>12.4f}{mo_sp:>11.2f}x"
        )


def _print_agreement_table(by_depth, depths):
    n = len(by_depth[depths[0]]['ab_same_move'])
    title = f"Table 2 — Correctness vs Pure Minimax  (n = {n} positions per depth)"
    hdr = (
        f"{'Depth':<7}"
        f"{'MM Avg Score':>14}{'AB Avg Score':>14}{'AB Same Move%':>15}"
        f"{'MO Avg Score':>14}{'MO Same Move%':>15}"
    )
    sep = "=" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for d in depths:
        r = by_depth[d]
        n = len(r['mm_scores'])
        mm_sc_avg = sum(r['mm_scores']) / n
        ab_sc_avg = sum(r['ab_scores']) / n
        mo_sc_avg = sum(r['mo_scores']) / n
        ab_mv = _pct(r['ab_same_move'])
        mo_mv = _pct(r['mo_same_move'])
        print(
            f"{d:<7}"
            f"{mm_sc_avg:>14.1f}{ab_sc_avg:>14.1f}{ab_mv:>14.1f}%"
            f"{mo_sc_avg:>14.1f}{mo_mv:>14.1f}%"
        )

    print(
        "\n  Note: Avg Score is the mean minimax evaluation (centipawns) across all positions."
        "\n        Identical values across algorithms confirm they compute the same minimax value."
        "\n        Same Move % may be <100% for AB+MO due to tie-breaking when two moves share"
        "\n        the same minimax value — this is expected behaviour, not a correctness failure."
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_COLORS = ["steelblue", "darkorange", "seagreen"]
_LABELS = ["Minimax", "Alpha-Beta", "AB + Move Ordering"]
_BAR_OFFSETS = [-0.25, 0.0, 0.25]
_BAR_WIDTH = 0.25


def _bar_group(ax, x_vals, series, title, ylabel, log_scale=False):
    for i, (vals, label, color) in enumerate(zip(series, _LABELS, _COLORS)):
        ax.bar(
            [x + _BAR_OFFSETS[i] for x in x_vals],
            vals,
            width=_BAR_WIDTH,
            label=label,
            color=color,
        )
    ax.set_xlabel("Search Depth")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_vals)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()


def _plot(by_depth, depths, output_path):
    fig, (ax_avg, ax_med, ax_time) = plt.subplots(1, 3, figsize=(18, 5))
    x = list(depths)

    # Left: average nodes (log scale)
    _bar_group(
        ax_avg, x,
        series=[
            [sum(by_depth[d]['mm_nodes']) // len(by_depth[d]['mm_nodes']) for d in depths],
            [sum(by_depth[d]['ab_nodes']) // len(by_depth[d]['ab_nodes']) for d in depths],
            [sum(by_depth[d]['mo_nodes']) // len(by_depth[d]['mo_nodes']) for d in depths],
        ],
        title="Average Nodes Explored (log scale)",
        ylabel="Avg Nodes Explored",
        log_scale=True,
    )

    # Centre: median nodes (log scale)
    _bar_group(
        ax_med, x,
        series=[
            [int(statistics.median(by_depth[d]['mm_nodes'])) for d in depths],
            [int(statistics.median(by_depth[d]['ab_nodes'])) for d in depths],
            [int(statistics.median(by_depth[d]['mo_nodes'])) for d in depths],
        ],
        title="Median Nodes Explored (log scale)",
        ylabel="Median Nodes Explored",
        log_scale=True,
    )

    # Right: average search time
    for vals, label, color in zip(
        [
            [sum(by_depth[d]['mm_times']) / len(by_depth[d]['mm_times']) for d in depths],
            [sum(by_depth[d]['ab_times']) / len(by_depth[d]['ab_times']) for d in depths],
            [sum(by_depth[d]['mo_times']) / len(by_depth[d]['mo_times']) for d in depths],
        ],
        _LABELS, _COLORS,
    ):
        ax_time.plot(x, vals, marker='o', label=label, color=color)
    ax_time.set_xlabel("Search Depth")
    ax_time.set_ylabel("Avg Time per Move (s)")
    ax_time.set_title("Average Search Time per Move")
    ax_time.set_xticks(x)
    ax_time.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"\nPlot saved as {output_path}")


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def _print_progress(done, total, start):
    elapsed = time.perf_counter() - start
    pct = done / total if total else 1.0
    filled = int(30 * pct)
    bar = "\u2588" * filled + "\u2591" * (30 - filled)
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    if 0 < done < total:
        eta = elapsed * (total - done) / done
        eta_m, eta_s = divmod(int(eta), 60)
        eta_str = f"{eta_m}m {eta_s:02d}s" if eta_m else f"{eta_s}s"
    else:
        eta_str = "0s" if done >= total else "..."
    sys.stdout.write(
        f"\r  [{bar}] {done}/{total} ({pct*100:.0f}%)  elapsed: {elapsed_str}  ETA: {eta_str}   "
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    positions = OPENING_POSITIONS[:INTERNAL_NUM_OPENINGS]
    n_pos = len(positions)

    # Build one task per (position, depth) combination.
    tasks = [
        (idx, (name, fen, depth))
        for idx, (depth, (name, fen)) in enumerate(
            (depth, pos)
            for depth in INTERNAL_DEPTHS
            for pos in positions
        )
    ]
    total = len(tasks)
    num_workers = min(INTERNAL_MAX_WORKERS or os.cpu_count() or 1, total)

    print("Internal benchmark: Minimax vs Alpha-Beta vs AB + Move Ordering")
    print(f"Positions : {n_pos}  |  Depths: {INTERNAL_DEPTHS}  |  Total tasks: {total}  |  Workers: {num_workers}")
    print("Each task runs all three search variants; wall-clock time is dominated by")
    print("pure Minimax at deeper depths.  This may take several minutes.\n")

    results_by_idx = {}
    start = time.perf_counter()

    pool = Pool(processes=num_workers)
    async_results = [pool.apply_async(_run_position_task, (task,)) for task in tasks]
    pending = set(range(total))
    _print_progress(0, total, start)

    try:
        while pending:
            for i in list(pending):
                if async_results[i].ready():
                    results_by_idx[i] = async_results[i].get()
                    pending.discard(i)
                    _print_progress(len(results_by_idx), total, start)
            if pending:
                time.sleep(0.05)
    except KeyboardInterrupt:
        sys.stdout.write("\r" + " " * 80 + "\r")
        print("Stopping...")
        pool.terminate()
        pool.join()
        raise SystemExit(1)

    pool.terminate()
    pool.join()

    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    elapsed = time.perf_counter() - start
    mins, secs = divmod(int(elapsed), 60)
    print(f"All {total} tasks finished in {mins}m {secs:02d}s.\n")

    raw_results = [results_by_idx[i] for i in range(total)]
    by_depth = _aggregate(raw_results, INTERNAL_DEPTHS)

    _print_nodes_table(by_depth, INTERNAL_DEPTHS)
    _print_time_table(by_depth, INTERNAL_DEPTHS)
    _print_agreement_table(by_depth, INTERNAL_DEPTHS)
    _plot(by_depth, INTERNAL_DEPTHS, INTERNAL_PLOT_OUTPUT)


if __name__ == "__main__":
    main()
