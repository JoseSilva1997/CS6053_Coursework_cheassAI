"""
reporting.py
------------
Output and analysis for the vs-engine benchmark.

Provides four categories of output:

  Progress  — _print_progress()       : live progress bar during game collection.
  Terminal  — print_summary_table()   : per-preset W/D/L and estimated Elo.
              _print_game_results()   : per-game outcome lines grouped by preset.
  File I/O  — write_csv()             : full per-game data written to CSV.
  Plot      — plot_summaries()        : three-panel bar chart saved as PNG.

The Elo estimate is derived from the standard Elo expected-score formula:
    E = 1 / (1 + 10^((opponent_elo - player_elo) / 400))
Rearranged to solve for player_elo given E (= score_pct).
"""
import csv
import math
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise_records(records, preset_name):
    """
    Aggregate all GameRecords for one preset into a summary dict.

    Returns a dict with keys: preset, games, wins, draws, losses,
    score_pct, avg_plies, agent_avg_time_s, agent_avg_nodes.
    """
    subset = [r for r in records if r.preset_name == preset_name]
    games = len(subset)
    wins   = sum(1 for r in subset if r.outcome == "win")
    draws  = sum(1 for r in subset if r.outcome in {"draw", "truncated"})
    losses = sum(1 for r in subset if r.outcome == "loss")
    avg_score = sum(r.score for r in subset) / games if games else 0.0
    avg_plies = sum(r.plies for r in subset) / games if games else 0.0
    avg_time  = sum(r.agent_avg_time_s for r in subset) / games if games else 0.0
    avg_nodes = sum(r.agent_avg_nodes  for r in subset) / games if games else 0.0

    return {
        "preset":           preset_name,
        "games":            games,
        "wins":             wins,
        "draws":            draws,
        "losses":           losses,
        "score_pct":        avg_score * 100.0,
        "avg_plies":        avg_plies,
        "agent_avg_time_s": avg_time,
        "agent_avg_nodes":  avg_nodes,
    }


def collect_summaries(records, presets):
    """Return one summary dict per preset, in preset order."""
    return [summarise_records(records, preset.name) for preset in presets]


def estimate_elo(opponent_elo, score_pct):
    """
    Estimate the agent's Elo from its score percentage against a known-Elo opponent.

    Derived from the Elo expected-score formula:
        E = 1 / (1 + 10^((opponent - player) / 400))

    Solving for player_elo:
        player_elo = opponent_elo - 400 * log10(1/score_pct - 1)

    Returns None if score_pct is 0% or 100% (formula is undefined at extremes).
    """
    if score_pct <= 0.0 or score_pct >= 1.0:
        return None
    return round(opponent_elo - 400 * math.log10(1.0 / score_pct - 1))


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_summary_table(summaries, engine_elo):
    """Print a formatted W/D/L summary table with estimated Elo for each preset."""
    print(
        f"{'Preset':<18} {'Games':<6} {'W':<4} {'D':<4} {'L':<4} "
        f"{'Score %':<9} {'Est. Elo':<10} {'Avg Plies':<10} {'Avg Time(s)':<12} {'Avg Nodes'}"
    )
    print("-" * 102)
    for s in summaries:
        elo = estimate_elo(engine_elo, s["score_pct"] / 100.0)
        elo_str = str(elo) if elo is not None else "n/a*"
        print(
            f"{s['preset']:<18} "
            f"{s['games']:<6} "
            f"{s['wins']:<4} "
            f"{s['draws']:<4} "
            f"{s['losses']:<4} "
            f"{s['score_pct']:<9.1f} "
            f"{elo_str:<10} "
            f"{s['avg_plies']:<10.1f} "
            f"{s['agent_avg_time_s']:<12.3f} "
            f"{s['agent_avg_nodes']:.0f}"
        )
    if any(estimate_elo(engine_elo, s["score_pct"] / 100.0) is None for s in summaries):
        print(
            "  * Elo undefined at 0% or 100% score — "
            "play more games for a reliable estimate."
        )


def _print_game_results(records, presets):
    """Print per-game outcome lines grouped by preset."""
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


def _print_progress(done, total, start_time):
    """Overwrite the current line with a progress bar, elapsed time, and ETA."""
    elapsed = time.perf_counter() - start_time
    pct = done / total if total > 0 else 1.0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    if 0 < done < total:
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
# File output
# ---------------------------------------------------------------------------

def write_csv(records, output_path: Path):
    """Write all GameRecords to a CSV file at output_path."""
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


def plot_summaries(summaries, plot_output: Path, show_plot: bool):
    """
    Save a three-panel PNG summarising score, nodes, and time per preset.

    Panels:
      Left   — Score % vs engine (higher is better).
      Centre — Average nodes searched per agent move (lower = more efficient).
      Right  — Average wall-clock time per agent move (lower = faster).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    labels     = [s["preset"]           for s in summaries]
    score_pct  = [s["score_pct"]        for s in summaries]
    avg_nodes  = [s["agent_avg_nodes"]  for s in summaries]
    avg_time   = [s["agent_avg_time_s"] for s in summaries]

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
