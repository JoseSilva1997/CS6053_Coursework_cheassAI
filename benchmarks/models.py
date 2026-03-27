"""
models.py
---------
Shared data classes for the vs-engine benchmark.

AgentPreset  — pairs a human-readable name with a SearchConfig so results
               can be labelled clearly in tables and plots.

GameRecord   — one row of raw data produced by play_game(); written to CSV
               and aggregated by the reporting module.
"""
from dataclasses import dataclass

from agent.search import SearchConfig


@dataclass(frozen=True)
class AgentPreset:
    """A named search configuration to benchmark."""
    name: str
    config: SearchConfig


@dataclass
class GameRecord:
    """Statistics captured from a single agent-vs-engine game."""
    preset_name: str
    opening_name: str
    agent_color: str        # "white" or "black"
    outcome: str            # "win", "draw", "loss", or "truncated"
    score: float            # 1.0 win / 0.5 draw or truncated / 0.0 loss
    plies: int
    agent_moves: int
    agent_avg_time_s: float
    agent_avg_nodes: float
    engine_avg_time_s: float
