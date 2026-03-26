# Chess AI — Minimax with Alpha-Beta Pruning

CS6053 Artificial Intelligence and Machine Learning coursework.

A chess agent that uses **Minimax** (blind search baseline) and **Minimax with Alpha-Beta pruning** (informed search) to select moves. The agent is benchmarked both internally (efficiency) and externally against Stockfish (playing strength).

## Requirements

Python 3.10 or higher. Install dependencies:

```bash
pip install pygame-ce python-chess matplotlib
```

## Project Structure

```
agent/
  search.py       # Minimax and Alpha-Beta search algorithms
  evaluation.py   # Heuristic board evaluation (material + piece-square tables)

config/
  settings.py     # All benchmark parameters — edit this before running
  openings.py     # 50 curated opening positions (FENs) used in benchmarks

benchmarks/
  internal.py     # Efficiency comparison: Minimax vs Alpha-Beta (nodes, time)
  vs_stockfish.py # Strength benchmark: agent configurations vs Stockfish

stockfish/        # Place the Stockfish executable here
```

## Configuration

All parameters are in [config/settings.py](config/settings.py). Edit this file before running benchmarks:

```python
AGENT_CONFIGS = [
    ("mm_d2",          2, False, False),   # pure Minimax, depth 2
    ("ab_d2",          2, True,  False),   # Alpha-Beta, depth 2
    ("ab_d3",          3, True,  False),   # Alpha-Beta, depth 3
    ("ab_d3_ordered",  3, True,  True),    # Alpha-Beta, depth 3 + move ordering
]

ENGINE_ELO   = 1000   # Stockfish Elo limit
ENGINE_TIME  = 0.05   # seconds per engine move
NUM_OPENINGS = 6      # opening positions to test
```

## Running the Benchmarks

### Internal benchmark (Minimax vs Alpha-Beta efficiency)

```bash
python benchmarks/internal.py
```

Compares pure Minimax vs Alpha-Beta pruning across depths 1–4. Prints a results table and saves a chart as `internal_benchmark.png`.

### Stockfish benchmark (agent strength)

```bash
python benchmarks/vs_stockfish.py
```

The script auto-discovers `stockfish/stockfish*.exe`. To specify the path manually:

```bash
python benchmarks/vs_stockfish.py --engine-path "C:\path\to\stockfish.exe"
```

Outputs:
- Terminal summary table (W/D/L, score %, avg nodes, avg time per config)
- `engine_benchmark_summary.png` — bar charts for score, nodes, and time
- `engine_benchmark_results.csv` — full per-game results

All defaults come from `config/settings.py`. CLI flags override them when needed (see `--help`).

## How the AI Works

1. **Problem formulation** — Chess is modelled as a state space search problem. Each board position is a state; legal moves are transitions; the goal is to reach a winning terminal state.

2. **Minimax** — The AI builds a game tree to a fixed depth, assuming both players always play their best move. White maximises the score; Black minimises it.

3. **Alpha-Beta pruning** — Skips branches that cannot affect the final decision, dramatically reducing nodes searched without changing the chosen move.

4. **Evaluation function** — At the depth limit, the board is scored using piece material values plus piece-square tables that reward good positioning. This heuristic is what makes the search *informed* rather than blind.

## Original Contributions

| File | Description |
|------|-------------|
| `agent/search.py` | Minimax and Alpha-Beta implementation, `SearchConfig` dataclass |
| `agent/evaluation.py` | Heuristic evaluation with piece-square tables |
| `config/settings.py` | Centralised benchmark configuration |
| `config/openings.py` | Curated opening position suite |
| `benchmarks/internal.py` | Efficiency comparison benchmark |
| `benchmarks/vs_stockfish.py` | External engine strength benchmark |

The `python-chess` library is used for board representation, move generation, and UCI engine communication.
