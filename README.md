# Chess AI — Minimax with Alpha-Beta Pruning

A playable chess game where you face an AI that uses the **Minimax** algorithm with **Alpha-Beta pruning**.

## Requirements

- Python 3.10 or higher

Install all dependencies with:

```bash
pip install pygame-ce python-chess matplotlib
```

**Note:** Use `pygame-ce` (the community edition), not the original `pygame`. The original does not support Python 3.12+.

## How to Run

### Play the game
```bash
python gui.py
```
1. Choose to play as **White** or **Black**.
2. Choose a difficulty:
   - Easy — depth 1 (fast, weak)
   - Medium — depth 3 (balanced)
   - Hard — depth 5 (strong, slower)
3. Click a piece to select it, then click the destination square to move.

### Run the benchmark
```bash
python benchmark.py
```
Compares pure Minimax vs Alpha-Beta pruning across depths 1–4.
Prints a results table and saves a chart as `benchmark_results.png`.

## Project Structure

```
├── gui.py           # Menu screens (colour + difficulty selection)
├── game.py          # Board rendering and game loop
├── search.py        # Minimax + Alpha-Beta search algorithm
├── evaluation.py    # Heuristic board evaluation function
├── benchmark.py     # Performance comparison script
└── icons/           # Piece PNG images (must be present to run the game)
```

## How the AI Works

1. **Minimax** — the AI builds a game tree to a fixed depth, assuming both players always play their best move. White maximises the score; Black minimises it.
2. **Alpha-Beta pruning** — skips branches that cannot affect the final decision, dramatically reducing the nodes searched without changing the result.
3. **Evaluation function** — when the depth limit is reached, the board is scored using piece material values plus piece-square tables that reward good positioning.
