"""
evaluation.py
-------------
Heuristic evaluation function for the chess AI.

The AI cannot search the entire game tree (too large), so when it reaches
the maximum search depth it calls evaluate() to estimate how good the
current board position is.

Return value convention:
  Positive  → White is better
  Negative  → Black is better
  0         → Equal / draw

The score is made up of two parts for every piece on the board:
  1. Material value  — how much the piece is worth (e.g. queen = 900)
  2. Positional bonus — how good the square is for that piece type
                        (piece-square tables, one per piece type)
"""
import chess

# ---------------------------------------------------------------------------
# Material values (in centipawns — 100 centipawns = 1 pawn)
# These reflect the standard chess piece values used in competitive play.
# The king is given a very high value so the AI never willingly loses it.
# ---------------------------------------------------------------------------
PIECE_VALUES = {
   chess.PAWN:   100,
   chess.KNIGHT: 320,
   chess.BISHOP: 330,
   chess.ROOK:   500,
   chess.QUEEN:  900,
   chess.KING:   20000
}

# ---------------------------------------------------------------------------
# Piece-square tables
# Each table is a list of 64 values, one per board square.
# The table is written from White's perspective with rank 8 at the top
# (index 0 = a8) and rank 1 at the bottom (index 63 = h1).
# Positive values reward the piece for being on that square;
# negative values penalise it.
# ---------------------------------------------------------------------------

# Pawns: rewarded for advancing toward promotion and controlling the centre.
# Penalised for blocking the centre (e.g. f/g pawns on rank 2).
PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

# Knights: strongly penalised on edges and corners ("a knight on the rim
# is dim"). Best placed in the centre where they control up to 8 squares.
KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

# Bishops: rewarded for open diagonals through the centre.
# Penalised on back rank and corners where diagonals are short.
BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

# Rooks: rewarded for the 7th rank (attacks enemy pawns) and open files.
# Penalised on passive squares away from open files.
ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
]

# Queen: slightly penalised early (discourages premature queen development).
# Modest central bonuses — the queen is powerful anywhere, not just centre.
QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

# King: strongly penalised in the centre (dangerous in middlegame).
# Rewarded for castled positions (g1/c1 area) where it is safely tucked away.
KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     30, 40, 10,  0,  0, 10, 40, 30
]

# Map each piece type constant to its piece-square table.
TABLES = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_TABLE
}

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

WHITE_SCORES = {
    piece_type: tuple(
        PIECE_VALUES[piece_type] + TABLES[piece_type][chess.square_mirror(square)]
        for square in chess.SQUARES
    )
    for piece_type in PIECE_TYPES
}
BLACK_SCORES = {
    piece_type: tuple(
        PIECE_VALUES[piece_type] + TABLES[piece_type][square]
        for square in chess.SQUARES
    )
    for piece_type in PIECE_TYPES
}

# Flat lookup arrays indexed by (piece_type * 64 + square).
# A single tuple index replaces a dict lookup + tuple index per piece,
# which is measurably faster in CPython's eval loop.
_WHITE_FLAT = tuple(
    WHITE_SCORES.get(pt, (0,) * 64)[sq]
    for pt in range(7)          # 0 is unused, 1-6 are piece types
    for sq in range(64)
)
_BLACK_FLAT = tuple(
    BLACK_SCORES.get(pt, (0,) * 64)[sq]
    for pt in range(7)
    for sq in range(64)
)

# Pre-bind constants used inside evaluate() to avoid repeated global lookups.
_WHITE = chess.WHITE
_BLACK = chess.BLACK
_PIECE_TYPES = PIECE_TYPES


def evaluate(board):
    """
    Heuristic evaluation of the board position from White's perspective.

    Algorithm:
      1. Check for terminal states (checkmate / draw) and return
         extreme or zero scores immediately — no need to sum pieces.
      2. Loop over every occupied square.
      3. For each piece, add its material value plus its positional bonus
         (looked up in the piece-square table for its colour).
      4. Add the combined value to the score for White, subtract for Black.

    The result is used by the Minimax search when it hits the depth limit:
    a higher score means a better position for White (the maximising player).

    Args:
        board: chess.Board — the current game state

    Returns:
        Integer score in centipawns.
          > 0  White is ahead
          < 0  Black is ahead
            0  Equal position (or forced draw)
    """
    if not any(board.legal_moves):
        return (-100000 if board.turn else 100000) if board.is_check() else 0
    if board.is_insufficient_material():
        return 0

    score = 0
    pieces = board.pieces
    wf = _WHITE_FLAT
    bf = _BLACK_FLAT

    for piece_type in _PIECE_TYPES:
        offset = piece_type << 6            # piece_type * 64
        for sq in pieces(piece_type, _WHITE):
            score += wf[offset + sq]
        for sq in pieces(piece_type, _BLACK):
            score -= bf[offset + sq]

    return score

if __name__ == "__main__":
    board = chess.Board()
    print(evaluate(board))          # Should print 0 (perfectly symmetric start)

    board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    print(evaluate(board2))         # White is missing a rook → expect negative score
