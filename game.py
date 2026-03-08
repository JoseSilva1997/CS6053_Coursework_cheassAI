"""
game.py
-------
Chess board GUI and game loop.

Responsibilities:
  - Render the 8×8 board with light/dark squares.
  - Load and display piece images (from the icons/ folder).
  - Handle mouse clicks so the human player can select and move pieces.
  - Call the AI (get_best_move from search.py) on the opponent's turn.
  - Show a status bar at the bottom (whose turn, check, game-over message).
  - Display a game-over overlay when the game ends.
"""
import pygame
import sys
import chess
from search import get_best_move

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SQ_SIZE = 80                    # pixels per square
BOARD_W = SQ_SIZE * 8           # 640 — board width
BOARD_H = SQ_SIZE * 8           # 640 — board height
WIDTH   = BOARD_W               # window width equals board width
HEIGHT  = BOARD_H + 60          # extra 60 px below the board for status bar

FPS = 60                        # frame rate cap

# ---------------------------------------------------------------------------
# Colours (RGB)
# ---------------------------------------------------------------------------
LIGHT_SQ  = (240, 217, 181)     # cream — light squares
DARK_SQ   = (181, 136,  99)     # brown — dark squares
HIGHLIGHT = (186, 202,  68)     # yellow-green — selected square
DARK_BG   = (30,  30,  30)      # near-black — status bar background
TEXT_COL  = (255, 255, 255)     # white — status text
ACCENT    = (200, 160,  60)     # gold — game-over title text


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def get_screen_pos(square, flipped):
    """
    Convert a chess.Square to pixel coordinates (top-left of that square).

    When the board is NOT flipped (playing as White) rank 8 is at the top
    of the screen and rank 1 is at the bottom — the standard orientation.
    When flipped (playing as Black) the board is rotated 180°.

    Args:
        square:  chess.Square (0 = a1 … 63 = h8)
        flipped: bool — True when the player chose Black

    Returns:
        (x, y) pixel position of the square's top-left corner.
    """
    file = chess.square_file(square)   # 0 = a … 7 = h
    rank = chess.square_rank(square)   # 0 = rank 1 … 7 = rank 8
    if flipped:
        col = 7 - file
        row = rank
    else:
        col = file
        row = 7 - rank
    return col * SQ_SIZE, row * SQ_SIZE


def square_from_mouse(x, y, flipped):
    """
    Convert a mouse pixel position to a chess.Square (or None if off-board).

    This is the inverse of get_screen_pos — it maps screen coordinates back
    to a board square so the player's clicks can be interpreted.

    Args:
        x, y:    pixel position of the mouse click
        flipped: bool — True when the player chose Black

    Returns:
        chess.Square, or None if the click was outside the board.
    """
    col = x // SQ_SIZE
    row = y // SQ_SIZE
    if flipped:
        file = 7 - col
        rank = row
    else:
        file = col
        rank = 7 - row
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    return None


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

def draw_board(screen, selected_square, flipped):
    """
    Draw the 64 squares of the chessboard.

    Light and dark squares are determined by (rank + file) % 2.
    The currently selected square is drawn in HIGHLIGHT colour so the
    player can see which piece they have picked up.

    Args:
        screen:          pygame.Surface — the main display surface
        selected_square: chess.Square or None — the square to highlight
        flipped:         bool — board orientation
    """
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            # Work out the screen column/row from the board file/rank
            if flipped:
                col = 7 - file
                row = rank
            else:
                col = file
                row = 7 - rank
            x = col * SQ_SIZE
            y = row * SQ_SIZE

            # Colour: highlighted > light/dark pattern
            if square == selected_square:
                colour = HIGHLIGHT
            elif (rank + file) % 2 == 0:
                colour = LIGHT_SQ
            else:
                colour = DARK_SQ
            pygame.draw.rect(screen, colour, (x, y, SQ_SIZE, SQ_SIZE))


PIECE_OFFSET = 6   # pixels of padding so the piece image is centred in the square


def draw_pieces(screen, board, flipped, piece_images):
    """
    Blit each piece image onto its correct board square.

    piece_images is a dict keyed by (piece_type, color) → pygame.Surface,
    built once in game_loop() from the PNG files in icons/.

    Args:
        screen:       pygame.Surface
        board:        chess.Board — current position
        flipped:      bool — board orientation
        piece_images: dict mapping (piece_type, color) to a Surface
    """
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        x, y = get_screen_pos(square, flipped)
        img = piece_images[(piece.piece_type, piece.color)]
        # PIECE_OFFSET centres the (SQ_SIZE - 12) image inside the SQ_SIZE square
        screen.blit(img, (x + PIECE_OFFSET, y + PIECE_OFFSET))


def draw_status(screen, board, font):
    """
    Draw the status bar below the board showing game state text.

    Possible messages:
      - "Checkmate — <winner> wins!"
      - "Stalemate — Draw!"
      - "Insufficient material — Draw!"
      - "Check!"
      - "White to move" / "Black to move"

    Args:
        screen: pygame.Surface
        board:  chess.Board
        font:   pygame.font.Font for rendering the text
    """
    pygame.draw.rect(screen, DARK_BG, (0, BOARD_H, WIDTH, 60))

    if board.is_checkmate():
        winner = "Black wins!" if board.turn == chess.WHITE else "White wins!"
        msg = f"Checkmate — {winner}"
    elif board.is_stalemate():
        msg = "Stalemate — Draw!"
    elif board.is_insufficient_material():
        msg = "Insufficient material — Draw!"
    elif board.is_check():
        msg = "Check!"
    else:
        msg = "White to move" if board.turn == chess.WHITE else "Black to move"

    text_surf = font.render(msg, True, TEXT_COL)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, BOARD_H + 30))
    screen.blit(text_surf, text_rect)


def game_over_screen(screen, clock, board):
    """
    Show a semi-transparent overlay with the game result.

    Blocks until the player presses any key or clicks the mouse, then
    returns so the main loop can go back to the start menu.

    Args:
        screen: pygame.Surface
        clock:  pygame.time.Clock
        board:  chess.Board (used to build the result message)
    """
    font_title = pygame.font.SysFont("Arial", 38, bold=True)
    font_sub   = pygame.font.SysFont("Arial", 22)

    if board.is_checkmate():
        winner = "Black wins!" if board.turn == chess.WHITE else "White wins!"
        msg = f"Checkmate — {winner}"
    elif board.is_stalemate():
        msg = "Stalemate — Draw!"
    else:
        msg = "Draw!"

    # Dark semi-transparent overlay drawn on top of the final board position
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    title_surf = font_title.render(msg, True, ACCENT)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
    screen.blit(title_surf, title_rect)

    sub_surf = font_sub.render("Press any key to return to menu", True, TEXT_COL)
    sub_rect = sub_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
    screen.blit(sub_surf, sub_rect)

    pygame.display.flip()

    # Wait for any key press or mouse click before returning
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                return


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def game_loop(screen, clock, player_colour, depth):
    """
    Run one full chess game between the human and the AI.

    Flow each frame:
      1. If it is the AI's turn, call get_best_move() and push the result.
      2. Process mouse events so the human can select and move pieces.
      3. Redraw the board, pieces, and status bar.
      4. If the game is over, show the game-over overlay and return.

    The board is flipped (Black at the bottom) when the player chose Black
    so the human always sees their own pieces at the bottom.

    Args:
        screen:        pygame.Surface — the display (may be resized here)
        clock:         pygame.time.Clock
        player_colour: chess.WHITE or chess.BLACK
        depth:         int — AI search depth (1 = Easy, 3 = Medium, 5 = Hard)
    """
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    board           = chess.Board()
    selected_square = None                                         # piece the human has clicked
    flipped         = (player_colour == chess.BLACK)              # flip board for Black
    ai_colour       = chess.BLACK if player_colour == chess.WHITE else chess.WHITE

    font_status = pygame.font.SysFont("Arial", 22)

    # Map piece type constants to the base filename in icons/
    piece_files = {
        chess.PAWN:   'pawn',
        chess.KNIGHT: 'knight',
        chess.BISHOP: 'bishop',
        chess.ROOK:   'rook',
        chess.QUEEN:  'queen',
        chess.KING:   'king',
    }
    PIECE_SIZE = SQ_SIZE - 12   # slightly smaller than the square (12 px padding total)

    # Load and colorise piece images once before the game loop starts.
    # Both white and black pieces use the same source PNG.
    # White pieces: BLEND_RGBA_ADD adds (255,255,255,0) → pushes RGB channels to 255.
    # Black pieces: BLEND_RGBA_MULT multiplies by (0,0,0,255) → pulls RGB to 0.
    PIECE_IMAGES = {}
    for piece_type, filename in piece_files.items():
        base = pygame.image.load(f"icons/{filename}.png").convert_alpha()
        base = pygame.transform.scale(base, (PIECE_SIZE, PIECE_SIZE))

        white_img = base.copy()
        white_img.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_ADD)
        PIECE_IMAGES[(piece_type, chess.WHITE)] = white_img

        black_img = base.copy()
        black_img.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        PIECE_IMAGES[(piece_type, chess.BLACK)] = black_img

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    while True:

        # --- AI move ---------------------------------------------------
        # If it is the AI's turn and the game is not over, let the AI think.
        # A short delay (600 ms) makes the AI feel more natural and gives
        # the player a moment to see the board before the AI responds.
        if board.turn == ai_colour and not board.is_game_over():
            pygame.time.wait(600) # Added just for the move not to be instant — gives the player a moment to see the board before the AI responds
            move, nodes, t = get_best_move(board, depth)
            if move:
                board.push(move)
            selected_square = None

        # --- Event handling --------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Left mouse button: select or move a piece
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if board.turn == player_colour and not board.is_game_over():
                    x, y = event.pos
                    if y < BOARD_H:   # click is inside the board area
                        clicked = square_from_mouse(x, y, flipped)
                        if clicked is None:
                            continue

                        if selected_square is None:
                            # First click: select a piece that belongs to the player
                            piece = board.piece_at(clicked)
                            if piece and piece.color == player_colour:
                                selected_square = clicked
                        else:
                            # Second click: attempt to move the selected piece
                            move = chess.Move(selected_square, clicked)

                            # Automatically promote pawns to queen
                            piece = board.piece_at(selected_square)
                            if piece and piece.piece_type == chess.PAWN:
                                rank = chess.square_rank(clicked)
                                if rank == 7 or rank == 0:
                                    move = chess.Move(selected_square, clicked,
                                                      promotion=chess.QUEEN)

                            if move in board.legal_moves:
                                board.push(move)
                                selected_square = None
                            else:
                                # Illegal destination — allow re-selecting another piece
                                piece = board.piece_at(clicked)
                                if piece and piece.color == player_colour:
                                    selected_square = clicked
                                else:
                                    selected_square = None

        # --- Render ----------------------------------------------------
        draw_board(screen, selected_square, flipped)
        draw_pieces(screen, board, flipped, PIECE_IMAGES)
        draw_status(screen, board, font_status)
        pygame.display.flip()
        clock.tick(FPS)

        # --- Game over check -------------------------------------------
        if board.is_game_over():
            # Draw the final position one more time before the overlay
            draw_board(screen, None, flipped)
            draw_pieces(screen, board, flipped, PIECE_IMAGES)
            draw_status(screen, board, font_status)
            pygame.display.flip()
            pygame.time.wait(500)
            game_over_screen(screen, clock, board)
            return   # back to the start menu in gui.py