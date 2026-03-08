"""
gui.py
------
Menu screens for the Chess AI application.

Two screens are shown before each game:
  1. start_screen  — player chooses to play as White or Black.
  2. difficulty_screen — player selects Easy / Medium / Hard,
                         which maps to Minimax search depths 1 / 3 / 5.

After each game the main() loop returns here automatically, so the player
can start a new game without restarting the program.
"""
import pygame
import sys
import chess
from game import game_loop

# ---------------------------------------------------------------------------
# Window settings for the menu screens
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 700, 500
FPS = 60

# Colours (RGB)
DARK_BG    = (30,  30,  30)    # background
BUTTON_COL = (70,  70,  70)    # normal button
HOVER_COL  = (110, 110, 110)   # button when the mouse is over it
TEXT_COL   = (255, 255, 255)   # button label text
ACCENT     = (200, 160,  60)   # gold — title text and button border


def draw_button(screen, button, font, mouse_pos):
    """
    Draw a single rounded rectangle button with a hover effect.

    If the mouse is over the button it is drawn in HOVER_COL; otherwise
    BUTTON_COL.  An ACCENT-coloured border is always drawn.

    Args:
        screen:    pygame.Surface
        button:    dict with keys 'rect' (pygame.Rect) and 'text' (str)
        font:      pygame.font.Font for the label
        mouse_pos: (x, y) tuple from pygame.mouse.get_pos()
    """
    hovered = button["rect"].collidepoint(mouse_pos)
    colour = HOVER_COL if hovered else BUTTON_COL
    pygame.draw.rect(screen, colour, button["rect"], border_radius=10)
    pygame.draw.rect(screen, ACCENT, button["rect"], width=2, border_radius=10)
    text_surf = font.render(button["text"], True, TEXT_COL)
    text_rect = text_surf.get_rect(center=button["rect"].center)
    screen.blit(text_surf, text_rect)


def start_screen(screen, clock):
    """
    Display the title / colour-selection screen.

    Shows the game title and two buttons:
      - "Play as White" → the player controls the White pieces; AI plays Black.
      - "Play as Black" → the player controls the Black pieces; AI plays White.

    Blocks until the player clicks a button.

    Returns:
        str — "white" or "black"
    """
    font_title    = pygame.font.SysFont("Arial", 52, bold=True)
    font_subtitle = pygame.font.SysFont("Arial", 20)
    font_button   = pygame.font.SysFont("Arial", 26)

    btn_w, btn_h = 320, 55
    centre_x = WIDTH // 2 - btn_w // 2

    buttons = [
        {"rect": pygame.Rect(centre_x, 265, btn_w, btn_h), "text": "Play as White", "action": "white"},
        {"rect": pygame.Rect(centre_x, 345, btn_w, btn_h), "text": "Play as Black", "action": "black"},
    ]

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button["rect"].collidepoint(mouse_pos):
                        return button["action"]   # "white" or "black"

        screen.fill(DARK_BG)

        # Title
        title_surf = font_title.render("Chess AI", True, ACCENT)
        title_rect = title_surf.get_rect(center=(WIDTH // 2, 90))
        screen.blit(title_surf, title_rect)

        # Subtitle describing the algorithm
        sub_surf = font_subtitle.render("Minimax with Alpha-Beta Pruning", True, TEXT_COL)
        sub_rect = sub_surf.get_rect(center=(WIDTH // 2, 155))
        screen.blit(sub_surf, sub_rect)

        for button in buttons:
            draw_button(screen, button, font_button, mouse_pos)

        pygame.display.flip()
        clock.tick(FPS)


def difficulty_screen(screen, clock):
    """
    Display the difficulty-selection screen.

    The difficulty maps directly to the Minimax search depth:
      - Easy   → depth 1  (AI looks 1 move ahead — fast but weak)
      - Medium → depth 3  (AI looks 3 moves ahead — balanced)
      - Hard   → depth 5  (AI looks 5 moves ahead — strong, slower)

    A greater depth means the AI searches a larger game tree and finds
    better moves, but takes exponentially more time.

    Blocks until the player clicks a button.

    Returns:
        int — search depth (1, 3, or 5)
    """
    font_title  = pygame.font.SysFont("Arial", 42, bold=True)
    font_button = pygame.font.SysFont("Arial", 26)

    btn_w, btn_h = 320, 55
    centre_x = WIDTH // 2 - btn_w // 2

    buttons = [
        {"rect": pygame.Rect(centre_x, 185, btn_w, btn_h), "text": "Easy   (depth 1)", "action": 1},
        {"rect": pygame.Rect(centre_x, 265, btn_w, btn_h), "text": "Medium (depth 3)", "action": 3},
        {"rect": pygame.Rect(centre_x, 345, btn_w, btn_h), "text": "Hard   (depth 5)", "action": 5},
    ]

    while True:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button["rect"].collidepoint(mouse_pos):
                        return button["action"]   # 1, 3, or 5

        screen.fill(DARK_BG)

        title_surf = font_title.render("Select Difficulty", True, ACCENT)
        title_rect = title_surf.get_rect(center=(WIDTH // 2, 100))
        screen.blit(title_surf, title_rect)

        for button in buttons:
            draw_button(screen, button, font_button, mouse_pos)

        pygame.display.flip()
        clock.tick(FPS)


def main():
    """
    Entry point for the application.

    Initialises pygame then loops forever:
      1. Show the start screen  → get the player's colour choice.
      2. Show the difficulty screen → get the search depth.
      3. Run a game.
      4. When the game ends, loop back to step 1 automatically.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess AI")
    clock = pygame.time.Clock()

    while True:
        colour = start_screen(screen, clock)            # "white" or "black"
        depth  = difficulty_screen(screen, clock)       # 1, 3, or 5
        player_colour = chess.WHITE if colour == "white" else chess.BLACK
        game_loop(screen, clock, player_colour, depth)  # blocks until game over


if __name__ == "__main__":
    main()