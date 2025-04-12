import pygame
import sys
import numpy as np
# Adjust import if necessary
from tic_tac_toe_logic import (
    create_initial_state, get_valid_actions, apply_action,
    check_win_condition, check_draw_condition, get_next_player,
    BOARD_SIZE, EMPTY, PLAYER_X, PLAYER_O
)

# Pygame constants
WIDTH, HEIGHT = 300, 350 # Increased height for status message
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = BOARD_SIZE, BOARD_SIZE
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25

# Colors
BG_COLOR = (200, 200, 200)
LINE_COLOR = (50, 50, 50)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)
TEXT_COLOR = (10, 10, 10)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Local Tic Tac Toe")
font = pygame.font.Font(None, 36)

def draw_lines():
    """Draws the grid lines."""
    # Horizontal
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, WIDTH), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, WIDTH), LINE_WIDTH)

def draw_figures(board):
    """Draws X's and O's based on the board state."""
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center_x = int(col * SQUARE_SIZE + SQUARE_SIZE // 2)
            center_y = int(row * SQUARE_SIZE + SQUARE_SIZE // 2)

            if board[row, col] == PLAYER_O: # Draw O
                pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row, col] == PLAYER_X: # Draw X
                offset = SQUARE_SIZE // 4
                pygame.draw.line(screen, CROSS_COLOR, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (center_x + offset, center_y - offset), (center_x - offset, center_y + offset), CROSS_WIDTH)

def display_status(message):
    """Displays game status text at the bottom."""
    pygame.draw.rect(screen, BG_COLOR, (0, WIDTH, WIDTH, HEIGHT - WIDTH)) # Clear bottom area
    text = font.render(message, True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, WIDTH + (HEIGHT - WIDTH) // 2))
    screen.blit(text, text_rect)

def main():
    board = create_initial_state()
    current_player = PLAYER_X
    game_over = False
    status_message = f"Player {'X' if current_player == PLAYER_X else 'O'}'s turn"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                mouseX = event.pos[0] # x
                mouseY = event.pos[1] # y

                clicked_row = int(mouseY // SQUARE_SIZE)
                clicked_col = int(mouseX // SQUARE_SIZE)

                # Ensure click is within the board grid area
                if clicked_row < BOARD_ROWS:
                    action = (clicked_row, clicked_col)

                    # Check if the move is valid (using logic function)
                    if action in get_valid_actions(board):
                        try:
                            board = apply_action(board, action, current_player)

                            if check_win_condition(board, current_player):
                                game_over = True
                                status_message = f"Player {'X' if current_player == PLAYER_X else 'O'} wins!"
                            elif check_draw_condition(board):
                                game_over = True
                                status_message = "It's a Draw!"
                            else:
                                current_player = get_next_player(current_player)
                                status_message = f"Player {'X' if current_player == PLAYER_X else 'O'}'s turn"

                        except ValueError as e:
                            print(f"Error applying action: {e}") # Should not happen if get_valid_actions is correct
                    else:
                        print("Invalid move attempted.")


        # Drawing
        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures(board)
        display_status(status_message)

        pygame.display.update()

if __name__ == "__main__":
    main()