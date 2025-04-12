import pygame
import asyncio
import json
import sys
import numpy as np
import logging
# Adjust import if necessary
from tic_tac_toe_logic import (
    BOARD_SIZE, PLAYER_X, PLAYER_O, EMPTY # We don't need game logic functions here, only constants/board size
)

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Change to server's IP if not running locally
SERVER_PORT = 8888
logging.basicConfig(level=logging.INFO)

# Pygame constants (same as main_local.py)
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

# --- Client State (Global, updated by network task) ---
# Network components will be assigned in connect_to_server
reader = None
writer = None
# Game state components will be updated by listen_to_server
player_id = None        # Will be PLAYER_X or PLAYER_O, assigned by server
current_turn = None     # Whose turn is it? Updated by server
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int) # Local copy of the board state
game_over = False
status_message = "Connecting..." # Message displayed at the bottom


# --- Pygame Setup ---
# We initialize pygame later in the main async function after connection attempt
screen = None
font = None

# --- Pygame Drawing Functions (Copied/adapted from main_local.py) ---
def draw_lines():
    if not screen: return
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, WIDTH), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, WIDTH), LINE_WIDTH)

def draw_figures():
    if not screen: return
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            center_x = int(col * SQUARE_SIZE + SQUARE_SIZE // 2)
            center_y = int(row * SQUARE_SIZE + SQUARE_SIZE // 2)

            if board[row, col] == PLAYER_O:
                pygame.draw.circle(screen, CIRCLE_COLOR, (center_x, center_y), CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row, col] == PLAYER_X:
                offset = SQUARE_SIZE // 4
                pygame.draw.line(screen, CROSS_COLOR, (center_x - offset, center_y - offset), (center_x + offset, center_y + offset), CROSS_WIDTH)
                pygame.draw.line(screen, CROSS_COLOR, (center_x + offset, center_y - offset), (center_x - offset, center_y + offset), CROSS_WIDTH)

def display_status(message):
    if not screen or not font: return
    # Clear previous status message area
    pygame.draw.rect(screen, BG_COLOR, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
    text = font.render(message, True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, WIDTH + (HEIGHT - WIDTH) // 2))
    screen.blit(text, text_rect)

# --- Networking Functions ---
async def send_message(writer_local, data):
    """Encodes dict to JSON and sends to the server."""
    if not writer_local:
        logging.error("Writer is None, cannot send message.")
        return
    try:
        message = json.dumps(data).encode('utf-8')
        writer_local.write(message + b'\n') # Use newline as delimiter
        await writer_local.drain()
        logging.debug(f"Sent: {data}")
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        # Handle potential disconnection or errors more robustly if needed
        global status_message, game_over
        status_message = "Connection error"
        game_over = True # Stop game on send error

async def listen_to_server(reader_local):
    """Listens for messages from the server and updates client state."""
    global board, current_turn, player_id, game_over, status_message
    try:
        while True:
            data = await reader_local.readuntil(b'\n') # Read until newline
            if not data:
                logging.warning("Server closed connection (no data).")
                status_message = "Server disconnected."
                game_over = True
                break

            message_str = data.decode('utf-8').strip()
            logging.debug(f"Received: {message_str}")

            try:
                message = json.loads(message_str)
                msg_type = message.get("type")

                if msg_type == "GAME_START":
                    player_id = message.get("player_id")
                    status_message = message.get("message", f"You are Player {player_id}")
                    logging.info(f"Game started. Assigned Player ID: {player_id}")

                elif msg_type == "STATE_UPDATE":
                    board = np.array(message.get("board", [])) # Update board from server data
                    current_turn = message.get("current_turn")
                    if not game_over: # Don't overwrite win/loss message
                         if current_turn == player_id:
                             status_message = "Your turn"
                         else:
                             status_message = f"Player {'X' if current_turn == PLAYER_X else 'O'}'s turn"

                elif msg_type == "GAME_OVER":
                    board = np.array(message.get("board", board)) # Show final board
                    winner = message.get("winner")
                    game_over = True
                    if winner == player_id:
                        status_message = "You win!"
                    elif winner == 0: # Draw
                        status_message = "It's a Draw!"
                    else: # Opponent won
                        status_message = "You lose!"
                    logging.info(f"Game Over. Winner: {winner}")

                elif msg_type == "INVALID_MOVE":
                    status_message = message.get("message", "Invalid Move")
                    # Optionally revert status message after a delay?

                elif msg_type == "WAITING":
                    status_message = message.get("message", "Waiting for opponent...")

                elif msg_type == "OPPONENT_DISCONNECTED":
                    status_message = "Opponent disconnected. You win!"
                    game_over = True

                elif msg_type == "ERROR":
                     status_message = f"Server Error: {message.get('message', 'Unknown error')}"
                     game_over = True

            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON received: {message_str}")
            except Exception as e:
                logging.error(f"Error processing server message: {e}", exc_info=True)
                status_message = "Processing Error"
                game_over = True # Stop game on processing error
                break # Exit loop on error

    except asyncio.IncompleteReadError:
        logging.warning("Server disconnected unexpectedly (IncompleteReadError).")
        status_message = "Server disconnected."
        game_over = True
    except ConnectionResetError:
         logging.warning("Connection to server reset.")
         status_message = "Connection Reset."
         game_over = True
    except Exception as e:
        logging.error(f"Error in listen_to_server: {e}", exc_info=True)
        status_message = "Network Error"
        game_over = True
    finally:
        logging.info("Listener task finished.")
        game_over = True # Ensure game stops if listener exits


async def game_loop(writer_local):
    """Handles Pygame display and user input."""
    global status_message # Allow modification
    running = True
    while running:
        if not screen or not font: # Check if pygame is initialized
            await asyncio.sleep(0.1)
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break # Exit inner loop

            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if current_turn == player_id: # Check if it's our turn
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]
                    clicked_row = int(mouseY // SQUARE_SIZE)
                    clicked_col = int(mouseX // SQUARE_SIZE)

                    if clicked_row < BOARD_ROWS: # Ensure click is on the board
                        # Basic check if cell seems empty locally (server will validate definitively)
                        if board[clicked_row, clicked_col] == EMPTY:
                            cell_index = clicked_row * BOARD_COLS + clicked_col
                            logging.info(f"Player {player_id} clicked cell: {(clicked_row, clicked_col)}, index: {cell_index}")
                            await send_message(writer_local, {"type": "MAKE_MOVE", "cell": cell_index})
                        else:
                            logging.warning("Clicked on an occupied cell.")
                            # status_message = "Cell occupied!" # Can provide instant feedback, but server message is more reliable
                elif player_id is not None: # It's not our turn, but we are in game
                     logging.info("Clicked, but not your turn.")
                     # status_message = "Not your turn!" # Instant feedback


        # Drawing
        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures() # Draws based on the global board state updated by listen_to_server
        display_status(status_message) # Display the current status

        pygame.display.update()

        # Check if listener task indicated game over outside of events
        if game_over and running:
            # Keep displaying the final state for a bit? Or exit?
            # For now, we let the loop run until QUIT event
            pass

        # Yield control to asyncio event loop
        await asyncio.sleep(0.01) # Adjust sleep time as needed for responsiveness

    # Cleanup after loop exit
    if writer_local:
        writer_local.close()
        await writer_local.wait_closed()
    pygame.quit()
    logging.info("Pygame quit.")


async def main():
    """Main function to connect and start client tasks."""
    global reader, writer, screen, font, status_message
    try:
        reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
        logging.info(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")
        status_message = "Connected, waiting for game..."

        # Initialize Pygame *after* successful connection attempt
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Networked Tic Tac Toe Client")
        font = pygame.font.Font(None, 36)

        # Start listener and game loop concurrently
        listener_task = asyncio.create_task(listen_to_server(reader))
        gameloop_task = asyncio.create_task(game_loop(writer))

        # Wait for tasks to complete (game_loop will finish on pygame quit)
        await asyncio.gather(listener_task, gameloop_task)

    except ConnectionRefusedError:
        logging.error(f"Connection refused. Is the server running at {SERVER_HOST}:{SERVER_PORT}?")
        print(f"Error: Could not connect to the server at {SERVER_HOST}:{SERVER_PORT}. Please ensure the server is running.")
    except Exception as e:
        logging.error(f"An error occurred in main: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    finally:
        if writer:
            writer.close()
            try:
                 await writer.wait_closed()
            except Exception as e:
                logging.debug(f"Exception during writer close: {e}") # Can happen if already closed
        if pygame.get_init():
            pygame.quit()
        logging.info("Client shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())