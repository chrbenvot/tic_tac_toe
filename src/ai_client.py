import pygame
import asyncio
import json
import sys
import numpy as np
import logging
import os
from stable_baselines3 import DQN # Make sure to import the algorithm used for training

# Adjust import if necessary
from tic_tac_toe_logic import (
    BOARD_SIZE, PLAYER_X, PLAYER_O, EMPTY
)

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Change to server's IP if not running locally
SERVER_PORT = 8888
import os # Make sure os is imported at the top

# --- Calculate paths relative to this script's location ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory containing ai_client.py (i.e., src/)
MODEL_DIR = os.path.join(script_dir, "models")   # Models directory *inside* src/
MODEL_NAME = "tictactoe_dqn_agent.zip" # Ensure this matches your saved model file
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME) # MODEL_PATH now points inside src/models/
AI_THINK_DELAY = 0.5 # Optional delay in seconds to make AI moves less instantaneous

logging.basicConfig(level=logging.INFO)

# Pygame constants (same as client.py)
WIDTH, HEIGHT = 300, 350
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = BOARD_SIZE, BOARD_SIZE
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
BG_COLOR = (200, 200, 200)
LINE_COLOR = (50, 50, 50)
CIRCLE_COLOR = (239, 231, 200) # O color
CROSS_COLOR = (66, 66, 66)   # X color
TEXT_COLOR = (10, 10, 10)

# --- Client State ---
reader = None
writer = None
player_id = None
current_turn = None
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
game_over = False
status_message = "Connecting AI..."
ai_model = None

# --- Pygame Setup ---
screen = None
font = None

# --- Pygame Drawing Functions (Copied from client.py) ---
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
    pygame.draw.rect(screen, BG_COLOR, (0, WIDTH, WIDTH, HEIGHT - WIDTH))
    text = font.render(message, True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, WIDTH + (HEIGHT - WIDTH) // 2))
    screen.blit(text, text_rect)

# --- Networking Functions (Copied/adapted from client.py) ---
async def send_message(writer_local, data):
    if not writer_local: return
    try:
        message = json.dumps(data).encode('utf-8')
        writer_local.write(message + b'\n')
        await writer_local.drain()
        logging.debug(f"AI Sent: {data}")
    except Exception as e:
        logging.error(f"AI Error sending message: {e}")
        global status_message, game_over
        status_message = "AI Connection error"
        game_over = True

async def listen_to_server(reader_local):
    """Listens for messages from the server and updates client state."""
    global board, current_turn, player_id, game_over, status_message
    try:
        while True:
            data = await reader_local.readuntil(b'\n')
            if not data:
                logging.warning("AI: Server closed connection.")
                status_message = "AI: Server disconnected."
                game_over = True
                break

            message_str = data.decode('utf-8').strip()
            logging.debug(f"AI Received: {message_str}")

            try:
                message = json.loads(message_str)
                msg_type = message.get("type")

                # Update state based on message type (same logic as human client)
                if msg_type == "GAME_START":
                    player_id = message.get("player_id")
                    status_message = message.get("message", f"AI is Player {player_id}")
                    logging.info(f"AI Game started. Assigned Player ID: {player_id}")
                elif msg_type == "STATE_UPDATE":
                    board = np.array(message.get("board", []))
                    current_turn = message.get("current_turn")
                    # Status update logic is handled in game_loop now based on turn
                elif msg_type == "GAME_OVER":
                    board = np.array(message.get("board", board))
                    winner = message.get("winner")
                    game_over = True
                    if winner == player_id: status_message = "AI wins!"
                    elif winner == 0: status_message = "AI: Draw!"
                    else: status_message = "AI loses!"
                    logging.info(f"AI Game Over. Winner: {winner}")
                elif msg_type == "INVALID_MOVE":
                    status_message = f"AI received: {message.get('message', 'Invalid Move')}"
                    # AI might receive this if server rejects its move (should be rare if model is good)
                    logging.warning(f"AI received invalid move: {message.get('message')}")
                    # If AI caused invalid move, maybe force random move next? For now, just log.
                elif msg_type == "WAITING":
                    status_message = message.get("message", "AI waiting for opponent...")
                elif msg_type == "OPPONENT_DISCONNECTED":
                    status_message = "AI: Opponent disconnected. AI wins!"
                    game_over = True
                elif msg_type == "ERROR":
                     status_message = f"AI Server Error: {message.get('message', 'Unknown error')}"
                     game_over = True

            except json.JSONDecodeError:
                logging.warning(f"AI: Invalid JSON received: {message_str}")
            except Exception as e:
                logging.error(f"AI: Error processing server message: {e}", exc_info=True)
                status_message = "AI Processing Error"
                game_over = True
                break

    except asyncio.IncompleteReadError:
        logging.warning("AI: Server disconnected unexpectedly.")
        status_message = "AI: Server disconnected."
    except ConnectionResetError:
         logging.warning("AI: Connection to server reset.")
         status_message = "AI: Connection Reset."
    except Exception as e:
        logging.error(f"AI: Error in listen_to_server: {e}", exc_info=True)
        status_message = "AI Network Error"
    finally:
        logging.info("AI Listener task finished.")
        game_over = True # Ensure game loop knows listener exited


def get_ai_move(current_board, ai_player_id):
    """Uses the loaded model to predict the best move."""
    if ai_model is None:
        logging.error("AI model is not loaded!")
        return None # Or maybe return a random valid move?

    # --- Observation Preparation ---
    # 1. Get board in the model's expected format (flattened numpy array)
    observation = current_board.flatten().astype(np.int8)

    # 2. Canonicalize observation (CRITICAL!)
    # The environment assumes the agent is PLAYER_X (1).
    # If the server assigned the AI to be PLAYER_O (2), we need to swap
    # the numbers 1 and 2 in the observation so the AI sees *itself* as 1
    # and the *opponent* as 2, just like during training.
    canonical_obs = observation.copy()
    if ai_player_id == PLAYER_O:
        # Swap 1s and 2s
        player_x_mask = (canonical_obs == PLAYER_X)
        player_o_mask = (canonical_obs == PLAYER_O)
        canonical_obs[player_x_mask] = PLAYER_O # Opponent's pieces become 2
        canonical_obs[player_o_mask] = PLAYER_X # AI's pieces become 1

    # --- Predict Action ---
    try:
        action_index, _ = ai_model.predict(canonical_obs, deterministic=True)
        # `deterministic=True` means the AI picks the best-known action, not exploring randomly.
        action_index = int(action_index) # Ensure it's a standard Python int

        # --- Validate Action (Optional but Recommended) ---
        # Check if the predicted action is actually valid on the *current* board
        row, col = divmod(action_index, BOARD_COLS)
        if current_board[row, col] != EMPTY:
            logging.warning(f"AI model predicted invalid move ({action_index} on non-empty cell). Falling back to random.")
            # Fallback: Choose a random valid move
            valid_actions = np.where(current_board.flatten() == EMPTY)[0]
            if len(valid_actions) > 0:
                 action_index = int(np.random.choice(valid_actions))
            else:
                 return None # No valid moves left (should mean draw/win already happened)

        logging.info(f"AI (Player {ai_player_id}) predicts move index: {action_index}")
        return action_index

    except Exception as e:
        logging.error(f"Error during AI prediction: {e}", exc_info=True)
        return None


async def game_loop(writer_local):
    """Handles Pygame display and triggers AI moves."""
    global status_message # Allow modification
    running = True
    last_ai_move_time = asyncio.get_event_loop().time()

    while running:
        if not screen or not font:
            await asyncio.sleep(0.1)
            continue

        current_time = asyncio.get_event_loop().time()

        # Check for Pygame quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break # Exit inner loop

        # --- AI's Turn Logic ---
        if not game_over and current_turn == player_id:
            if current_time - last_ai_move_time >= AI_THINK_DELAY: # Add delay
                status_message = f"AI (Player {player_id}) is thinking..."
                display_status(status_message) # Update display immediately
                pygame.display.update() # Show the "thinking" message

                action_index = get_ai_move(board, player_id)

                if action_index is not None:
                    await send_message(writer_local, {"type": "MAKE_MOVE", "cell": action_index})
                    # Server response will update current_turn, so AI won't immediately move again
                else:
                    logging.error("AI could not determine a valid move.")
                    # What should happen here? Maybe skip turn? For now, it will just log.
                    status_message = "AI Error: No move"

                last_ai_move_time = current_time # Reset timer after attempting move
        elif not game_over and current_turn is not None:
             # Update status if it's the other player's turn
             other_player_symbol = 'X' if current_turn == PLAYER_X else 'O'
             status_message = f"Player {other_player_symbol}'s turn"
        # If game is over, status_message is set by the listener

        # Drawing (happens every loop iteration)
        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures() # Draws board based on state updated by listener
        display_status(status_message) # Display current status

        pygame.display.update()

        # Check if listener task indicated game over
        if game_over and running:
            pass # Keep displaying final state

        await asyncio.sleep(0.05) # Yield control, slightly longer sleep might be ok for AI client

    # Cleanup
    if writer_local:
        writer_local.close()
        await writer_local.wait_closed()
    pygame.quit()
    logging.info("AI Pygame quit.")


async def main():
    """Main function to load model, connect, and start client tasks."""
    global reader, writer, screen, font, status_message, ai_model

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at: {MODEL_PATH}")
        print(f"Error: Trained model '{MODEL_NAME}' not found in '{MODEL_DIR}'.")
        print("Please ensure you have run the training script (`src/train_agent.py`) first.")
        return
    try:
        # Load the model using the same algorithm class it was trained with (DQN)
        ai_model = DQN.load(MODEL_PATH)
        logging.info(f"AI model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error loading AI model: {e}", exc_info=True)
        print(f"Error: Could not load the AI model from {MODEL_PATH}. Ensure it's a valid Stable Baselines 3 DQN model.")
        return

    # --- Connect to Server ---
    try:
        reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
        logging.info(f"AI Client connected to server at {SERVER_HOST}:{SERVER_PORT}")
        status_message = "AI Connected, waiting..."

        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Networked Tic Tac Toe - AI Client")
        font = pygame.font.Font(None, 36)

        # Start listener and game loop
        listener_task = asyncio.create_task(listen_to_server(reader))
        gameloop_task = asyncio.create_task(game_loop(writer))

        await asyncio.gather(listener_task, gameloop_task)

    except ConnectionRefusedError:
        logging.error(f"AI Client: Connection refused. Is the server running at {SERVER_HOST}:{SERVER_PORT}?")
        print(f"Error: Could not connect AI client to the server at {SERVER_HOST}:{SERVER_PORT}.")
    except Exception as e:
        logging.error(f"An error occurred in AI main: {e}", exc_info=True)
        print(f"An unexpected error occurred in the AI client: {e}")
    finally:
        if writer:
            writer.close()
            try: await writer.wait_closed()
            except Exception: pass
        if pygame.get_init(): pygame.quit()
        logging.info("AI Client shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())