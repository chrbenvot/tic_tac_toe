import pygame
import asyncio
import json
import sys
import numpy as np
import logging
import os
from stable_baselines3 import PPO # *** Import PPO instead of DQN ***

# Adjust import if necessary
from tic_tac_toe_logic import (
    BOARD_SIZE, PLAYER_X, PLAYER_O, EMPTY
)

# --- Configuration ---
SERVER_HOST = '127.0.0.1' # Change to server's IP if not running locally
SERVER_PORT = 8888

# --- Path Calculation (Use the version that matches your structure) ---
# Option 1: If 'models' is in project root
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# MODEL_DIR = os.path.join(project_root, "models")

# Option 2: If 'models' is inside 'src' (use this if you modified paths earlier)
script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir) # Not needed here if model_dir is relative to script
MODEL_DIR = os.path.join(script_dir, "models")

MODEL_NAME = "tictactoe_ppo_1M.zip" # *** Ensure this matches your PPO model file ***
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
AI_THINK_DELAY = 0.3 # Optional delay (can be shorter if PPO is faster)

logging.basicConfig(level=logging.INFO)

# Pygame constants (same as before)
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
status_message = "Connecting PPO AI..." # Updated status
ai_model = None # Will hold the loaded PPO model

# --- Pygame Setup ---
screen = None
font = None

# --- Pygame Drawing Functions (Identical to previous ai_client.py) ---
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

# --- Networking Functions (Identical to previous ai_client.py) ---
async def send_message(writer_local, data):
    if not writer_local: return
    try:
        message = json.dumps(data).encode('utf-8')
        writer_local.write(message + b'\n')
        await writer_local.drain()
        logging.debug(f"PPO AI Sent: {data}")
    except Exception as e:
        logging.error(f"PPO AI Error sending message: {e}")
        global status_message, game_over
        status_message = "PPO AI Connection error"
        game_over = True

async def listen_to_server(reader_local):
    global board, current_turn, player_id, game_over, status_message # Make sure all globals are declared
    try:
        while True:
            data = await reader_local.readuntil(b'\n')
            if not data:
                logging.warning("PPO AI: Server closed connection.")
                status_message = "PPO AI: Server disconnected."
                game_over = True
                break

            message_str = data.decode('utf-8').strip()
            logging.debug(f"PPO AI Received: {message_str}")

            try:
                message = json.loads(message_str)
                msg_type = message.get("type")

                # Update state based on message type (same logic as human client)
                if msg_type == "GAME_START":
                    player_id = message.get("player_id")
                    status_message = message.get("message", f"PPO AI is Player {player_id}")
                    logging.info(f"PPO AI Game started. Assigned Player ID: {player_id}")
                elif msg_type == "STATE_UPDATE":
                    board = np.array(message.get("board", []))
                    current_turn = message.get("current_turn")
                    # Status update logic is handled in game_loop now based on turn
                elif msg_type == "GAME_OVER":
                    board = np.array(message.get("board", board))
                    winner = message.get("winner")
                    game_over = True
                    if winner == player_id: status_message = "PPO AI wins!"
                    elif winner == 0: status_message = "PPO AI: Draw!"
                    else: status_message = "PPO AI loses!"
                    logging.info(f"PPO AI Game Over. Winner: {winner}")
                elif msg_type == "INVALID_MOVE":
                    status_message = f"PPO AI received: {message.get('message', 'Invalid Move')}"
                    logging.warning(f"PPO AI received invalid move: {message.get('message')}")
                elif msg_type == "WAITING":
                    status_message = message.get("message", "PPO AI waiting for opponent...")
                elif msg_type == "OPPONENT_DISCONNECTED":
                    status_message = "PPO AI: Opponent disconnected. AI wins!"
                    game_over = True
                elif msg_type == "ERROR":
                     status_message = f"PPO AI Server Error: {message.get('message', 'Unknown error')}"
                     game_over = True

            except json.JSONDecodeError:
                logging.warning(f"PPO AI: Invalid JSON received: {message_str}")
            except Exception as e:
                logging.error(f"PPO AI: Error processing server message: {e}", exc_info=True)
                status_message = "PPO AI Processing Error"
                game_over = True
                break

    except asyncio.IncompleteReadError:
        logging.warning("PPO AI: Server disconnected unexpectedly.")
        status_message = "PPO AI: Server disconnected."
    except ConnectionResetError:
         logging.warning("PPO AI: Connection to server reset.")
         status_message = "PPO AI: Connection Reset."
    except Exception as e:
        logging.error(f"PPO AI: Error in listen_to_server: {e}", exc_info=True)
        status_message = "PPO AI Network Error"
    finally:
        logging.info("PPO AI Listener task finished.")
        game_over = True

# --- AI Move Function (Identical logic, uses global ai_model) ---
def get_ai_move(current_board, ai_player_id):
    if ai_model is None:
        logging.error("PPO AI model is not loaded!")
        return None

    observation = current_board.flatten().astype(np.int8)
    canonical_obs = observation.copy()
    if ai_player_id == PLAYER_O:
        player_x_mask = (canonical_obs == PLAYER_X)
        player_o_mask = (canonical_obs == PLAYER_O)
        canonical_obs[player_x_mask] = PLAYER_O
        canonical_obs[player_o_mask] = PLAYER_X

    try:
        # *** Use the loaded PPO model ***
        action_index, _ = ai_model.predict(canonical_obs, deterministic=True)
        action_index = int(action_index)

        # --- Validate Action (Optional but Recommended) ---
        row, col = divmod(action_index, BOARD_COLS)
        if current_board[row, col] != EMPTY:
            logging.warning(f"PPO AI model predicted invalid move ({action_index}). Falling back to random.")
            valid_actions = np.where(current_board.flatten() == EMPTY)[0]
            if len(valid_actions) > 0:
                 action_index = int(np.random.choice(valid_actions))
            else:
                 return None
        logging.info(f"PPO AI (Player {ai_player_id}) predicts move index: {action_index}")
        return action_index

    except Exception as e:
        logging.error(f"Error during PPO AI prediction: {e}", exc_info=True)
        return None


# --- Game Loop (Identical logic, uses global ai_model via get_ai_move) ---
async def game_loop(writer_local):
    global status_message
    running = True
    last_ai_move_time = asyncio.get_event_loop().time()

    while running:
        if not screen or not font:
            await asyncio.sleep(0.1)
            continue

        current_time = asyncio.get_event_loop().time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not game_over and current_turn == player_id:
            if current_time - last_ai_move_time >= AI_THINK_DELAY:
                status_message = f"PPO AI (Player {player_id}) is thinking..."
                display_status(status_message)
                pygame.display.update()

                action_index = get_ai_move(board, player_id) # Calls the modified function

                if action_index is not None:
                    await send_message(writer_local, {"type": "MAKE_MOVE", "cell": action_index})
                else:
                    logging.error("PPO AI could not determine a valid move.")
                    status_message = "PPO AI Error: No move"
                last_ai_move_time = current_time
        elif not game_over and current_turn is not None:
             other_player_symbol = 'X' if current_turn == PLAYER_X else 'O'
             status_message = f"Player {other_player_symbol}'s turn"

        screen.fill(BG_COLOR)
        draw_lines()
        draw_figures()
        display_status(status_message)
        pygame.display.update()

        if game_over and running:
            pass

        await asyncio.sleep(0.05)

    if writer_local:
        writer_local.close()
        await writer_local.wait_closed()
    pygame.quit()
    logging.info("PPO AI Pygame quit.")


# --- Main Function (Loads PPO model) ---
async def main():
    global reader, writer, screen, font, status_message, ai_model # Ensure ai_model is global

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at: {MODEL_PATH}")
        print(f"Error: Trained model '{MODEL_NAME}' not found in '{MODEL_DIR}'.")
        print("Please ensure you have run the powerful training script first.")
        return
    try:
        # *** Load the PPO model ***
        ai_model = PPO.load(MODEL_PATH)
        logging.info(f"PPO AI model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logging.error(f"Error loading PPO AI model: {e}", exc_info=True)
        print(f"Error: Could not load the AI model from {MODEL_PATH}. Ensure it's a valid Stable Baselines 3 PPO model.")
        return

    # --- Connect and Run (Identical logic) ---
    try:
        reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
        logging.info(f"PPO AI Client connected to server at {SERVER_HOST}:{SERVER_PORT}")
        status_message = "PPO AI Connected, waiting..."

        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Networked Tic Tac Toe - PPO AI Client") # Updated caption
        font = pygame.font.Font(None, 36)

        listener_task = asyncio.create_task(listen_to_server(reader))
        gameloop_task = asyncio.create_task(game_loop(writer))

        await asyncio.gather(listener_task, gameloop_task)

    except ConnectionRefusedError:
        logging.error(f"PPO AI Client: Connection refused.")
        print(f"Error: Could not connect PPO AI client to the server.")
    except Exception as e:
        logging.error(f"An error occurred in PPO AI main: {e}", exc_info=True)
        print(f"An unexpected error occurred in the PPO AI client: {e}")
    finally:
        if writer:
            writer.close()
            try: await writer.wait_closed()
            except Exception: pass
        if pygame.get_init(): pygame.quit()
        logging.info("PPO AI Client shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())