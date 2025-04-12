import asyncio
import json
import random
import logging
import numpy as np
# Adjust import if necessary
from tic_tac_toe_logic import (
    create_initial_state, get_valid_actions, apply_action,
    check_win_condition, check_draw_condition, get_next_player,
    BOARD_SIZE, PLAYER_X, PLAYER_O, board_to_string
)

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 8888        # Port for clients to connect to
logging.basicConfig(level=logging.INFO)

# --- Server State ---
# Dictionary to store active games. Key: game_id, Value: game state dict
games = {}
# Dictionary to store connected clients waiting for a game. Key: writer obj, Value: reader obj (or vice versa)
waiting_clients = {}
# Dictionary mapping writers to their game_id
client_to_game = {}

async def send_json(writer, data):
    """Encodes data to JSON and sends it to the client."""
    try:
        message = json.dumps(data).encode('utf-8')
        writer.write(message + b'\n') # Add newline as delimiter
        await writer.drain()
        logging.debug(f"Sent to {writer.get_extra_info('peername')}: {data}")
    except ConnectionResetError:
        logging.warning(f"Connection reset when sending to {writer.get_extra_info('peername')}")
        # Handle disconnection logic elsewhere if needed
    except Exception as e:
        logging.error(f"Error sending data to {writer.get_extra_info('peername')}: {e}")


async def broadcast(game_id, message_data, exclude_writer=None):
    """Sends a message to both players in a specific game."""
    if game_id in games:
        game = games[game_id]
        player1_writer = game.get('player1_writer')
        player2_writer = game.get('player2_writer')

        if player1_writer and player1_writer != exclude_writer:
            await send_json(player1_writer, message_data)
        if player2_writer and player2_writer != exclude_writer:
            await send_json(player2_writer, message_data)

async def start_game(writer1, writer2):
    """Starts a new game session for two connected clients."""
    game_id = f"game_{random.randint(1000, 9999)}"
    logging.info(f"Starting new game: {game_id} between {writer1.get_extra_info('peername')} and {writer2.get_extra_info('peername')}")

    game_state = {
        'board': create_initial_state().tolist(), # Store as list for JSON
        'current_turn': PLAYER_X,
        'player1_writer': writer1, # Player X
        'player2_writer': writer2, # Player O
        'players': {PLAYER_X: writer1, PLAYER_O: writer2}
    }
    games[game_id] = game_state
    client_to_game[writer1] = game_id
    client_to_game[writer2] = game_id

    # Notify players the game has started
    await send_json(writer1, {"type": "GAME_START", "player_id": PLAYER_X, "message": "Game started. You are Player X."})
    await send_json(writer2, {"type": "GAME_START", "player_id": PLAYER_O, "message": "Game started. You are Player O."})

    # Send initial state
    initial_state_msg = {"type": "STATE_UPDATE", "board": game_state['board'], "current_turn": game_state['current_turn']}
    await broadcast(game_id, initial_state_msg)


async def handle_client(reader, writer):
    """Handles communication with a single connected client."""
    peername = writer.get_extra_info('peername')
    logging.info(f"Client connected: {peername}")

    # Add client to waiting list or start a game
    if not waiting_clients:
        waiting_clients[writer] = reader
        await send_json(writer, {"type": "WAITING", "message": "Waiting for an opponent..."})
        logging.info(f"Client {peername} is waiting.")
    else:
        # Pair with waiting client
        other_writer, other_reader = waiting_clients.popitem() # Get the waiting client
        await start_game(other_writer, writer) # Player X is the one who waited, Player O is the new one

    game_id = None
    player_id = None # Will be assigned in start_game

    try:
        while True:
            # Read data from the client
            data = await reader.readuntil(b'\n') # Read until newline delimiter
            if not data: # Connection closed by client
                logging.info(f"Client {peername} closed connection (no data).")
                break

            message_str = data.decode('utf-8').strip()
            logging.debug(f"Received from {peername}: {message_str}")

            try:
                message = json.loads(message_str)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON received from {peername}: {message_str}")
                await send_json(writer, {"type": "ERROR", "message": "Invalid JSON format."})
                continue # Ignore invalid message

            # --- Process Client Messages ---
            msg_type = message.get("type")

            if writer not in client_to_game:
                 # Only allow initial messages if not yet in a game (should normally be handled by pairing logic)
                 logging.warning(f"Received message from {peername} who is not in a game: {message_str}")
                 continue

            game_id = client_to_game[writer]
            if game_id not in games:
                logging.error(f"Game {game_id} not found for client {peername}. Aborting connection.")
                await send_json(writer, {"type": "ERROR", "message": "Internal server error: Game not found."})
                break

            game = games[game_id]
            current_player_in_game = game['current_turn']

            # Identify which player this writer corresponds to
            if game['player1_writer'] == writer:
                player_id = PLAYER_X
            elif game['player2_writer'] == writer:
                player_id = PLAYER_O
            else:
                 logging.error(f"Writer {peername} not found in game {game_id}.")
                 break # Should not happen

            if msg_type == "MAKE_MOVE":
                if player_id != current_player_in_game:
                    await send_json(writer, {"type": "INVALID_MOVE", "message": "Not your turn."})
                    continue

                cell_index = message.get("cell")
                if cell_index is None or not (0 <= cell_index < BOARD_SIZE * BOARD_SIZE):
                     await send_json(writer, {"type": "INVALID_MOVE", "message": "Invalid cell index."})
                     continue

                row, col = divmod(cell_index, BOARD_SIZE)
                action = (row, col)
                board_np = np.array(game['board']) # Convert back to numpy for logic checks

                # Check if move is valid using logic function
                if action not in get_valid_actions(board_np):
                    await send_json(writer, {"type": "INVALID_MOVE", "message": "Cell is already occupied or invalid."})
                    continue

                # Apply the move
                try:
                    new_board_np = apply_action(board_np, action, player_id)
                    game['board'] = new_board_np.tolist() # Update game state (store as list)
                    logging.info(f"Game {game_id}: Player {player_id} moved to {action}")
                    print(f"Game {game_id} board:\n{board_to_string(new_board_np)}") # Log board state to console

                    # Check for win/draw
                    if check_win_condition(new_board_np, player_id):
                        logging.info(f"Game {game_id}: Player {player_id} wins!")
                        await broadcast(game_id, {"type": "GAME_OVER", "winner": player_id, "board": game['board']})
                        # Clean up game after notifying players
                        games.pop(game_id, None)
                        client_to_game.pop(game.get('player1_writer'), None)
                        client_to_game.pop(game.get('player2_writer'), None)
                        break # End connection handling for this client

                    elif check_draw_condition(new_board_np):
                        logging.info(f"Game {game_id}: Draw!")
                        await broadcast(game_id, {"type": "GAME_OVER", "winner": 0, "board": game['board']}) # 0 for draw
                        # Clean up game
                        games.pop(game_id, None)
                        client_to_game.pop(game.get('player1_writer'), None)
                        client_to_game.pop(game.get('player2_writer'), None)
                        break # End connection handling

                    else:
                        # Switch turn and broadcast new state
                        game['current_turn'] = get_next_player(player_id)
                        state_update_msg = {"type": "STATE_UPDATE", "board": game['board'], "current_turn": game['current_turn']}
                        await broadcast(game_id, state_update_msg)

                except ValueError as e: # Should be caught by get_valid_actions, but as safeguard
                     logging.warning(f"Invalid move attempted by {peername} despite checks: {e}")
                     await send_json(writer, {"type": "INVALID_MOVE", "message": str(e)})

            # Add handling for other message types if needed (e.g., CHAT, DISCONNECT explicitly)

    except asyncio.IncompleteReadError:
        logging.info(f"Client {peername} disconnected unexpectedly (IncompleteReadError).")
    except ConnectionResetError:
         logging.info(f"Client {peername} reset the connection.")
    except Exception as e:
        logging.error(f"Error handling client {peername}: {e}", exc_info=True) # Log stack trace
    finally:
        logging.info(f"Cleaning up connection for {peername}")
        # Remove client from waiting list if they were there
        waiting_clients.pop(writer, None)

        # Remove client from game and notify opponent if game was active
        game_id = client_to_game.pop(writer, None)
        if game_id and game_id in games:
            game = games[game_id]
            opponent_writer = None
            if game.get('player1_writer') == writer:
                opponent_writer = game.get('player2_writer')
            elif game.get('player2_writer') == writer:
                 opponent_writer = game.get('player1_writer')

            if opponent_writer:
                logging.info(f"Notifying opponent in game {game_id} about disconnect.")
                await send_json(opponent_writer, {"type": "OPPONENT_DISCONNECTED"})
                client_to_game.pop(opponent_writer, None) # Remove opponent too as game is over

            # Remove game itself
            games.pop(game_id, None)
            logging.info(f"Removed game {game_id} due to client disconnect.")


        writer.close()
        await writer.wait_closed()
        logging.info(f"Connection closed for {peername}")


async def main():
    """Main function to start the server."""
    server = await asyncio.start_server(
        handle_client, HOST, PORT)

    addr = server.sockets[0].getsockname()
    logging.info(f'Server listening on {addr}')

    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutting down.")