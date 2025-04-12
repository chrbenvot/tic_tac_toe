import numpy as np

BOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

def create_initial_state():
    """Creates an empty Tic Tac Toe board."""
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

def get_valid_actions(board):
    """Returns a list of (row, col) tuples for empty cells."""
    actions = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == EMPTY:
                actions.append((r, c))
    return actions

def apply_action(board, action, player):
    """
    Applies a move to the board.
    Returns a new board state.
    Raises ValueError if the action is invalid.
    """
    row, col = action
    if board[row, col] != EMPTY:
        raise ValueError(f"Cell {action} is not empty.")
    new_board = board.copy()
    new_board[row, col] = player
    return new_board

def check_win_condition(board, player):
    """Checks if the given player has won."""
    # Check rows
    for r in range(BOARD_SIZE):
        if np.all(board[r, :] == player):
            return True
    # Check columns
    for c in range(BOARD_SIZE):
        if np.all(board[:, c] == player):
            return True
    # Check diagonals
    if np.all(np.diag(board) == player):
        return True
    if np.all(np.diag(np.fliplr(board)) == player): # Flipped left-right diagonal
        return True
    return False

def check_draw_condition(board):
    """Checks if the game is a draw (board full, no winner)."""
    # Assumes win condition is checked first
    return np.all(board != EMPTY) # True if no cells are EMPTY

def get_next_player(current_player):
    """Switches the player."""
    return PLAYER_O if current_player == PLAYER_X else PLAYER_X

def board_to_string(board):
    """Optional: Helper to print the board nicely."""
    rows = []
    for r in range(BOARD_SIZE):
        row_str = " | ".join(['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else ' ' for cell in board[r]])
        rows.append(row_str)
    return "\n" + "---|---|---\n".join(rows) + "\n"

# --- You can add simple test calls here for quick checks if needed ---
# if __name__ == '__main__':
#     b = create_initial_state()
#     print("Initial Board:")
#     print(board_to_string(b))
#     b = apply_action(b, (0, 0), PLAYER_X)
#     b = apply_action(b, (1, 1), PLAYER_O)
#     print("\nBoard after moves:")
#     print(board_to_string(b))
#     print(f"Valid actions: {get_valid_actions(b)}")