import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Import logic constants, but be careful not to create circular dependencies
# It's often better to redefine constants here if they are simple
# Or have a separate constants.py file
BOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1 # Agent will typically be Player X
PLAYER_O = 2 # Opponent will be Player O

class TicTacToeEnv(gym.Env):
    """
    Custom Environment for Tic Tac Toe that follows gym interface.
    The agent is always Player X (1).
    The opponent plays randomly.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        super().__init__()

        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = PLAYER_X # Agent always starts
        self.opponent = PLAYER_O

        # Define action and observation space
        # They must be gym.spaces objects
        # Example: Action is discrete, corresponding to cells 0-8
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)

        # Example: Observation is the flattened board state
        # Values can be 0 (empty), 1 (agent X), 2 (opponent O)
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(BOARD_SIZE * BOARD_SIZE,), dtype=np.int8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    def _get_obs(self):
        """Returns the current observation (flattened board)."""
        return self.board.flatten()

    def _get_info(self):
        """Returns auxiliary information (e.g., valid actions)."""
        valid_actions_mask = (self.board.flatten() == EMPTY).astype(np.int8)
        return {"action_mask": valid_actions_mask}


    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed) # Important for reproducibility

        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = PLAYER_X # Agent always starts

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _check_win(self, board, player):
        """Checks if the given player has won (Internal helper)."""
        # Check rows & columns
        for i in range(BOARD_SIZE):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        # Check diagonals
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False

    def _is_draw(self, board):
         """Checks if the board is full."""
         return np.all(board != EMPTY)


    def step(self, action):
        """
        Executes one time step within the environment.
        Agent (Player X) takes an action.
        Opponent (Player O) responds randomly.
        """
        terminated = False
        truncated = False # Not used here, but part of the standard return
        reward = 0

        # 1. Agent's Turn (Player X)
        if self.current_player != PLAYER_X:
             # This shouldn't happen if the flow is correct, but as a safeguard
             raise ValueError("It's not the agent's (Player X) turn.")

        row, col = divmod(action, BOARD_SIZE)

        # Check if the action is valid (cell is empty)
        if self.board[row, col] != EMPTY:
            # Invalid move - penalize heavily and end the episode? Or small penalty and let agent try again?
            # Let's penalize heavily and end. Agent must learn to choose valid spots.
            reward = -10
            terminated = True
            observation = self._get_obs()
            info = self._get_info()
            info["error"] = "Invalid move attempted by agent."
            return observation, reward, terminated, truncated, info

        # Apply agent's valid move
        self.board[row, col] = PLAYER_X

        # Check if agent won
        if self._check_win(self.board, PLAYER_X):
            reward = 1 # Positive reward for winning
            terminated = True
        # Check for draw after agent's move
        elif self._is_draw(self.board):
            reward = 0.5 # Neutral or slightly positive reward for a draw? Let's try 0.5
            terminated = True

        info = self._get_info() # Get info after agent's move

        # 2. Opponent's Turn (Player O), if game not over
        if not terminated:
            self.current_player = PLAYER_O
            valid_opponent_actions = np.where(self.board.flatten() == EMPTY)[0]

            if len(valid_opponent_actions) > 0:
                # Simple random opponent
                opponent_action_idx = self.np_random.choice(valid_opponent_actions)
                opp_row, opp_col = divmod(opponent_action_idx, BOARD_SIZE)
                self.board[opp_row, opp_col] = PLAYER_O

                # Check if opponent won
                if self._check_win(self.board, PLAYER_O):
                    reward = -1 # Negative reward if opponent wins
                    terminated = True
                # Check for draw after opponent's move (can happen if opponent takes last spot)
                elif self._is_draw(self.board):
                    reward = 0.5 # Draw reward
                    terminated = True

            # If no valid moves left for opponent but game wasn't terminated before, it's a draw
            elif not self._is_draw(self.board):
                 # This case should theoretically be covered by previous draw checks, but as safety
                 reward = 0.5
                 terminated = True

            # Switch back to Agent's turn for the next step (if game continues)
            if not terminated:
                self.current_player = PLAYER_X

        # Get final observation and info after both turns (or game end)
        observation = self._get_obs()
        info = self._get_info() # Re-calculate info based on final board state for the step

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info


    def render(self):
        """Renders the environment."""
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "human":
             self._render_frame() # We need a separate rendering mechanism for human mode

    def _render_text(self):
        """Helper for ANSI rendering."""
        rows = []
        for r in range(BOARD_SIZE):
            row_str = " | ".join(['X' if cell == PLAYER_X else 'O' if cell == PLAYER_O else ' ' for cell in self.board[r]])
            rows.append(row_str)
        return "\n" + "---|---|---\n".join(rows) + "\n"

    def _render_frame(self):
        # For 'human' mode, we could potentially integrate Pygame drawing here
        # For simplicity now, just print to console using the ANSI helper
        print(self._render_text())


    def close(self):
        """Performs any necessary cleanup."""
        # If using Pygame for rendering, close it here
        pass

# --- Optional: Check the environment ---
if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    env = TicTacToeEnv()
    # It will check your custom environment and output additional warnings if needed
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")

    # Example usage:
    # obs, info = env.reset()
    # done = False
    # while not done:
    #     action_mask = info['action_mask']
    #     valid_actions = np.where(action_mask == 1)[0]
    #     if len(valid_actions) == 0: break # Should not happen if logic is correct
    #     action = env.action_space.sample(mask=action_mask) # Sample a valid random action
    #     print(f"Action taken: {action}")
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     env.render(mode="ansi")
    #     print(f"Observation: {obs}")
    #     print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    #     print("-" * 10)
    #     done = terminated or truncated
    # env.close()