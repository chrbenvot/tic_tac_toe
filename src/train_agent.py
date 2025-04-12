import os
from stable_baselines3 import DQN, PPO, A2C # We'll use DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import gymnasium as gym # Ensure gymnasium is imported if needed for env check etc.

# Adjust import if necessary - assumes tic_tac_toe_env.py is in the same src dir
from tic_tac_toe_env import TicTacToeEnv

# --- Configuration ---
MODEL_ALGORITHM = DQN     # *** Recommended: DQN ***
POLICY_TYPE = 'MlpPolicy' # Use 'MlpPolicy' for flat observation spaces like ours
TOTAL_TIMESTEPS = 100000  # *** Recommended: 100,000 ***
MODEL_SAVE_NAME = "tictactoe_dqn_agent" # Name to save the model as (matches DQN)
import os # Make sure os is imported at the top

# --- Calculate paths relative to this script's location ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # Directory containing train_agent.py (i.e., src/)
project_root = os.path.dirname(script_dir) # Directory containing src/ (i.e., tic_tac_toe/)

LOG_DIR = os.path.join(project_root, "ttt_logs") # Place logs in project root
MODEL_DIR = os.path.join(script_dir, "models")   # Models directory *inside* src/

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True) # This will create src/models/ if it doesn't exist

SAVE_PATH = os.path.join(MODEL_DIR, MODEL_SAVE_NAME) # SAVE_PATH now points inside src/models/

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SAVE_PATH = os.path.join(MODEL_DIR, MODEL_SAVE_NAME)

if __name__ == "__main__":
    print("Creating environment...")
    # For TicTacToe, a single env is usually sufficient and simpler
    env = TicTacToeEnv()
    # Optional: Check the environment before training (good practice)
    # from stable_baselines3.common.env_checker import check_env
    # try:
    #     check_env(env)
    #     print("Environment check passed.")
    # except Exception as e:
    #     print(f"Environment check failed: {e}")
    #     exit() # Exit if env check fails

    print(f"Using Algorithm: {MODEL_ALGORITHM.__name__}")
    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")

    # --- Model Definition ---
    # Using DQN as recommended
    model = DQN(
        POLICY_TYPE,
        env,
        verbose=1,                  # Print training progress
        learning_rate=1e-4,         # Learning rate
        buffer_size=50000,          # Size of the replay buffer
        learning_starts=1000,       # Number of steps before learning starts
        batch_size=64,              # Samples per gradient update
        tau=1.0,                    # Soft update coefficient (1.0 = hard update)
        gamma=0.99,                 # Discount factor for future rewards
        train_freq=4,               # Update the model every 4 steps
        gradient_steps=1,           # How many gradient steps to perform per update
        target_update_interval=500, # Update the target network every N steps
        exploration_fraction=0.2,   # Fraction of training dedicated to exploration decay
        exploration_final_eps=0.02, # Final value of random action probability
        tensorboard_log=LOG_DIR     # Log training statistics for TensorBoard
    )

    # --- Training ---
    # To monitor training: In another terminal, navigate to project dir and run:
    # tensorboard --logdir ./ttt_logs/
    print(f"Starting training... (Monitor with TensorBoard: tensorboard --logdir {LOG_DIR})")
    try:
        # Simple training loop without complex callbacks for now
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,  # Log stats every N training updates
            progress_bar=True # Show a progress bar
        )
        print("Training finished.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # --- Saving Model ---
    # The model is automatically saved with the correct name from SAVE_PATH
    print(f"Saving model to {SAVE_PATH}.zip")
    model.save(SAVE_PATH)
    print("Model saved.")

    # Clean up the environment
    env.close()
    print("Environment closed. Done.")