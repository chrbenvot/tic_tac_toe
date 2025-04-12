import os
import time # To time the training
from stable_baselines3 import PPO # *** Switched to PPO ***
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import gymnasium as gym

# Adjust import if necessary
from tic_tac_toe_env import TicTacToeEnv

# --- Configuration ---
MODEL_ALGORITHM = PPO     # *** Using PPO ***
# *** Slightly deeper network: [Input] -> 64 neurons -> 64 neurons -> [Output] ***
# Default is [64, 64]. Let's keep it simple for now, maybe [128, 128] if needed later.
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64])) # Standard architecture for MlpPolicy
POLICY_TYPE = 'MlpPolicy'
TOTAL_TIMESTEPS = 1_000_000  # *** Significantly Increased: 1 Million Timesteps ***
                            # (Adjust based on patience: 500k minimum, 2M+ even better)
MODEL_SAVE_NAME = "tictactoe_ppo_1M" # *** Updated Name ***
LOG_DIR = "./ttt_logs_powerful/"   # *** Separate logs directory ***
MODEL_DIR = "./models/"   # Save in the same models directory (within src if you modified paths)

# --- Path Calculation (Use the version that matches your structure) ---
# Option 1: If 'models' is in project root
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# MODEL_DIR = os.path.join(project_root, "models")
# LOG_DIR = os.path.join(project_root, "ttt_logs_powerful")

# Option 2: If 'models' is inside 'src' (use this if you modified paths earlier)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODEL_DIR = os.path.join(script_dir, "models")
LOG_DIR = os.path.join(project_root, "ttt_logs_powerful") # Keep logs in root for simplicity

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SAVE_PATH = os.path.join(MODEL_DIR, MODEL_SAVE_NAME)

if __name__ == "__main__":
    start_time = time.time()
    print("Creating environment...")
    env = TicTacToeEnv()

    print(f"Using Algorithm: {MODEL_ALGORITHM.__name__}")
    print(f"Policy Network Architecture: {POLICY_KWARGS}")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...") # Format number with commas

    # --- Model Definition ---
    # PPO hyperparameters can also be tuned, using defaults for now.
    model = PPO(
        POLICY_TYPE,
        env,
        policy_kwargs=POLICY_KWARGS, # Pass the network architecture
        verbose=1,
        tensorboard_log=LOG_DIR,
        # Common PPO parameters you could tune later:
        # n_steps=2048, # Steps per rollout collection per environment
        # batch_size=64,
        # n_epochs=10,
        # learning_rate=3e-4,
        # gamma=0.99, # Discount factor
        # gae_lambda=0.95, # Factor for Generalized Advantage Estimation
        # clip_range=0.2, # PPO clipping parameter
    )

    # --- Training ---
    print(f"Starting training... (Monitor with TensorBoard: tensorboard --logdir {LOG_DIR})")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10,
            progress_bar=True
        )
        print("Training finished.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # --- Saving Model ---
    print(f"Saving model to {SAVE_PATH}.zip")
    model.save(SAVE_PATH)
    print("Model saved.")

    # --- Timing ---
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training took {training_duration:.2f} seconds ({training_duration/60:.2f} minutes).")

    # Clean up
    env.close()
    print("Environment closed. Done.")