import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# --- Configuration ---
# Adjust this path to your specific log directory
# It should point to the *parent* directory containing the run folder (e.g., PPO_1)
LOG_DIR_PARENT = "./ttt_logs_powerful/"
# The specific metric we want to plot from the logs
REWARD_TAG = 'rollout/ep_rew_mean'
# --- End Configuration ---

def find_latest_run_dir(parent_log_dir):
    """Finds the latest run directory (e.g., PPO_1, PPO_2) inside the parent log dir."""
    subdirs = [os.path.join(parent_log_dir, d) for d in os.listdir(parent_log_dir)
               if os.path.isdir(os.path.join(parent_log_dir, d))]
    if not subdirs:
        return None
    # Assume latest is the one with the highest number or most recent modification time
    # Simple approach: just take the first one found if only one exists, or prompt user/fail
    # More robust: check modification times or parse names like PPO_1, PPO_2
    if len(subdirs) == 1:
        return subdirs[0]
    else:
        # If multiple runs exist, try sorting by name (assuming format like Algo_N)
        try:
            subdirs.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)
            print(f"Found multiple run directories, using latest: {os.path.basename(subdirs[0])}")
            return subdirs[0]
        except:
            print(f"Warning: Could not determine latest run directory automatically from {subdirs}. Using the first one found.")
            return subdirs[0]


def plot_reward_curve(log_dir, tag):
    """Reads TensorBoard logs and plots the specified scalar tag."""

    print(f"Attempting to load logs from: {log_dir}")
    if not os.path.isdir(log_dir):
        print(f"Error: Log directory not found at {log_dir}")
        return

    # Initialize the EventAccumulator
    # Set size_guidance for scalars; 0 means load all data points
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={event_accumulator.SCALARS: 0})

    # Load the events from the log file
    try:
        print("Loading events... (This may take a moment for large logs)")
        ea.Reload()
        print("Events loaded.")
    except Exception as e:
        print(f"Error loading event data: {e}")
        return

    # Check if the desired tag exists
    if tag not in ea.Tags()['scalars']:
        print(f"Error: Tag '{tag}' not found in logs.")
        print("Available scalar tags:", ea.Tags()['scalars'])
        return

    # Extract the scalar events for the tag
    scalar_events = ea.Scalars(tag)

    # Extract steps and values
    steps = np.array([event.step for event in scalar_events])
    values = np.array([event.value for event in scalar_events])

    if len(steps) == 0:
        print(f"No data found for tag '{tag}'.")
        return

    print(f"Plotting {len(steps)} data points for '{tag}'...")

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, values)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title(f"Training Curve: {tag}")
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

if __name__ == "__main__":
    # Find the specific run directory (e.g., PPO_1) inside the parent log dir
    run_log_dir = find_latest_run_dir(LOG_DIR_PARENT)

    if run_log_dir:
        plot_reward_curve(run_log_dir, REWARD_TAG)
    else:
        print(f"Error: No run directory found inside {LOG_DIR_PARENT}")