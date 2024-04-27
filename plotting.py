from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np


# Function to calculate a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot(frame_idx, rewards, losses, game, game_data):
    clear_output(True)
    plt.figure(figsize=(20,5))

    moving_avg_window = 10  #How many data points to consider for the moving average

    # Plotting the rewards
    plt.subplot(131)
    plt.title(f'Frame: {frame_idx}, Avg Reward: {np.mean(rewards[-10:]):.2f}')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards, color='b', label="Rewards")
    # Add a moving average as a trend line
    if len(rewards) >= moving_avg_window:
        reward_trend = moving_average(rewards, moving_avg_window)
        plt.plot(
            range(moving_avg_window - 1, len(rewards)),  # Align the x-axis
            reward_trend,
            color='orange',
            label=f'Moving Avg (window={moving_avg_window})'
        )
    plt.grid(True)
    plt.legend()

    # Losses Plot
    plt.subplot(132)
    plt.title("Loss Over Time")
    plt.xlabel("Frames")
    plt.ylabel("Loss")
    plt.plot(losses, color='r', label="Losses")
    plt.grid(True)
    plt.legend()

    plt.show()