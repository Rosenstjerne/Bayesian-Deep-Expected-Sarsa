import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Function to calculate a moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Data for plotting
frame_idx = 100
rewards = np.random.normal(0, 1, 100).cumsum()  # Simulated rewards data
losses = np.random.normal(0, 1, 100).cumsum()   # Simulated losses data

# Plot function
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))

    # Customizing the background
    plt.rcParams['axes.facecolor'] = '#f3fafd'
    plt.rcParams['axes.edgecolor'] = 'none'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.linewidth'] = 1

    # Plotting the rewards
    plt.subplot(131)
    plt.title(f'Frame: {frame_idx}, Avg Reward: {np.mean(rewards[-10:]):.2f}')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(rewards, color='b', label="Rewards")
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
    plt.tight_layout()

    plt.show()

plot(frame_idx, rewards, losses)
