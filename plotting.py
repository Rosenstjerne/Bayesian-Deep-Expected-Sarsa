import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Plot function
def plot(frame_idx, rewards, losses):
    plt.figure(figsize=(20,5))

    # Customizing the background
    plt.rcParams['axes.facecolor'] = '#f7f7f7'
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
    plt.plot(rewards, color='#0a75ad', label="Rewards")
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
