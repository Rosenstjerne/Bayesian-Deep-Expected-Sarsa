from collections import deque
import random
import numpy as np


class Replay_Buffer:
    """
    Replay Buffer class for storing
    experiences and sampling for training.

    Args:
        capacity (int): Maximum size of the buffer.

    Attributes:
        buffer (deque): A deque object to store experiences.

    """
    def __init__(self, capacity):
        # Init replay buffer with a maximum capacity, deque for efficient appending and popping
        self.buffer = deque(maxlen=capacity)
    
    def store(self, state, action, new_state, reward, done):
        # Store new experience in replay buffer
        # Expand_dims to add a dimension to the array for consistency in shape
        state = np.expand_dims(state, 0)
        new_state = np.expand_dims(new_state, 0)
        
        # Append the experience to the buffer.
        self.buffer.append([state, action, new_state, reward, done])
    
    def replay(self, batch_size):
        # Sample a batch of experiences from the buffer for training.
        # The batch size determines how many experiences to sample.
        
        # Unzip the sampled experiences into separate components.
        state, action, new_state, reward, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        # Concatenate the state and new_state arrays for consistent batch shape.
        return np.concatenate(state), action, np.concatenate(new_state), reward, done
    
    def __len__(self):
        # Return the current size of the buffer.
        return len(self.buffer)