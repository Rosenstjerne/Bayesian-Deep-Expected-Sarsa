from collections import deque
import torch
import random
import numpy as np
from DQN import *


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    EPSILON_START = 1.0
    EPSILON_END = 0.1 #Insures there is always some exploration
    EPSILON_DECAY = 5e5 # Higher = slower decay
    TAU = 1

    epsilon = EPSILON_START

    #TODO Plagiat filter VVV
    def softmax(self, qvalues):
        preferences = qvalues / self.TAU
        max_preference = torch.argmax(qvalues) / self.TAU
        reshaped_max_preference = max_preference.reshape((-1, 1)) # Reshape maybe not needed

        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = torch.exp(preferences - reshaped_max_preference)
        # Compute the denominator, i.e., the sum over the numerator along the actions axis.
        sum_of_exp_preferences = torch.sum(exp_preferences)

        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences
        action_probs = action_probs.squeeze()
        return action_probs

    
    def qvalue(self, state):
        return self.model(state).detach()

    def action(self, state):
        qvalues = self.qvalue(state)
        action_prob = self.softmax(qvalues)
        action = np.random.choice(self.num_actions, p=action_prob.squeeze())
        return action


    def optimal_action(self, state):
        qvalues = self.qvalue(state)
        action = torch.argmax(qvalues) 
        return action.item()

    def update(self, state, next_state, reward, action, next_action):
        predict = self.optimal_action(state)
        
        expected_q = self.qvalue(next_state)
        q_max = self.optimal_action(next_state)
        #TODO calc some kind of loss?
    
    def select_action(self, state, policy_net):
        
        if np.random.random_sample() <= self.epsilon:
                action = np.random.choice(self.num_actions)
        else:
            action = self.optimal_action(state)

        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)
        
    # TODO Decay Epsilon
        # If epsilon != finale epsilon
        # epsilon = epsilon * 0.9
    epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (steps_done / EPSILON_DECAY))
        
