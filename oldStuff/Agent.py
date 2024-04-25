import torch
import random
import numpy as np
import gymnasium as gym
import torch.optim as optim

#custom imports
from DQN import *
from replayBuffer import *

class Agent:
    def __init__(self, strategy, env, device):
        self.current_step = 0
        self.strategy = strategy
        self.env = gym.make(env)
        self.device = device

    #TODO Move into __init__ VVV
    EPSILON_START = 1.0
    EPSILON_END = 0.1 #Insures there is always some exploration
    EPSILON_DECAY = 5e5 # Higher = slower decay
    ALPHA = 1
    GAMMA = 1
    TAU = 1
    N_FRAMES = 4
    batch_size = 32
    env = gym.make("Breakout-v4")
    epsilon = EPSILON_START
    num_actions = env.action_space.n
    memorybuffer = ReplayBuffer(10000)

    model = DQN(N_FRAMES, num_actions)
    optimizer = optim.Adam(model.parameters(), lr=ALPHA)

    #TODO Plagiat filter VVV
    def softmax(self, qvalues): #Can maybe be replaced by torch.softmax
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
    
    #   Approximated Q function
    def Q(self, state):
        return self.model(state).detach()

    #   Probabilistic action selection with epsilon pertubation
    def action(self, state):
        if np.random.random_sample() <= self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            Q = self.Q(state)
            action_prob = self.softmax(Q)
            action = np.random.choice(self.num_actions, p=action_prob.squeeze())
        return action

    def optimal_action(self, state):
        Q = self.Q(state)
        action = torch.argmax(Q) 
        return action.item()

    def process(self, state):
        return torch.tensor(state, dtype=torch.float)

    def train(self):
        #Initial state
        env = self.env
        done = False
        trunc = False
        total_reward = 0
        buffer = self.memorybuffer
        self.t = 0

        state = env.reset()
        state = self.process(state)
        while not done or not trunc:
            # Ensures we have correct number of frames for Network Input
            while state.size()[1] < self.N_FRAMES:
                #We need at least 4 frames to use the network
                #Therfore arbitrary action is chosen
                action = 1

                new_frame, reward, done, trunc , _ = env.step(action)
                new_frame = self.process(new_frame)

                state = torch.cat([state, new_frame], 1)
            #After this while state.size()[1] = 4¨
            
            #Gets action from network
            action = self.action(state)
            new_frame, reward, done, trunc, _= env.step(action)
            new_frame = self.process(new_frame)

            new_state = torch.cat([state, new_frame], 1)
            new_state = new_state[:, 1:, :, :] #Remove first

            if buffer > self.N_STEPS:
                reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                action = torch.tensor([action], device=self.device, dtype=torch.long)
                done = torch.tensor([done], device=self.device, dtype=torch.uint8)
                
                buffer.push(state, action, reward, new_state, done )

            state = new_state
            total_reward += reward #Adds reward to total reward
            self.t += 1 #Increments time step t
            
            #TODO Add learning
            #   Load sample from buffer

            state, action, reward, terminal, next_state = self.retrieve(self.batch_size)
            q = self.model(state).gather(1, action.view(self.batch_size, 1))
            
            state, action, reward, terminal, next_state = self.retrieve(self.batch_size)
            
            qcalc = reward + (1-int(terminal))*(self.GAMMA * sum(self.Q(next_state)[act] * self.softmax(self.Q(next_state))[act] for act in range(self.num_actions)-self.Q(state)[action]))
            
            self.optimizer.zero_grad()
            loss = self.loss(q - qcalc)
            loss.backward()
            self.optimizer.step()

            
            #Train on buffer
            #   Calc loss from buffer?
            #   Update model?
            #   Choose action from model?
            #   step(action)?
            #   Push to buffer?
            #   Rins repeat?

            #Decays epsilon
            self.epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (steps_done / EPSILON_DECAY))
            
            #TODO Maybe just take Melih's calculation. It should be possible with the current pieces.

            # Q[s][a] = Q[s][a] + alpha*(r + gamma*((1-epsilon)*Q[s_new][best_action]+(epsilon/mdp.A)*sum(Q[s_new][act] for act in range(mdp.A))) - Q[s][a])
            expectedq = (epsilon/num_actions) * sum(Q(ns)[act] * softmax(Q(ns))[act] for act in range(num_actions)) - Q(s)[a]
            #qreal = Q(s)[a] + ALPHA * (r + GAMMA * ((1-epsilon) * Q(ns)[optimal_action(ns)])) + expectedq
            qestim = (r + GAMMA * ((1-epsilon) * Q(ns)[optimal_action(ns)])) + expectedq
            # TODO Laundry
            #   in "sum" add probability distribution of action (prob(a|ns)) [IS DONE?]
            #   ns picked from probability?
            #   move into update()?
            #   loss = Q(s)[a] - qestim ? 

    """
 __    __   ___ _        __  ___  ___ ___   ___      ______  ___       ______ __ __   ___       ____ ____   ____ __ __   ___ __ __  ____ ____  ___   
|  |__|  | /  _| |      /  ]/   \|   |   | /  _]    |      |/   \     |      |  |  | /  _]     /    |    \ /    |  |  | /  _|  |  |/    |    \|   \  
|  |  |  |/  [_| |     /  /|     | _   _ |/  [_     |      |     |    |      |  |  |/  [_     |   __|  D  |  o  |  |  |/  [_|  |  |  o  |  D  |    \ 
|  |  |  |    _| |___ /  / |  O  |  \_/  |    _]    |_|  |_|  O  |    |_|  |_|  _  |    _]    |  |  |    /|     |  |  |    _|  ~  |     |    /|  D  |
|  `  '  |   [_|     /   \_|     |   |   |   [_       |  | |     |      |  | |  |  |   [_     |  |_ |    \|  _  |  :  |   [_|___, |  _  |    \|     |
 \      /|     |     \     |     |   |   |     |      |  | |     |      |  | |  |  |     |    |     |  .  |  |  |\   /|     |     |  |  |  .  |     |
  \_/\_/ |_____|_____|\____|\___/|___|___|_____|      |__|  \___/       |__| |__|__|_____|    |___,_|__|\_|__|__| \_/ |_____|____/|__|__|__|\_|_____|
                                                                                                                                                                                                            
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠄⠒⠒⠢⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢞⠏⠀⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⡎⡎⢀⠄⠀⣀⠀⠀⣀⠀⠄⡇⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⢃⢂⣴⣿⣿⡧⢸⣿⣷⢨⠃⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⠂⠙⢿⣿⣷⡾⡛⠋⢪⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢺⣧⣀⣀⣠⣿⠒⠋⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠉⠻⠿⠋⠉⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⢶⣤⣄⣶⠀⠀⠀⠀⠀⢀⣤⣾⣶⣶⡶⢾⣷⢶⡟⡆⠀⠀⢰⣀⣤⣤⠄
                                                        ⠐⢛⣿⣿⡿⣤⣄⡀⠀⡀⢊⢼⢽⡿⢿⢹⠷⠶⠝⡷⠜⠄⣀⢸⣿⣿⡻⠃
                                                        ⠀⠻⠋⠀⠁⠒⠠⠭⢇⡐⠁⠘⠩⣬⠛⣶⣇⠐⣿⠁⠘⢮⣕⠜⠁⠀⠁⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠉⢁⡿⠛⢉⡄⡹⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⠦⠀⠀⡀⠀⢺⠳⠤⠄⢤⡀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢎⠜⠒⠉⠈⠁⠉⢉⣩⢏⣾⠇⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣋⣎⠀⠀⠀⠀⡤⣤⢟⠟⠁⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢴⠺⣄⠀⠀⠀⢸⡑⡁⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠡⣙⠢⡄⠀⠀⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⢗⢻⠝⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀
                                                        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⠮⠕⠁⠀ 
                                                        
                                            
    rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)

    #   Greedy-epsilon action selection
    def select_action(self, state, policy_net):
        if np.random.random_sample() <= self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            action = self.optimal_action(state)
        return action


    def update(self, state, next_state, reward, action, next_action):
        predict = self.optimal_action(state)
        expected_q = self.Q(next_state)
        q_max = self.optimal_action(next_state)
    """
    