import cv2
import gc
import gym
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

#CUSTOM IMPORTS
from wrappers import *
from nets import *
from replayBuffer import Replay_Buffer
from plotting import plot


cv2.ocl.setUseOpenCL(False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilon_S = 1.0
epsilon_E = 0.01
epsilon_decay = 30000

_epsilon = lambda frame: epsilon_E + (epsilon_S - epsilon_E)*np.exp(-frame/epsilon_decay)

#Propably need to change this
def make_atari(env_id, render_mode=None):
    env = gym.make(env_id, render_mode = render_mode)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def wrap_pytorch(env):
    return ImageToPyTorch(env)


env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id, render_mode='rgb_array')
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)



def compute_td_loss(batch_size, device):
    state, action, reward, next_state, done = replay_buffer.replay(batch_size)
    state      = torch.tensor(state).to(device)
    next_state = torch.tensor(np.array(next_state), requires_grad=False).to(device)
    action     = torch.LongTensor(action).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    action_probs = F.softmax(next_q_values, dim=1)
    expected_q_values = torch.sum(action_probs * next_q_values, dim=1) #Removed subtraction. Redundant?

    target_q_values = reward + gamma * expected_q_values * (1 - done)
    loss = (q_value - target_q_values.data).pow(2).mean()
    
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

model = CnnDQN(env.observation_space.shape, env.action_space.n, device)

if device == 'cuda':
    model = model.cuda()
else:
    model = model.cpu()
    
optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = Replay_Buffer(100000)

num_frames = 2000000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0
step = 0
ep = 0
max_steps = 0

state, _ = env.reset()
for frame_idx in tqdm(range(1, num_frames + 1)):

    epsilon = _epsilon(frame_idx)
    action = model.act(state, epsilon) ####
    
    result = env.step(action)
 
   
    next_state, reward, terminated, truncated, info = result
    done = terminated or truncated

    if len(result) == 4:
        raise ValueError("Unexpected tuple length returned from env.step(action)")

    # Store in replay buffer
    replay_buffer.store(state, action, reward, next_state, float(done))
    
    state = next_state
    episode_reward += reward
    step += 1
    
    if done:
        state, _ = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        max_steps = max(step, max_steps)
        step = 0
        ep += 1
        gc.collect()

    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size, device)
        losses.append(loss)
      
    if frame_idx % 100000 == 0:
        rgb_array = env.render()
        plot(frame_idx, all_rewards, losses, rgb_array, (step, ep, max_steps))