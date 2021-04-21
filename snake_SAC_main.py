#%% definition and initialization
import math
import random
from collections import namedtuple
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image

from snake_env.snake_game import snake_game, snake_game_easy
from snake_SAC_networks import PolicyNetwork, SoftQNetwork, ValueNetwork
from snake_SAC_utils import ReplayBuffer, plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = snake_game_easy(10)

# shape definition
action_dim = 4         
state_dim = env.observation_space.shape[0]   
hidden_dim = 256

#hyperparameters
learning_rate = 1e-5
replay_buffer_size = 1000000
batch_size = 16

max_frames = 40000
max_steps = 300

# networks
value_network = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_network = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_network_1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_network_2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

policy_network = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)

# sync value network parameters
for (target_param, param) in zip(target_value_network.parameters(), value_network.parameters()):
    target_param.data.copy_(param.data)

# loss function - except policy net
value_criterion = nn.MSELoss()
soft_q_1_criterion = nn.MSELoss()
soft_q_2_criterion = nn.MSELoss()

# optimizer
value_optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
soft_q_1_optimizer = optim.Adam(soft_q_network_1.parameters(), lr=learning_rate)
soft_q_2_optimizer = optim.Adam(soft_q_network_2.parameters(), lr=learning_rate)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

replay_buffer = ReplayBuffer(replay_buffer_size)

# viz
fig = plt.figure()
plt.show(block=False)
ax = plt.gca()

#%% definition of network update
def update(batch_size, gamma=0.99, soft_tau=1e-2):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device, dtype=torch.int64)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_1_value = soft_q_network_1(state, action)
    predicted_q_2_value = soft_q_network_2(state, action)
    predicted_value    = value_network(state)
    new_action, log_prob, log_probs = policy_network.evaluate(state)

    # train Q networks
    target_value = target_value_network(next_state)
    target_q_value = reward + (1 - done) * gamma * target_value

    q_value_loss_1 = soft_q_1_criterion(predicted_q_1_value, target_q_value.detach())
    q_value_loss_2 = soft_q_2_criterion(predicted_q_2_value, target_q_value.detach())

    soft_q_1_optimizer.zero_grad()
    q_value_loss_1.backward()
    soft_q_1_optimizer.step()

    soft_q_2_optimizer.zero_grad()
    q_value_loss_2.backward()
    soft_q_2_optimizer.step()

    # train value network
    predicted_q_value = torch.min(soft_q_network_1(state, new_action), soft_q_network_2(state, new_action))
    target_value = predicted_q_value - log_prob

    value_loss = value_criterion(predicted_value, target_value.detach())
    
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # train policy network
    policy_loss = (log_prob - predicted_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # update value function by exponentially moving average
    
    for (target_param, param) in zip(target_value_network.parameters(), value_network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

# %% main - training
rewards = []
frame_idx = 0

ax = plt.gca()

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        if frame_idx >1000:
            action = policy_network.sample_action(
                torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            ).detach().item()
            next_state, reward, done, _ = env.step(action)
        else:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
        
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1
        
        if len(replay_buffer) > batch_size:
            update(batch_size)
        
        if frame_idx % 1000 == 0:
            plot(ax, frame_idx, rewards)
            plt.show(block=False)
            plt.pause(0.1)
            print(frame_idx)

        if done:
            break
        
    rewards.append(episode_reward)

# %%
plot(ax, frame_idx, rewards)
plt.show()

torch.save(policy_network, "trained_networks/policy_network.p")

# %% render animation

state = env.reset()
frame_idx = 0

env.render()
plt.pause(2)

while frame_idx < 1000:
    action = policy_network.sample_action(state).detach()
    next_state, reward, done, _ = env.step(action.numpy())

    env.render()

    state = next_state

    frame_idx += 1
# %%
