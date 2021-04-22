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

from snake_env.snake_game import snake_game, snake_game_easy, snake_game_sparse
from snake_SAC_networks import PolicyNetwork, SoftQNetwork
from snake_SAC_utils import ReplayBuffer, plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = snake_game_sparse()

# shape definition
action_dim = 4    
state_dim = env.observation_space.shape[0]   
hidden_dim = 256

#hyperparameters
learning_rate = 4e-4
H_0 = 0.1
replay_buffer_size = 1000000
batch_size = 256

max_frames = 24000
max_steps = 400

# networks
soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

soft_q_net1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
soft_q_net2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

# temperature
alpha = torch.tensor([1.0], requires_grad=True, device=device)

# synchronize target and original
for (target_param, param) in zip(soft_q_net1_target.parameters(), soft_q_net1.parameters()):
    target_param.data.copy_(param.data)
for (target_param, param) in zip(soft_q_net2_target.parameters(), soft_q_net2.parameters()):
    target_param.data.copy_(param.data)

# loss functions
soft_q_criterion1 = nn.MSELoss()
soft_q_criterion2 = nn.MSELoss()

# optimizers
soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=learning_rate)
soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=learning_rate)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
temperature_optimizer = optim.Adam([alpha], lr=learning_rate)

# replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)

# visualization

#%% temp: fill the replay buffer with something
state = env.reset()
done = False
for i in range(100):
    if (done):
        env.reset()
    
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state

# visualization
fig = plt.figure()
plt.show(block=False)
ax = plt.gca()

#%% definition of network update
def update(batch_size, gamma=0.99, soft_tau=1e-2):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device, dtype=torch.int64)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(np.float32(done)).to(device)

    ### TODO: Not very sure of below:
    # update Q networks
    next_state_value_estimate1 = torch.sum(
        policy_net(next_state) * (soft_q_net1_target(next_state) - alpha * policy_net(next_state).log())
        , axis=1)
    next_state_value_estimate2 = torch.sum(
        policy_net(next_state) * (soft_q_net2_target(next_state) - alpha * policy_net(next_state).log())
        , axis=1)

    predicted_q_1_value = reward + (1 - done) * gamma * next_state_value_estimate1
    predicted_q_2_value = reward + (1 - done) * gamma * next_state_value_estimate2

    soft_q_value1 = soft_q_net1(state).gather(1, action.view(-1,1)).squeeze(1)
    soft_q_value2 = soft_q_net2(state).gather(1, action.view(-1,1)).squeeze(1)

    soft_q_loss_1 = soft_q_criterion1(soft_q_value1, predicted_q_1_value)
    soft_q_loss_2 = soft_q_criterion2(soft_q_value2, predicted_q_2_value)

    soft_q_optimizer1.zero_grad()
    soft_q_loss_1.backward()
    soft_q_optimizer1.step()

    soft_q_optimizer2.zero_grad()
    soft_q_loss_2.backward()
    soft_q_optimizer2.step()

    # update policy network
    state_q_estimate = torch.min(soft_q_net1(state), soft_q_net2(state))
    policy_loss_batch = torch.sum(policy_net(state) * (alpha * policy_net(state).log() - state_q_estimate), axis=1)
    policy_loss = torch.mean(policy_loss_batch)

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # update temperature TODO: 
    #alpha_loss = torch.mean(torch.sum(policy_net(state) * - (alpha * policy_net(state).log() + H_0), axis=1))
    #alpha_loss.backward()
    #temperature_optimizer.step()


    # exponentially sync Q targets and Q
    for (target_param, param) in zip(soft_q_net1_target.parameters(), soft_q_net1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
    for (target_param, param) in zip(soft_q_net2_target.parameters(), soft_q_net2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

#%% main training loop 
rewards = []
frame_idx = 0

ax = plt.gca()

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        if frame_idx > 500:
            action = policy_net.sample_action(
                torch.FloatTensor([state]).to(device)
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
        
        if frame_idx % 200 == 0:
            plot(ax, frame_idx, rewards)
            plt.show(block=False)
            plt.pause(0.1)
            print(frame_idx, alpha)
            print(policy_net( torch.FloatTensor([state]).to(device)))

        if done:
            break
        
    rewards.append(episode_reward)

#%% play animation
plot(ax, frame_idx, rewards)
plt.show()

torch.save(policy_net, "trained_networks/policy_network.p")

# %% render animation

state = env.reset()
frame_idx = 0

env.render()
plt.pause(2)

while frame_idx < 1000:
    action = policy_net.sample_action(
                torch.FloatTensor([state]).to(device)
            ).detach().item()
    next_state, reward, done, _ = env.step(action)

    env.render(block=False)
    plt.pause(0.1)

    state = next_state

    frame_idx += 1

    if (done):
        break