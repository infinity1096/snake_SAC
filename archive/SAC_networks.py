
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ValueNetwork(nn.Module):

    def __init__(self, state_dim, hidden_dim, w_init=3e-3):

        super().__init__()

        self.fully_connected_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):

        x = self.fully_connected_net(state)

        return x

class SoftQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, w_init=3e-3):

        super().__init__()

        self.action_dim = action_dim

        linear_in = state_dim + action_dim

        self.fully_connected_net = nn.Sequential(
            nn.Linear(linear_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        

    def forward(self, state, action):

        action_one_hot = F.one_hot(action, self.action_dim)

        x = torch.cat([state, action_one_hot],1)
        
        x = self.fully_connected_net(x)

        return x

class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim, w_init=3e-3, log_prob_min=-20, log_prob_max=0):

        super().__init__()

        self.log_prob_min = log_prob_min
        self.log_prob_max = log_prob_max

        self.fully_connected_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
       
        log_prob = self.fully_connected_net(state)

        # limit log_std range
        log_prob = torch.clamp(log_prob, self.log_prob_min, self.log_prob_max)

        probs = torch.softmax(log_prob, dim=1)

        return probs

    def evaluate(self, state, epsilon=1e-6):
        
        probs = self.forward(state)

        categorical = Categorical(probs)
        
        action = categorical.sample().to(device)

        log_prob = categorical.log_prob(action)

        log_probs = probs.log()

        return action, log_prob, log_probs

    def sample_action(self, state):
        probs = self.forward(state)
        
        categorical = Categorical(probs)
        
        action = categorical.sample().to(device)

        return action.cpu()[0]



