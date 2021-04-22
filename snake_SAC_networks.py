import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SoftQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, w_init=3e-3):

        super().__init__()

        self.action_dim = action_dim

        self.fully_connected_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

    def forward(self, state):
        '''
        Output the Q value for all actions as a vector. 
        '''
        
        x = self.fully_connected_net(state)

        return x

class PolicyNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim, w_init=3e-3, log_prob_min=-20, log_prob_max=0):

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

        lim = 100

        log_prob = lim * torch.tanh(log_prob / lim)

        probs = torch.softmax(log_prob, dim=1)

        return probs

    def sample_action(self, state):
        probs = self.forward(state)
        
        try:
            categorical = Categorical(probs)
        except:
            print(probs)
            categorical = Categorical(torch.ones(probs.shape) / (probs.shape[0] * probs.shape[1]))

        action = categorical.sample().to(device)

        return action.cpu()[0]