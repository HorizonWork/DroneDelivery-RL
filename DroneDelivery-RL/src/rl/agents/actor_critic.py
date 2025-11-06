"""
Actor-Critic network with [256,128,64] architecture and tanh activation
""",
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        shared = self.shared_layers(x)
        action = self.actor(shared)
        value = self.critic(shared)
        return action, value
