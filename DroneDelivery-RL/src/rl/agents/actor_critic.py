"""
Actor-Critic Network
Combined policy and value networks with shared feature extraction.
Architecture: 256→128→64 hidden layers with tanh activation (Table 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Shared feature extraction with separate policy and value heads.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(ActorCriticNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.action_dim = config['action_dim'] 
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])  # Table 2
        self.activation = config.get('activation', 'tanh')              # Table 2
        
        # Shared feature extractor
        layers = []
        input_dim = self.observation_dim
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head (Actor)
        self.policy_mean = nn.Linear(self.hidden_sizes[-1], self.action_dim)
        
        # Learnable standard deviation for exploration
        self.policy_log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Value head (Critic)
        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)
        
        # Initialize networks
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Actor-Critic Network initialized: {self.hidden_sizes}")
        self.logger.info(f"Observation dim: {self.observation_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Activation: {self.activation}")
    
    def _initialize_weights(self):
        """Initialize network weights using common practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for tanh, He initialization for ReLU
                if self.activation == 'tanh':
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                
                nn.init.zeros_(module.bias)
        
        # Initialize policy output with smaller values for stable learning
        nn.init.uniform_(self.policy_mean.weight, -0.1, 0.1)
        nn.init.zeros_(self.policy_mean.bias)
        
        # Initialize log_std to reasonable values
        nn.init.constant_(self.policy_log_std, -0.5)  # std ≈ 0.6
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through actor-critic network.
        
        Args:
            observations: Batch of observations [batch_size, observation_dim]
            
        Returns:
            (action_distribution, value_estimates) tuple
        """
        # Shared feature extraction
        features = self.shared_layers(observations)
        
        # Policy head - continuous action distribution
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Create normal distribution for continuous actions
        action_distribution = Normal(action_mean, action_std)
        
        # Value head
        values = self.value_head(features)
        
        return action_distribution, values
    
    def get_action_and_value(self, observations: torch.Tensor, 
                           actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values with log probabilities.
        
        Args:
            observations: Observation batch
            actions: Optional actions for log prob calculation
            
        Returns:
            (actions, log_probs, values) tuple
        """
        action_dist, values = self.forward(observations)
        
        if actions is None:
            # Sample new actions
            actions = action_dist.sample()
        
        log_probs = action_dist.log_prob(actions)
        
        # Sum log probs across action dimensions
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
        
        return actions, log_probs, values.squeeze(-1)
    
    def evaluate_actions(self, observations: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions under current policy.
        
        Args:
            observations: Observation batch
            actions: Action batch to evaluate
            
        Returns:
            (log_probs, values, entropy) tuple
        """
        action_dist, values = self.forward(observations)
        
        log_probs = action_dist.log_prob(actions)
        if len(log_probs.shape) > 1:
            log_probs = log_probs.sum(dim=-1)
        
        entropy = action_dist.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': {
                'observation_dim': self.observation_dim,
                'action_dim': self.action_dim,
                'hidden_sizes': self.hidden_sizes,
                'activation': self.activation
            },
            'parameters': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'policy_parameters': sum(p.numel() for p in self.policy_mean.parameters()) + self.policy_log_std.numel(),
                'value_parameters': sum(p.numel() for p in self.value_head.parameters()),
                'shared_parameters': sum(p.numel() for p in self.shared_layers.parameters())
            },
            'current_std': torch.exp(self.policy_log_std).detach().cpu().numpy().tolist()
        }
