"""
Policy Networks
Dedicated policy network implementation for fine-grained control.
Supports both discrete and continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

class PolicyNetwork(nn.Module):
    """
    Policy network for continuous action spaces.
    Maps 35D observations to 4D continuous actions [vx, vy, vz, yaw_rate].
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(PolicyNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']    # 35D as per report
        self.action_dim = config['action_dim']              # 4D continuous
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])  # Table 2
        self.activation = config.get('activation', 'tanh')
        
        # Action space bounds
        self.action_bounds = config.get('action_bounds', {
            'velocity': [-5.0, 5.0],      # m/s
            'yaw_rate': [-2.0, 2.0]       # rad/s
        })
        
        # Network architecture
        layers = []
        input_dim = self.observation_dim
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'elu':
                layers.append(nn.ELU())
            
            input_dim = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Policy output layers
        self.action_mean = nn.Linear(self.hidden_sizes[-1], self.action_dim)
        
        # Learnable log standard deviation for each action dimension
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        # Action scaling parameters
        self.register_buffer('action_scale', torch.tensor([
            self.action_bounds['velocity'][1],  # vx scale
            self.action_bounds['velocity'][1],  # vy scale  
            self.action_bounds['velocity'][1],  # vz scale
            self.action_bounds['yaw_rate'][1]   # yaw_rate scale
        ], dtype=torch.float32))
        
        self.register_buffer('action_bias', torch.zeros(self.action_dim, dtype=torch.float32))
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Policy Network initialized: {self.hidden_sizes}")
        self.logger.info(f"Action bounds: {self.action_bounds}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        # Initialize final layer with smaller weights for stable learning
        nn.init.uniform_(self.action_mean.weight, -0.1, 0.1)
        nn.init.zeros_(self.action_mean.bias)
        
        # Initialize log_std to reasonable exploration values
        nn.init.constant_(self.log_std, -0.5)  # std ≈ 0.6
    
    def forward(self, observations: torch.Tensor) -> Normal:
        """
        Forward pass to get action distribution.
        
        Args:
            observations: Batch of observations [batch_size, observation_dim]
            
        Returns:
            Normal distribution for continuous actions
        """
        # Feature extraction
        features = self.backbone(observations)
        
        # Action mean
        action_mean_raw = self.action_mean(features)
        
        # Scale actions to proper bounds using tanh squashing
        action_mean = torch.tanh(action_mean_raw) * self.action_scale + self.action_bias
        
        # Standard deviation (learnable per dimension)
        action_std = torch.exp(self.log_std).expand_as(action_mean)
        
        # Create distribution
        action_distribution = Normal(action_mean, action_std)
        
        return action_distribution
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        Get action for single observation.
        
        Args:
            observation: Single observation
            deterministic: Use mean action if True
            
        Returns:
            (action, log_probability)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_dist = self.forward(obs_tensor)
            
            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.sample()
            
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def get_log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of actions under current policy.
        
        Args:
            observations: Observation batch
            actions: Action batch
            
        Returns:
            Log probabilities
        """
        action_dist = self.forward(observations)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        return log_probs
    
    def get_entropy(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get policy entropy for given observations.
        
        Args:
            observations: Observation batch
            
        Returns:
            Entropy values
        """
        action_dist = self.forward(observations)
        entropy = action_dist.entropy().sum(dim=-1)
        return entropy
    
    def get_action_bounds(self) -> Dict[str, List[float]]:
        """Get action space bounds."""
        return self.action_bounds
    
    def set_exploration_noise(self, noise_std: Union[float, List[float]]):
        """
        Set exploration noise level.
        
        Args:
            noise_std: Noise standard deviation (single value or per-dimension)
        """
        if isinstance(noise_std, (int, float)):
            self.log_std.data.fill_(np.log(noise_std))
        else:
            assert len(noise_std) == self.action_dim
            self.log_std.data = torch.tensor(np.log(noise_std), dtype=torch.float32)
        
        self.logger.info(f"Exploration noise updated: {torch.exp(self.log_std).detach().cpu().numpy()}")
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get policy network information."""
        return {
            'type': 'continuous_policy',
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'action_bounds': self.action_bounds,
            'current_std': torch.exp(self.log_std).detach().cpu().numpy().tolist(),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

class ValueNetwork(nn.Module):
    """
    Value network for state value estimation.
    Maps 35D observations to scalar value estimates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(ValueNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])  # Table 2
        self.activation = config.get('activation', 'tanh')
        
        # Network architecture (identical to policy network backbone)
        layers = []
        input_dim = self.observation_dim
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'elu':
                layers.append(nn.ELU())
            
            input_dim = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Value output
        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Value Network initialized: {self.hidden_sizes}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        # Initialize value head with smaller weights
        nn.init.uniform_(self.value_head.weight, -0.1, 0.1)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for value estimation.
        
        Args:
            observations: Batch of observations [batch_size, observation_dim]
            
        Returns:
            Value estimates [batch_size, 1]
        """
        features = self.backbone(observations)
        values = self.value_head(features)
        return values
    
    def get_value(self, observation: np.ndarray) -> float:
        """
        Get value estimate for single observation.
        
        Args:
            observation: Single observation
            
        Returns:
            Value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            value = self.forward(obs_tensor)
            return value.cpu().item()
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get value network information."""
        return {
            'type': 'value_network',
            'observation_dim': self.observation_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

class MultiActionPolicyNetwork(nn.Module):
    """
    Multi-head policy network for mixed discrete/continuous actions.
    Extension for more complex action spaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(MultiActionPolicyNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.continuous_action_dim = config.get('continuous_action_dim', 4)
        self.discrete_action_dims = config.get('discrete_action_dims', [])  # List of discrete action space sizes
        
        # Shared backbone
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        self.activation = config.get('activation', 'tanh')
        
        # Build shared layers
        layers = []
        input_dim = self.observation_dim
        
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_size
        
        self.shared_backbone = nn.Sequential(*layers)
        
        # Continuous action head
        if self.continuous_action_dim > 0:
            self.continuous_mean = nn.Linear(self.hidden_sizes[-1], self.continuous_action_dim)
            self.continuous_log_std = nn.Parameter(torch.zeros(self.continuous_action_dim))
        
        # Discrete action heads
        self.discrete_heads = nn.ModuleList()
        for discrete_dim in self.discrete_action_dims:
            self.discrete_heads.append(nn.Linear(self.hidden_sizes[-1], discrete_dim))
        
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multi-Action Policy: {self.continuous_action_dim}D continuous + {len(self.discrete_action_dims)} discrete")
    
    def _initialize_weights(self):
        """Initialize weights for all network components."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize output layers with smaller weights
        if hasattr(self, 'continuous_mean'):
            nn.init.uniform_(self.continuous_mean.weight, -0.1, 0.1)
        
        for discrete_head in self.discrete_heads:
            nn.init.uniform_(discrete_head.weight, -0.1, 0.1)
    
    def forward(self, observations: torch.Tensor) -> Tuple[Optional[Normal], List[Categorical]]:
        """
        Forward pass for multi-action policy.
        
        Args:
            observations: Observation batch
            
        Returns:
            (continuous_distribution, list_of_discrete_distributions)
        """
        features = self.shared_backbone(observations)
        
        # Continuous action distribution
        continuous_dist = None
        if self.continuous_action_dim > 0:
            continuous_mean = torch.tanh(self.continuous_mean(features))
            continuous_std = torch.exp(self.continuous_log_std).expand_as(continuous_mean)
            continuous_dist = Normal(continuous_mean, continuous_std)
        
        # Discrete action distributions
        discrete_dists = []
        for discrete_head in self.discrete_heads:
            discrete_logits = discrete_head(features)
            discrete_dists.append(Categorical(logits=discrete_logits))
        
        return continuous_dist, discrete_dists
