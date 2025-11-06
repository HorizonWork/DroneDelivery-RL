"""
Value Networks
Dedicated value network implementations with advanced features.
Supports state values, Q-values, and energy-aware value estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

class ValueNetwork(nn.Module):
    """
    State value network V(s).
    Maps 35D observations to scalar value estimates for PPO.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(ValueNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])  # Table 2
        self.activation = config.get('activation', 'tanh')
        
        # Value function specific parameters
        self.value_clipping = config.get('value_clipping', False)
        self.value_clip_range = config.get('value_clip_range', 0.2)
        
        # Network layers
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
            
            # Optional batch normalization
            if config.get('use_batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Optional dropout for regularization
            dropout_rate = config.get('dropout_rate', 0.0)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Value output head
        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Value Network initialized: {self.hidden_sizes}")
        if self.value_clipping:
            self.logger.info(f"Value clipping enabled: ±{self.value_clip_range}")
    
    def _initialize_weights(self):
        """Initialize network weights for stable value learning."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal initialization for value networks
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        # Initialize value head with smaller scale
        nn.init.orthogonal_(self.value_head.weight, gain=0.1)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for value estimation.
        
        Args:
            observations: Observation batch [batch_size, observation_dim]
            
        Returns:
            Value estimates [batch_size, 1]
        """
        features = self.feature_extractor(observations)
        values = self.value_head(features)
        
        return values
    
    def compute_value_loss(self, predicted_values: torch.Tensor, target_returns: torch.Tensor,
                          old_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute value function loss with optional clipping.
        
        Args:
            predicted_values: Current value predictions
            target_returns: Target return values
            old_values: Previous value predictions for clipping
            
        Returns:
            Value loss tensor
        """
        if self.value_clipping and old_values is not None:
            # Clipped value loss (similar to PPO policy clipping)
            value_pred_clipped = old_values + torch.clamp(
                predicted_values - old_values,
                -self.value_clip_range,
                self.value_clip_range
            )
            
            value_loss_unclipped = (predicted_values - target_returns) ** 2
            value_loss_clipped = (value_pred_clipped - target_returns) ** 2
            
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            # Standard MSE loss
            value_loss = F.mse_loss(predicted_values.squeeze(-1), target_returns)
        
        return value_loss

class QValueNetwork(nn.Module):
    """
    Q-value network Q(s,a) for action-value estimation.
    Alternative to value network for Q-learning variants.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(QValueNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.action_dim = config['action_dim']
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        self.activation = config.get('activation', 'tanh')
        
        # Combined input (observation + action)
        input_dim = self.observation_dim + self.action_dim
        
        # Network layers
        layers = []
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            
            input_dim = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Q-value output
        self.q_head = nn.Linear(self.hidden_sizes[-1], 1)
        
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Q-Value Network initialized: {self.hidden_sizes}")
    
    def _initialize_weights(self):
        """Initialize Q-network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Initialize Q-head with smaller weights
        nn.init.uniform_(self.q_head.weight, -0.1, 0.1)
        nn.init.zeros_(self.q_head.bias)
    
    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Q-value estimation.
        
        Args:
            observations: Observation batch [batch_size, obs_dim]
            actions: Action batch [batch_size, action_dim]
            
        Returns:
            Q-value estimates [batch_size, 1]
        """
        # Concatenate observations and actions
        combined_input = torch.cat([observations, actions], dim=-1)
        
        # Extract features
        features = self.feature_extractor(combined_input)
        
        # Q-value output
        q_values = self.q_head(features)
        
        return q_values

class EnergyAwareValueNetwork(nn.Module):
    """
    Energy-aware value network.
    Explicitly models energy consumption in value estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(EnergyAwareValueNetwork, self).__init__()
        
        self.observation_dim = config['observation_dim']
        self.hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
        self.activation = config.get('activation', 'tanh')
        
        # Energy modeling parameters
        self.energy_weight = config.get('energy_weight', 0.1)
        self.base_energy_consumption = config.get('base_energy', 10.0)  # Watts
        
        # Shared feature extraction
        layers = []
        input_dim = self.observation_dim
        
        for hidden_size in self.hidden_sizes[:-1]:  # All but last layer
            layers.append(nn.Linear(input_dim, hidden_size))
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            input_dim = hidden_size
        
        self.shared_features = nn.Sequential(*layers)
        
        # Separate heads for value and energy estimation
        final_feature_dim = self.hidden_sizes[-1]
        
        # Standard value head
        self.value_layers = nn.Sequential(
            nn.Linear(input_dim, final_feature_dim),
            nn.Tanh() if self.activation == 'tanh' else nn.ReLU(),
            nn.Linear(final_feature_dim, 1)
        )
        
        # Energy consumption head
        self.energy_layers = nn.Sequential(
            nn.Linear(input_dim, final_feature_dim),
            nn.Tanh() if self.activation == 'tanh' else nn.ReLU(),
            nn.Linear(final_feature_dim, 1),
            nn.Softplus()  # Ensure positive energy values
        )
        
        self._initialize_weights()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Energy-Aware Value Network initialized")
        self.logger.info(f"Energy weight: {self.energy_weight}")
    
    def _initialize_weights(self):
        """Initialize weights for energy-aware network."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for energy-aware value estimation.
        
        Args:
            observations: Observation batch
            
        Returns:
            (value_estimates, energy_estimates) tuple
        """
        # Shared feature extraction
        shared_features = self.shared_features(observations)
        
        # Value and energy estimation
        values = self.value_layers(shared_features)
        energy_consumption = self.energy_layers(shared_features)
        
        return values, energy_consumption
    
    def get_combined_value(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get combined value incorporating energy costs.
        
        Args:
            observations: Observation batch
            
        Returns:
            Energy-aware value estimates
        """
        values, energy = self.forward(observations)
        
        # Combine value and energy cost
        combined_value = values - self.energy_weight * energy
        
        return combined_value
    
    def get_energy_prediction(self, observation: np.ndarray) -> float:
        """
        Get energy consumption prediction for observation.
        
        Args:
            observation: Single observation
            
        Returns:
            Predicted energy consumption (Watts)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            _, energy = self.forward(obs_tensor)
            return energy.cpu().item()
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get energy-aware network information."""
        return {
            'type': 'energy_aware_value',
            'observation_dim': self.observation_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'energy_weight': self.energy_weight,
            'base_energy_consumption': self.base_energy_consumption,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

# Network factory functions
def create_policy_network(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create policy network."""
    network_type = config.get('type', 'standard')
    
    if network_type == 'standard':
        return PolicyNetwork(config)
    elif network_type == 'multi_action':
        return MultiActionPolicyNetwork(config)
    else:
        raise ValueError(f"Unknown policy network type: {network_type}")

def create_value_network(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create value network."""
    network_type = config.get('type', 'standard')
    
    if network_type == 'standard':
        return ValueNetwork(config)
    elif network_type == 'q_value':
        return QValueNetwork(config)
    elif network_type == 'energy_aware':
        return EnergyAwareValueNetwork(config)
    else:
        raise ValueError(f"Unknown value network type: {network_type}")
