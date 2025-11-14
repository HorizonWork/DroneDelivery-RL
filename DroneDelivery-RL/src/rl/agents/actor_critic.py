import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

dataclass
class NetworkConfig:

    observation_dim: int
    action_dim: int

    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    activation: str = 'tanh'

    init_log_std: float = -0.5

    init_method: str = 'xavier'

    hidden_sizes: Optional[List[int]] = None

    def __post_init__(self):

        if self.hidden_sizes is not None:
            self.hidden_dims = self.hidden_sizes

        if self.observation_dim = 0:
            raise ValueError(f"observation_dim must be  0, got {self.observation_dim}")
        if self.action_dim = 0:
            raise ValueError(f"action_dim must be  0, got {self.action_dim}")

        if self.activation not in ['tanh', 'relu']:
            raise ValueError(f"activation must be 'tanh' or 'relu', got {self.activation}")

        if not self.hidden_dims or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")

    def to_dict(self) - Dict[str, Any]:

        return {
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_dims,
            'activation': self.activation,
            'init_log_std': self.init_log_std,
            'init_method': self.init_method
        }

    classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) - 'NetworkConfig':

        return cls(
            observation_dim=config_dict['observation_dim'],
            action_dim=config_dict['action_dim'],
            hidden_dims=config_dict.get('hidden_sizes', config_dict.get('hidden_dims', [256, 128, 64])),
            activation=config_dict.get('activation', 'tanh'),
            init_log_std=config_dict.get('init_log_std', -0.5),
            init_method=config_dict.get('init_method', 'xavier')
        )

class ActorCriticNetwork(nn.Module):

    def __init__(self, config: NetworkConfig):

        super(ActorCriticNetwork, self).__init__()

        if isinstance(config, dict):
            config = NetworkConfig.from_dict(config)

        self.config = config
        self.observation_dim = config.observation_dim
        self.action_dim = config.action_dim
        self.hidden_sizes = config.hidden_dims
        self.activation = config.activation

        layers = []
        input_dim = self.observation_dim

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))

            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())

            input_dim = hidden_size

        self.shared_layers = nn.Sequential(layers)

        self.policy_mean = nn.Linear(self.hidden_sizes[-1], self.action_dim)

        self.policy_log_std = nn.Parameter(
            torch.ones(self.action_dim)  config.init_log_std
        )

        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)

        self._initialize_weights()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Actor-Critic Network initialized: {self.hidden_sizes}")
        self.logger.info(f"Observation dim: {self.observation_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Activation: {self.activation}")

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == 'tanh' or self.config.init_method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif self.config.init_method == 'orthogonal':
                    nn.init.orthogonal_(module.weight)

                nn.init.zeros_(module.bias)

        nn.init.uniform_(self.policy_mean.weight, -0.1, 0.1)
        nn.init.zeros_(self.policy_mean.bias)

    def forward(self, observations: torch.Tensor) - Tuple[torch.distributions.Distribution, torch.Tensor]:

        features = self.shared_layers(observations)

        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)

        action_distribution = Normal(action_mean, action_std)

        values = self.value_head(features)

        return action_distribution, values

    def get_action_and_value(self, observations: torch.Tensor,
                            actions: Optional[torch.Tensor] = None) - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action_dist, values = self.forward(observations)

        if actions is None:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        if len(log_probs.shape)  1:
            log_probs = log_probs.sum(dim=-1)

        return actions, log_probs, values.squeeze(-1)

    def evaluate_actions(self, observations: torch.Tensor,
                        actions: torch.Tensor) - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action_dist, values = self.forward(observations)

        log_probs = action_dist.log_prob(actions)
        if len(log_probs.shape)  1:
            log_probs = log_probs.sum(dim=-1)

        entropy = action_dist.entropy()
        if len(entropy.shape)  1:
            entropy = entropy.sum(dim=-1)

        return log_probs, values.squeeze(-1), entropy

    def get_network_info(self) - Dict[str, Any]:

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

__all__ = [
    'NetworkConfig',
    'ActorCriticNetwork'
]
