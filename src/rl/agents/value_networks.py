import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

class ValueNetwork(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(ValueNetwork, self).__init__()

        self.observation_dim = config["observation_dim"]
        self.hidden_sizes = config.get("hidden_sizes", [256, 128, 64])
        self.activation = config.get("activation", "tanh")

        self.value_clipping = config.get("value_clipping", False)
        self.value_clip_range = config.get("value_clip_range", 0.2)

        layers = []
        input_dim = self.observation_dim

        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))

            if self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "elu":
                layers.append(nn.ELU())

            if config.get("use_batch_norm", False):
                layers.append(nn.BatchNorm1d(hidden_size))

            dropout_rate = config.get("dropout_rate", 0.0)
            if dropout_rate  0:
                layers.append(nn.Dropout(dropout_rate))

            input_dim = hidden_size

        self.feature_extractor = nn.Sequential(layers)

        self.value_head = nn.Linear(self.hidden_sizes[-1], 1)

        self._initialize_weights()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Value Network initialized: {self.hidden_sizes}")
        if self.value_clipping:
            self.logger.info(f"Value clipping enabled: {self.value_clip_range}")

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=0.1)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, observations: torch.Tensor) - torch.Tensor:

        features = self.feature_extractor(observations)
        values = self.value_head(features)

        return values

    def compute_value_loss(
        self,
        predicted_values: torch.Tensor,
        target_returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
    ) - torch.Tensor:

        if self.value_clipping and old_values is not None:
            value_pred_clipped = old_values + torch.clamp(
                predicted_values - old_values,
                -self.value_clip_range,
                self.value_clip_range,
            )

            value_loss_unclipped = (predicted_values - target_returns)  2
            value_loss_clipped = (value_pred_clipped - target_returns)  2

            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = F.mse_loss(predicted_values.squeeze(-1), target_returns)

        return value_loss

class QValueNetwork(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(QValueNetwork, self).__init__()

        self.observation_dim = config["observation_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_sizes = config.get("hidden_sizes", [256, 128, 64])
        self.activation = config.get("activation", "tanh")

        input_dim = self.observation_dim + self.action_dim

        layers = []
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))

            if self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "relu":
                layers.append(nn.ReLU())

            input_dim = hidden_size

        self.feature_extractor = nn.Sequential(layers)

        self.q_head = nn.Linear(self.hidden_sizes[-1], 1)

        self._initialize_weights()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Q-Value Network initialized: {self.hidden_sizes}")

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.uniform_(self.q_head.weight, -0.1, 0.1)
        nn.init.zeros_(self.q_head.bias)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) - torch.Tensor:

        combined_input = torch.cat([observations, actions], dim=-1)

        features = self.feature_extractor(combined_input)

        q_values = self.q_head(features)

        return q_values

class EnergyAwareValueNetwork(nn.Module):

    def __init__(self, config: Dict[str, Any]):
        super(EnergyAwareValueNetwork, self).__init__()

        self.observation_dim = config["observation_dim"]
        self.hidden_sizes = config.get("hidden_sizes", [256, 128, 64])
        self.activation = config.get("activation", "tanh")

        self.energy_weight = config.get("energy_weight", 0.1)
        self.base_energy_consumption = config.get("base_energy", 10.0)

        layers = []
        input_dim = self.observation_dim

        for hidden_size in self.hidden_sizes[:-1]:
            layers.append(nn.Linear(input_dim, hidden_size))
            if self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "relu":
                layers.append(nn.ReLU())
            input_dim = hidden_size

        self.shared_features = nn.Sequential(layers)

        final_feature_dim = self.hidden_sizes[-1]

        self.value_layers = nn.Sequential(
            nn.Linear(input_dim, final_feature_dim),
            nn.Tanh() if self.activation == "tanh" else nn.ReLU(),
            nn.Linear(final_feature_dim, 1),
        )

        self.energy_layers = nn.Sequential(
            nn.Linear(input_dim, final_feature_dim),
            nn.Tanh() if self.activation == "tanh" else nn.ReLU(),
            nn.Linear(final_feature_dim, 1),
            nn.Softplus(),
        )

        self._initialize_weights()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Energy-Aware Value Network initialized")
        self.logger.info(f"Energy weight: {self.energy_weight}")

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) - Tuple[torch.Tensor, torch.Tensor]:

        shared_features = self.shared_features(observations)

        values = self.value_layers(shared_features)
        energy_consumption = self.energy_layers(shared_features)

        return values, energy_consumption

    def get_combined_value(self, observations: torch.Tensor) - torch.Tensor:

        values, energy = self.forward(observations)

        combined_value = values - self.energy_weight  energy

        return combined_value

    def get_energy_prediction(self, observation: np.ndarray) - float:

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            _, energy = self.forward(obs_tensor)
            return energy.cpu().item()

    def get_network_info(self) - Dict[str, Any]:

        return {
            "type": "energy_aware_value",
            "observation_dim": self.observation_dim,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "energy_weight": self.energy_weight,
            "base_energy_consumption": self.base_energy_consumption,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

def create_policy_network(config: Dict[str, Any]) - nn.Module:

    network_type = config.get("type", "standard")

    if network_type == "standard":
        return PolicyNetwork(config)
    elif network_type == "multi_action":
        return MultiActionPolicyNetwork(config)
    else:
        raise ValueError(f"Unknown policy network type: {network_type}")

def create_value_network(config: Dict[str, Any]) - nn.Module:

    network_type = config.get("type", "standard")

    if network_type == "standard":
        return ValueNetwork(config)
    elif network_type == "q_value":
        return QValueNetwork(config)
    elif network_type == "energy_aware":
        return EnergyAwareValueNetwork(config)
    else:
        raise ValueError(f"Unknown value network type: {network_type}")
