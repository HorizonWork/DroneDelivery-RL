import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, cast
from dataclasses import dataclass, field

dataclass
class NetworkConfig:

    observation_dim: int
    action_dim: int

    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    activation: str = "tanh"

    init_log_std: float = -0.5

    init_method: str = "xavier"

    hidden_sizes: Optional[List[int]] = None

    def __post_init__(self):

        if self.hidden_sizes is not None:
            self.hidden_dims = self.hidden_sizes

        if self.observation_dim = 0:
            raise ValueError(f"observation_dim must be  0, got {self.observation_dim}")
        if self.action_dim = 0:
            raise ValueError(f"action_dim must be  0, got {self.action_dim}")

        if self.activation not in ["tanh", "relu"]:
            raise ValueError(
                f"activation must be 'tanh' or 'relu', got {self.activation}"
            )

        if not self.hidden_dims or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims cannot be empty")

    def to_dict(self) - Dict[str, Any]:

        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_dims,
            "activation": self.activation,
            "init_log_std": self.init_log_std,
            "init_method": self.init_method,
        }

    classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) - "NetworkConfig":

        return cls(
            observation_dim=config_dict["observation_dim"],
            action_dim=config_dict["action_dim"],
            hidden_dims=config_dict.get(
                "hidden_sizes", config_dict.get("hidden_dims", [256, 128, 64])
            ),
            activation=config_dict.get("activation", "tanh"),
            init_log_std=config_dict.get("init_log_std", -0.5),
            init_method=config_dict.get("init_method", "xavier"),
        )

class ActorCriticNetwork(nn.Module):

    def __init__(self, config: Union[Dict[str, Any], NetworkConfig]):

        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.config = self._normalize_config(config)
        self.config_dict = self.config.to_dict()

        self.observation_dim = self.config.observation_dim
        self.action_dim = self.config.action_dim
        self.obs_dim = self.observation_dim
        self.act_dim = self.action_dim
        self.hidden_sizes = list(self.config.hidden_dims)
        self.activation = self.config.activation.lower()
        self.init_log_std = float(self.config.init_log_std)
        self.init_method = self.config.init_method.lower()

        use_layer_norm = self.config_dict.get("use_layer_norm", True)

        layers = []
        prev_dim = self.observation_dim

        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self._get_activation(self.activation))
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(layers)

        self.policy_mean = nn.Linear(prev_dim, self.action_dim)
        init_log_std_tensor = torch.full(
            (self.action_dim,), self.init_log_std, dtype=torch.float32
        )
        self.policy_log_std = nn.Parameter(init_log_std_tensor)

        self.value_head = nn.Linear(prev_dim, 1)

        self._initialize_weights()

        self.logger.info(
            f"ActorCritic initialized: {self.observation_dim}D obs - {self.action_dim}D act"
        )

    staticmethod
    def _normalize_config(
        config: Union[Dict[str, Any], NetworkConfig],
    ) - NetworkConfig:

        if isinstance(config, NetworkConfig):
            return config

        cfg = dict(config or {})
        observation_dim = cfg.get("observation_dim", cfg.get("obs_dim", 40))
        action_dim = cfg.get("action_dim", cfg.get("act_dim", 4))
        hidden_dims = cfg.get("hidden_dims", cfg.get("hidden_sizes", [256, 128, 64]))
        hidden_dims = [int(h) for h in hidden_dims]
        activation = cfg.get("activation", "tanh")
        init_log_std = cfg.get("init_log_std", cfg.get("log_std", -0.5))
        init_method = cfg.get("init_method", cfg.get("weight_init", "xavier"))

        return NetworkConfig(
            observation_dim=int(observation_dim),
            action_dim=int(action_dim),
            hidden_dims=hidden_dims,
            activation=activation,
            init_log_std=float(init_log_std),
            init_method=init_method,
        )

    def _initialize_weights(self):

        gain = cast(float, nn.init.calculate_gain(self.activation))
        init_method = self.init_method

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(module.weight, gain=gain)
                elif init_method == "kaiming":
                    nn.init.kaiming_uniform_(
                        module.weight, nonlinearity=self.activation
                    )
                else:
                    nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        nn.init.orthogonal_(self.policy_mean.weight, gain=cast(float, 0.01))
        nn.init.constant_(self.policy_mean.bias, 0.0)

    def _get_activation(self, activation: str) - nn.Module:

        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def forward(
        self, observations: torch.Tensor
    ) - Tuple[torch.distributions.Distribution, torch.Tensor]:

        features = self.shared_layers(observations)

        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)

        action_distribution = Normal(action_mean, action_std)

        values = self.value_head(features)

        return action_distribution, values

    def get_action_and_value(
        self, observations: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        action_dist, values = self.forward(observations)

        if actions is None:
            actions = action_dist.sample()

        log_probs = action_dist.log_prob(actions)

        if len(log_probs.shape)  1:
            log_probs = log_probs.sum(dim=-1)

        return actions, log_probs, values.squeeze(-1)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

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
            "architecture": {
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": self.hidden_sizes,
                "activation": self.activation,
            },
            "parameters": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "policy_parameters": sum(
                    p.numel() for p in self.policy_mean.parameters()
                )
                + self.policy_log_std.numel(),
                "value_parameters": sum(
                    p.numel() for p in self.value_head.parameters()
                ),
                "shared_parameters": sum(
                    p.numel() for p in self.shared_layers.parameters()
                ),
            },
            "current_std": torch.exp(self.policy_log_std)
            .detach()
            .cpu()
            .numpy()
            .tolist(),
        }

__all__ = ["NetworkConfig", "ActorCriticNetwork"]
