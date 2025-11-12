"""
Normalization Utilities
Observation and reward normalization for stable RL training.
Running statistics with exponential moving averages.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json


@dataclass
class NormalizationStats:
    """Statistics for normalization."""

    mean: np.ndarray
    variance: np.ndarray
    count: int

    def std(self) -> np.ndarray:
        """Get standard deviation."""
        return np.sqrt(self.variance)


class ObservationNormalizer:
    """
    Online observation normalization using running statistics.
    Maintains separate statistics for each observation dimension.
    """

    def __init__(self, observation_dim: int, config: Dict[str, Any]):
        self.observation_dim = observation_dim
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Normalization parameters
        self.epsilon = config.get("epsilon", 1e-8)  # Numerical stability
        self.momentum = config.get("momentum", 0.99)  # EMA momentum
        self.clip_range = config.get("clip_range", 10.0)  # Clip normalized values
        self.warm_up_steps = config.get(
            "warm_up_steps", 1000
        )  # Steps before normalization

        # Running statistics
        self.mean = np.zeros(observation_dim, dtype=np.float64)
        self.variance = np.ones(observation_dim, dtype=np.float64)
        self.count = 0

        # Normalization state
        self.normalization_enabled = config.get("enabled", True)
        self.update_stats = config.get("update_stats", True)

        self.logger.info(f"Observation Normalizer initialized: {observation_dim}D")
        self.logger.info(f"Momentum: {self.momentum}, Clip range: ±{self.clip_range}")

    def normalize(
        self, observations: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize observations using current statistics.

        Args:
            observations: Raw observations [batch_size, obs_dim] or [obs_dim]

        Returns:
            Normalized observations
        """
        if not self.normalization_enabled or self.count < self.warm_up_steps:
            return observations

        is_tensor = isinstance(observations, torch.Tensor)
        device = observations.device if is_tensor else None

        # Convert to numpy for processing
        obs_np = observations.detach().cpu().numpy() if is_tensor else observations

        # Handle single observation or batch
        original_shape = obs_np.shape
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        # Normalize
        std = np.sqrt(self.variance + self.epsilon)
        normalized = (obs_np - self.mean) / std

        # Clip to prevent extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        # Restore original shape
        normalized = normalized.reshape(original_shape)

        # Convert back to tensor if needed
        if is_tensor:
            normalized = torch.tensor(
                normalized, dtype=observations.dtype, device=device
            )

        return normalized

    def update_statistics(self, observations: Union[np.ndarray, torch.Tensor]):
        """
        Update running statistics with new observations.

        Args:
            observations: New observations for statistics update
        """
        if not self.update_stats:
            return

        # Convert to numpy
        obs_np = (
            observations.detach().cpu().numpy()
            if isinstance(observations, torch.Tensor)
            else observations
        )

        # Handle single observation or batch
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        batch_size = obs_np.shape[0]

        # Update statistics using online algorithm
        for obs in obs_np:
            self.count += 1

            # Online mean and variance update (Welford's algorithm adapted for EMA)
            if self.count == 1:
                self.mean = obs.copy()
                self.variance = np.ones_like(obs)
            else:
                # Exponential moving average
                delta = obs - self.mean
                self.mean += (1 - self.momentum) * delta
                self.variance = (
                    self.momentum * self.variance + (1 - self.momentum) * delta**2
                )

    def denormalize(
        self, normalized_observations: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize observations back to original scale.

        Args:
            normalized_observations: Normalized observations

        Returns:
            Denormalized observations
        """
        if not self.normalization_enabled or self.count < self.warm_up_steps:
            return normalized_observations

        is_tensor = isinstance(normalized_observations, torch.Tensor)
        device = normalized_observations.device if is_tensor else None

        # Convert to numpy
        norm_np = (
            normalized_observations.detach().cpu().numpy()
            if is_tensor
            else normalized_observations
        )
        original_shape = norm_np.shape

        if norm_np.ndim == 1:
            norm_np = norm_np.reshape(1, -1)

        # Denormalize
        std = np.sqrt(self.variance + self.epsilon)
        denormalized = norm_np * std + self.mean

        # Restore shape
        denormalized = denormalized.reshape(original_shape)

        # Convert back if needed
        if is_tensor:
            denormalized = torch.tensor(
                denormalized, dtype=normalized_observations.dtype, device=device
            )

        return denormalized

    def get_statistics(self) -> NormalizationStats:
        """Get current normalization statistics."""
        return NormalizationStats(
            mean=self.mean.copy(), variance=self.variance.copy(), count=self.count
        )

    def save_statistics(self, filepath: str):
        """Save normalization statistics."""
        stats_data = {
            "mean": self.mean.tolist(),
            "variance": self.variance.tolist(),
            "count": self.count,
            "config": self.config,
        }

        with open(filepath, "w") as f:
            json.dump(stats_data, f, indent=2)

        self.logger.info(f"Normalization statistics saved to {filepath}")

    def load_statistics(self, filepath: str):
        """Load normalization statistics."""
        try:
            with open(filepath, "r") as f:
                stats_data = json.load(f)

            self.mean = np.array(stats_data["mean"])
            self.variance = np.array(stats_data["variance"])
            self.count = stats_data["count"]

            self.logger.info(f"Normalization statistics loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load normalization statistics: {e}")


class RewardNormalizer:
    """
    Reward normalization using running mean and standard deviation.
    Helps stabilize value function learning.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Normalization parameters
        self.epsilon = config.get("epsilon", 1e-8)
        self.momentum = config.get("momentum", 0.99)
        self.clip_range = config.get("clip_range", 10.0)
        self.warm_up_episodes = config.get("warm_up_episodes", 100)

        # Running statistics
        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

        # Normalization control
        self.normalization_enabled = config.get("enabled", True)
        self.normalize_returns = config.get("normalize_returns", True)

        self.logger.info("Reward Normalizer initialized")

    def normalize_rewards(self, rewards: Union[List[float], np.ndarray]) -> np.ndarray:
        """Normalize reward sequence."""
        if not self.normalization_enabled or self.count < self.warm_up_episodes:
            return np.array(rewards)

        rewards_array = np.array(rewards)
        std = np.sqrt(self.variance + self.epsilon)
        normalized = (rewards_array - self.mean) / std

        # Clip extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def update_statistics(self, episode_rewards: List[float]):
        """Update statistics with episode rewards."""
        if not episode_rewards:
            return

        episode_return = sum(episode_rewards)
        self.count += 1

        # Update running statistics
        if self.count == 1:
            self.mean = episode_return
            self.variance = 1.0
        else:
            delta = episode_return - self.mean
            self.mean += (1 - self.momentum) * delta
            self.variance = (
                self.momentum * self.variance + (1 - self.momentum) * delta**2
            )
