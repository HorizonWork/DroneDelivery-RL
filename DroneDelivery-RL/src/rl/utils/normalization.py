import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json

dataclass
class NormalizationStats:

    mean: np.ndarray
    variance: np.ndarray
    count: int

    def std(self) - np.ndarray:

        return np.sqrt(self.variance)

class ObservationNormalizer:

    def __init__(self, observation_dim: int, config: Dict[str, Any]):
        self.observation_dim = observation_dim
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.epsilon = config.get('epsilon', 1e-8)
        self.momentum = config.get('momentum', 0.99)
        self.clip_range = config.get('clip_range', 10.0)
        self.warm_up_steps = config.get('warm_up_steps', 1000)

        self.mean = np.zeros(observation_dim, dtype=np.float64)
        self.variance = np.ones(observation_dim, dtype=np.float64)
        self.count = 0

        self.normalization_enabled = config.get('enabled', True)
        self.update_stats = config.get('update_stats', True)

        self.logger.info(f"Observation Normalizer initialized: {observation_dim}D")
        self.logger.info(f"Momentum: {self.momentum}, Clip range: {self.clip_range}")

    def normalize(self, observations: Union[np.ndarray, torch.Tensor]) - Union[np.ndarray, torch.Tensor]:

        if not self.normalization_enabled or self.count  self.warm_up_steps:
            return observations

        is_tensor = isinstance(observations, torch.Tensor)
        device = observations.device if is_tensor else None

        obs_np = observations.detach().cpu().numpy() if is_tensor else observations

        original_shape = obs_np.shape
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        std = np.sqrt(self.variance + self.epsilon)
        normalized = (obs_np - self.mean) / std

        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        normalized = normalized.reshape(original_shape)

        if is_tensor:
            normalized = torch.tensor(normalized, dtype=observations.dtype, device=device)

        return normalized

    def update_statistics(self, observations: Union[np.ndarray, torch.Tensor]):

        if not self.update_stats:
            return

        obs_np = observations.detach().cpu().numpy() if isinstance(observations, torch.Tensor) else observations

        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)

        batch_size = obs_np.shape[0]

        for obs in obs_np:
            self.count += 1

            if self.count == 1:
                self.mean = obs.copy()
                self.variance = np.ones_like(obs)
            else:
                delta = obs - self.mean
                self.mean += (1 - self.momentum)  delta
                self.variance = self.momentum  self.variance + (1 - self.momentum)  delta  2

    def denormalize(self, normalized_observations: Union[np.ndarray, torch.Tensor]) - Union[np.ndarray, torch.Tensor]:

        if not self.normalization_enabled or self.count  self.warm_up_steps:
            return normalized_observations

        is_tensor = isinstance(normalized_observations, torch.Tensor)
        device = normalized_observations.device if is_tensor else None

        norm_np = normalized_observations.detach().cpu().numpy() if is_tensor else normalized_observations
        original_shape = norm_np.shape

        if norm_np.ndim == 1:
            norm_np = norm_np.reshape(1, -1)

        std = np.sqrt(self.variance + self.epsilon)
        denormalized = norm_np  std + self.mean

        denormalized = denormalized.reshape(original_shape)

        if is_tensor:
            denormalized = torch.tensor(denormalized, dtype=normalized_observations.dtype, device=device)

        return denormalized

    def get_statistics(self) - NormalizationStats:

        return NormalizationStats(
            mean=self.mean.copy(),
            variance=self.variance.copy(),
            count=self.count
        )

    def save_statistics(self, filepath: str):

        stats_data = {
            'mean': self.mean.tolist(),
            'variance': self.variance.tolist(),
            'count': self.count,
            'config': self.config
        }

        with open(filepath, 'w') as f:
            json.dump(stats_data, f, indent=2)

        self.logger.info(f"Normalization statistics saved to {filepath}")

    def load_statistics(self, filepath: str):

        try:
            with open(filepath, 'r') as f:
                stats_data = json.load(f)

            self.mean = np.array(stats_data['mean'])
            self.variance = np.array(stats_data['variance'])
            self.count = stats_data['count']

            self.logger.info(f"Normalization statistics loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load normalization statistics: {e}")

class RewardNormalizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.epsilon = config.get('epsilon', 1e-8)
        self.momentum = config.get('momentum', 0.99)
        self.clip_range = config.get('clip_range', 10.0)
        self.warm_up_episodes = config.get('warm_up_episodes', 100)

        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

        self.normalization_enabled = config.get('enabled', True)
        self.normalize_returns = config.get('normalize_returns', True)

        self.logger.info("Reward Normalizer initialized")

    def normalize_rewards(self, rewards: Union[List[float], np.ndarray]) - np.ndarray:

        if not self.normalization_enabled or self.count  self.warm_up_episodes:
            return np.array(rewards)

        rewards_array = np.array(rewards)
        std = np.sqrt(self.variance + self.epsilon)
        normalized = (rewards_array - self.mean) / std

        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def update_statistics(self, episode_rewards: List[float]):

        if not episode_rewards:
            return

        episode_return = sum(episode_rewards)
        self.count += 1

        if self.count == 1:
            self.mean = episode_return
            self.variance = 1.0
        else:
            delta = episode_return - self.mean
            self.mean += (1 - self.momentum)  delta
            self.variance = self.momentum  self.variance + (1 - self.momentum)  delta  2
