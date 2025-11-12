"""
Replay Buffers
Experience replay utilities for RL training.
Supports standard and prioritized experience replay.
"""

import numpy as np
import torch
import logging
import random
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from collections import deque
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None
    advantage: Optional[float] = None


class ExperienceBuffer:
    """
    Experience buffer for PPO rollout storage.
    Optimized for on-policy learning with GAE calculation.
    """

    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.logger = logging.getLogger(__name__)

        # Storage arrays
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

        # Computed advantages and returns
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        # Buffer state
        self.size = 0
        self.position = 0

        self.logger.info(f"Experience Buffer initialized: capacity {capacity:,}")

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        """Add experience to buffer."""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer."""
        if self.size < batch_size:
            batch_size = self.size

        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            "observations": torch.FloatTensor(self.observations[indices]),
            "actions": torch.FloatTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "next_observations": torch.FloatTensor(self.next_observations[indices]),
            "dones": torch.BoolTensor(self.dones[indices]),
            "log_probs": torch.FloatTensor(self.log_probs[indices]),
            "values": torch.FloatTensor(self.values[indices]),
            "advantages": torch.FloatTensor(self.advantages[indices]),
            "returns": torch.FloatTensor(self.returns[indices]),
        }

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """Get all buffer data."""
        return {
            "observations": self.observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
            "log_probs": self.log_probs[: self.size].copy(),
            "values": self.values[: self.size].copy(),
            "advantages": self.advantages[: self.size].copy(),
            "returns": self.returns[: self.size].copy(),
        }

    def compute_gae(
        self,
        gae_lambda: float,
        discount_factor: float,
        next_value: float,
        next_done: bool,
    ):
        """Compute GAE advantages and returns."""
        advantages = np.zeros(self.size)
        returns = np.zeros(self.size)

        next_value = next_value if not next_done else 0.0
        next_advantage = 0.0

        # Backward computation
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]

            # TD error
            td_error = (
                self.rewards[t]
                + discount_factor * next_val * next_non_terminal
                - self.values[t]
            )

            # GAE
            advantages[t] = (
                td_error
                + discount_factor * gae_lambda * next_advantage * next_non_terminal
            )
            next_advantage = advantages[t]

            # Returns
            returns[t] = advantages[t] + self.values[t]

        self.advantages[: self.size] = advantages
        self.returns[: self.size] = returns

    def clear(self):
        """Clear buffer."""
        self.size = 0
        self.position = 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    For off-policy methods or advanced on-policy variants.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 1_000_000,
    ):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta_start = beta_start  # Initial importance sampling correction
        self.beta_steps = beta_steps

        # Storage
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Priority tracking
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # Buffer state
        self.size = 0
        self.position = 0
        self.total_steps = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Prioritized Replay Buffer initialized: {capacity:,} capacity"
        )

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ):
        """Add experience with maximum priority."""
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done

        # New experiences get maximum priority
        self.priorities[self.position] = self.max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample prioritized batch.

        Returns:
            (batch_data, indices, importance_weights)
        """
        if self.size < batch_size:
            batch_size = self.size

        # Calculate sampling probabilities
        priorities = self.priorities[: self.size] ** self.alpha
        probabilities = priorities / np.sum(priorities)

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Calculate importance sampling weights
        current_beta = min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * self.total_steps / self.beta_steps,
        )
        weights = (self.size * probabilities[indices]) ** (-current_beta)
        weights = weights / np.max(weights)  # Normalize

        batch_data = {
            "observations": torch.FloatTensor(self.observations[indices]),
            "actions": torch.FloatTensor(self.actions[indices]),
            "rewards": torch.FloatTensor(self.rewards[indices]),
            "next_observations": torch.FloatTensor(self.next_observations[indices]),
            "dones": torch.BoolTensor(self.dones[indices]),
        }

        return batch_data, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = np.abs(td_errors) + 1e-6  # Small epsilon to avoid zero priority
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))

    def increment_step(self):
        """Increment step counter for beta annealing."""
        self.total_steps += 1
