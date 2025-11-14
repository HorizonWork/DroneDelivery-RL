import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque

from src.rl.agents.actor_critic import ActorCriticNetwork
from src.rl.agents.gae_calculator import GAECalculator

dataclass
class PPOConfig:

    learning_rate: float = 3e-4
    rollout_length: int = 2048
    batch_size: int = 64
    epochs_per_update: int = 10
    clip_range: float = 0.2
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5

class RolloutBuffer:

    def __init__(self, rollout_length: int, observation_dim: int, action_dim: int):
        self.rollout_length = rollout_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.observations = np.zeros((rollout_length, observation_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=bool)

        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done):

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos = self.rollout_length:
            self.full = True
            self.pos = 0

    def get(self):

        return {
            'observations': self.observations.copy(),
            'actions': self.actions.copy(),
            'rewards': self.rewards.copy(),
            'values': self.values.copy(),
            'log_probs': self.log_probs.copy(),
            'dones': self.dones.copy(),
            'advantages': self.advantages.copy(),
            'returns': self.returns.copy()
        }

    def clear(self):

        self.pos = 0
        self.full = False

class PPOAgent:

    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.config = PPOConfig(config.get('ppo', {}))
        self.logger = logging.getLogger(__name__)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network_config = {
            'observation_dim': observation_dim,
            'action_dim': action_dim,
            'hidden_sizes': config.get('hidden_sizes', [256, 128, 64]),
            'activation': config.get('activation', 'tanh')
        }

        self.policy = ActorCriticNetwork(network_config).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

        self.gae_calculator = GAECalculator(self.config.gae_lambda, self.config.discount_factor)

        self.buffer = RolloutBuffer(self.config.rollout_length, observation_dim, action_dim)

        self.global_step = 0
        self.episode_count = 0
        self.total_environment_steps = 0

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = {
            'policy_loss': deque(maxlen=1000),
            'value_loss': deque(maxlen=1000),
            'entropy_loss': deque(maxlen=1000),
            'total_loss': deque(maxlen=1000)
        }

        self.learning_metrics = {
            'mean_reward': [],
            'mean_episode_length': [],
            'policy_loss': [],
            'value_loss': [],
            'explained_variance': []
        }

        self.logger.info("PPO Agent initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Network: {network_config['hidden_sizes']} hidden units")
        self.logger.info(f"Observation dim: {observation_dim}, Action dim: {action_dim}")
        self.logger.info(f"Hyperparameters: lr={self.config.learning_rate}, "
                        f"rollout={self.config.rollout_length}, "
                        f"batch={self.config.batch_size}")

    def select_action(self, observation: np.ndarray, deterministic: bool = False) - Tuple[np.ndarray, float]:

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

            action_dist, value = self.policy(obs_tensor)

            if deterministic:
                action = action_dist.mean
            else:
                action = action_dist.sample()

            log_prob = action_dist.log_prob(action)

            return action.cpu().numpy()[0], log_prob.cpu().item()

    def add_experience(self, observation: np.ndarray, action: np.ndarray,
                      reward: float, value: float, log_prob: float, done: bool):

        self.buffer.add(observation, action, reward, value, log_prob, done)
        self.total_environment_steps += 1

    def update_policy(self, next_observation: np.ndarray, next_done: bool) - Dict[str, float]:

        if not self.buffer.full and self.buffer.pos  self.config.rollout_length:
            return {}

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
            _, next_value = self.policy(obs_tensor)
            next_value = next_value.cpu().item()

        buffer_data = self.buffer.get()
        advantages, returns = self.gae_calculator.compute_gae(
            rewards=buffer_data['rewards'],
            values=buffer_data['values'],
            dones=buffer_data['dones'],
            next_value=next_value,
            next_done=next_done
        )

        self.buffer.advantages = advantages
        self.buffer.returns = returns

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        observations = torch.FloatTensor(buffer_data['observations']).to(self.device)
        actions = torch.FloatTensor(buffer_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer_data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        update_metrics = []

        for epoch in range(self.config.epochs_per_update):
            epoch_metrics = self._ppo_epoch_update(
                observations, actions, old_log_probs,
                advantages_tensor, returns_tensor
            )
            update_metrics.append(epoch_metrics)

        self.buffer.clear()

        aggregated_metrics = {}
        for key in update_metrics[0].keys():
            aggregated_metrics[key] = np.mean([m[key] for m in update_metrics])

        for key, value in aggregated_metrics.items():
            if key in self.training_losses:
                self.training_losses[key].append(value)

        self.global_step += 1

        return aggregated_metrics

    def _ppo_epoch_update(self, observations: torch.Tensor, actions: torch.Tensor,
                         old_log_probs: torch.Tensor, advantages: torch.Tensor,
                         returns: torch.Tensor) - Dict[str, float]:

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []

        batch_size = self.config.batch_size
        dataset_size = observations.shape[0]
        indices = torch.randperm(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            obs_batch = observations[batch_indices]
            actions_batch = actions[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            advantages_batch = advantages[batch_indices]
            returns_batch = returns[batch_indices]

            action_dist, values = self.policy(obs_batch)

            new_log_probs = action_dist.log_prob(actions_batch)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)

            surr1 = ratio  advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)  advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.MSELoss()(values.squeeze(-1), returns_batch)

            entropy = action_dist.entropy().mean()
            entropy_loss = -self.config.entropy_coefficient  entropy

            total_loss = policy_loss + self.config.value_loss_coefficient  value_loss + entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_entropy_losses.append(entropy_loss.item())

        return {
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'entropy_loss': np.mean(epoch_entropy_losses),
            'total_loss': np.mean(epoch_policy_losses) +
                         self.config.value_loss_coefficient  np.mean(epoch_value_losses) +
                         np.mean(epoch_entropy_losses)
        }

    def evaluate_observation(self, observation: np.ndarray) - float:

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            _, value = self.policy(obs_tensor)
            return value.cpu().item()

    def train_episode(self, episode_reward: float, episode_length: int):

        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        if len(self.episode_rewards) = 10:
            self.learning_metrics['mean_reward'].append(np.mean(list(self.episode_rewards)[-10:]))
            self.learning_metrics['mean_episode_length'].append(np.mean(list(self.episode_lengths)[-10:]))

    def save_model(self, filepath: str):

        save_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'total_environment_steps': self.total_environment_steps,
            'learning_metrics': self.learning_metrics
        }

        torch.save(save_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):

        try:
            save_data = torch.load(filepath, map_location=self.device)

            self.policy.load_state_dict(save_data['policy_state_dict'])
            self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
            self.global_step = save_data.get('global_step', 0)
            self.episode_count = save_data.get('episode_count', 0)
            self.total_environment_steps = save_data.get('total_environment_steps', 0)
            self.learning_metrics = save_data.get('learning_metrics', self.learning_metrics)

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def get_training_statistics(self) - Dict[str, Any]:

        stats = {
            'training_progress': {
                'global_step': self.global_step,
                'episode_count': self.episode_count,
                'total_environment_steps': self.total_environment_steps,
                'steps_per_episode': self.total_environment_steps / max(1, self.episode_count)
            },
            'performance': {
                'mean_episode_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                'std_episode_reward': np.std(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                'mean_episode_length': np.mean(list(self.episode_lengths)) if self.episode_lengths else 0.0,
                'recent_rewards': list(self.episode_rewards)[-10:] if len(self.episode_rewards) = 10 else list(self.episode_rewards)
            },
            'losses': {
                'recent_policy_loss': np.mean(list(self.training_losses['policy_loss'])[-10:]) if self.training_losses['policy_loss'] else 0.0,
                'recent_value_loss': np.mean(list(self.training_losses['value_loss'])[-10:]) if self.training_losses['value_loss'] else 0.0,
                'recent_entropy_loss': np.mean(list(self.training_losses['entropy_loss'])[-10:]) if self.training_losses['entropy_loss'] else 0.0
            },
            'hyperparameters': {
                'learning_rate': self.config.learning_rate,
                'rollout_length': self.config.rollout_length,
                'batch_size': self.config.batch_size,
                'clip_range': self.config.clip_range,
                'discount_factor': self.config.discount_factor,
                'gae_lambda': self.config.gae_lambda
            }
        }

        return stats

    def set_learning_rate(self, new_lr: float):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.config.learning_rate = new_lr
        self.logger.info(f"Learning rate updated to {new_lr}")

    def get_action_statistics(self, actions: np.ndarray) - Dict[str, Any]:

        if len(actions.shape) != 2 or actions.shape[1] != 4:
            return {'error': 'Invalid action shape'}

        action_stats = {}
        action_names = ['vx', 'vy', 'vz', 'yaw_rate']

        for i, name in enumerate(action_names):
            action_stats[name] = {
                'mean': float(np.mean(actions[:, i])),
                'std': float(np.std(actions[:, i])),
                'min': float(np.min(actions[:, i])),
                'max': float(np.max(actions[:, i])),
                'abs_mean': float(np.mean(np.abs(actions[:, i])))
            }

        action_magnitudes = np.linalg.norm(actions[:, :3], axis=1)
        action_stats['overall'] = {
            'mean_magnitude': float(np.mean(action_magnitudes)),
            'max_magnitude': float(np.max(action_magnitudes)),
            'action_diversity': float(np.std(action_magnitudes))
        }

        return action_stats

    def is_ready_for_update(self) - bool:

        return self.buffer.full or self.buffer.pos = self.config.rollout_length

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        pass
