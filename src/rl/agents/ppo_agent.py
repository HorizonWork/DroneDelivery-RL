"""
PPO Agent
Main Proximal Policy Optimization agent for energy-aware drone control.
Implements exact specifications from Table 2 in report.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque

# ABSOLUTE IMPORTS - Simple and clear
from src.rl.agents.actor_critic import ActorCriticNetwork
from src.rl.agents.gae_calculator import GAECalculator

@dataclass
class PPOConfig:
    """PPO Configuration"""
    learning_rate: float = 0.0003
    rollout_length: int = 2048
    batch_size: int = 64
    epochs_per_update: int = 10  # ĐÚNG
    discount_factor: float = 0.99  # ĐÚNG
    gae_lambda: float = 0.95
    clip_range: float = 0.2  # ĐÚNG
    value_loss_coefficient: float = 0.5  # ĐÚNG
    entropy_coefficient: float = 0.01  # ĐÚNG
    max_grad_norm: float = 0.5

class RolloutBuffer:
    """Buffer for storing rollout data."""
    def __init__(self, rollout_length: int, observation_dim: int, action_dim: int):
        self.rollout_length = rollout_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Storage arrays
        self.observations = np.zeros((rollout_length, observation_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=bool)
        
        # Computed during GAE calculation
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add experience to buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos >= self.rollout_length:
            self.full = True
            self.pos = 0
    
    def get(self):
        """Get all buffer data."""
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
        """Clear buffer."""
        self.pos = 0
        self.full = False

class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    Implements energy-aware control policy with exact Table 2 hyperparameters.
    """
    
    def __init__(self, observation_dim: int, action_dim: int, config: Dict[str, Any]):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        ppo_dict = config.get('ppo', {})
        ppo_config_dict = {
            'learning_rate': ppo_dict.get('learning_rate', ppo_dict.get('lr', 0.0003)),
            'rollout_length': ppo_dict.get('rollout_length', 2048),
            'batch_size': ppo_dict.get('batch_size', 64),
            'epochs_per_update': ppo_dict.get('epochs_per_update', ppo_dict.get('epochs', 10)),
            'discount_factor': ppo_dict.get('discount_factor', ppo_dict.get('gamma', 0.99)),
            'gae_lambda': ppo_dict.get('gae_lambda', 0.95),
            'clip_range': ppo_dict.get('clip_range', ppo_dict.get('clip_epsilon', 0.2)),
            'value_loss_coefficient': ppo_dict.get('value_loss_coefficient', ppo_dict.get('value_loss_coef', 0.5)),
            'entropy_coefficient': ppo_dict.get('entropy_coefficient', ppo_dict.get('entropy_coef', 0.01)),
            'max_grad_norm': ppo_dict.get('max_grad_norm', 0.5),
        }
        self.config = PPOConfig(**ppo_config_dict)
        self.logger = logging.getLogger(__name__)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture (Table 2: Hidden layer sizes 256, 128, 64)
        network_config = {
            'observation_dim': observation_dim,
            'action_dim': action_dim,
            'hidden_sizes': config.get('hidden_sizes', [256, 128, 64]),
            'activation': config.get('activation', 'tanh')
        }
        
        # Initialize actor-critic network
        self.policy = ActorCriticNetwork(network_config).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        
        # GAE calculator
        self.gae_calculator = GAECalculator(self.config.gae_lambda, self.config.discount_factor)
        
        # Rollout buffer
        self.buffer = RolloutBuffer(self.config.rollout_length, observation_dim, action_dim)
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.total_environment_steps = 0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_losses = {
            'policy_loss': deque(maxlen=1000),
            'value_loss': deque(maxlen=1000),
            'entropy_loss': deque(maxlen=1000),
            'total_loss': deque(maxlen=1000)
        }
        
        # Learning curves
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
    
    def select_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action for given observation.

        Args:
            observation: Environment observation [obs_dim]
            deterministic: If True, use mean action (no sampling)

        Returns:
            Tuple of (action, log_prob, value_estimate)
        """
        # Convert observation to tensor
        if not isinstance(observation, torch.Tensor):
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        else:
            obs_tensor = observation.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Forward pass through policy network
            action_dist, value = self.policy(obs_tensor)

            # Select action
            if deterministic:
                # Use mean for deterministic action
                action = action_dist.mean
            else:
                # Sample from distribution
                action = action_dist.sample()

            # Compute log probability
            log_prob = action_dist.log_prob(action)

            # Sum log probs across action dimensions (for continuous multi-dim actions)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)

        # Convert to numpy/scalar
        action_np = action.squeeze(0).cpu().numpy()
        log_prob_scalar = log_prob.item()
        value_scalar = value.item()

        return action_np, log_prob_scalar, value_scalar
    
    def add_experience(self, observation: np.ndarray, action: np.ndarray, 
                      reward: float, value: float, log_prob: float, done: bool):
        """
        Add experience to rollout buffer.
        
        Args:
            observation: Environment observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Episode termination flag
        """
        self.buffer.add(observation, action, reward, value, log_prob, done)
        self.total_environment_steps += 1
    
    def update_policy(self, next_observation: np.ndarray, next_done: bool) -> Dict[str, float]:
        """
        Perform PPO policy update.
        
        Args:
            next_observation: Final observation for bootstrap
            next_done: Final done flag
            
        Returns:
            Training metrics dictionary
        """
        if not self.buffer.full and self.buffer.pos < self.config.rollout_length:
            return {}  # Buffer not ready
        
        # Get final value for GAE calculation
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
            _, next_value = self.policy(obs_tensor)
            next_value = next_value.cpu().item()
        
        # Calculate advantages and returns using GAE
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
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        observations = torch.FloatTensor(buffer_data['observations']).to(self.device)
        actions = torch.FloatTensor(buffer_data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer_data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO updates
        update_metrics = []
        
        for epoch in range(self.config.epochs_per_update):
            epoch_metrics = self._ppo_epoch_update(
                observations, actions, old_log_probs, 
                advantages_tensor, returns_tensor
            )
            update_metrics.append(epoch_metrics)
        
        # Clear buffer
        self.buffer.clear()
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in update_metrics[0].keys():
            aggregated_metrics[key] = np.mean([m[key] for m in update_metrics])
        
        # Store training losses
        for key, value in aggregated_metrics.items():
            if key in self.training_losses:
                self.training_losses[key].append(value)
        
        self.global_step += 1
        
        return aggregated_metrics
    
    def _ppo_epoch_update(self, observations: torch.Tensor, actions: torch.Tensor,
                         old_log_probs: torch.Tensor, advantages: torch.Tensor,
                         returns: torch.Tensor) -> Dict[str, float]:
        """
        Single epoch of PPO updates.
        
        Args:
            observations: Observation batch
            actions: Action batch
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return targets
            
        Returns:
            Epoch metrics
        """
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropy_losses = []
        
        # Create mini-batches
        batch_size = self.config.batch_size
        dataset_size = observations.shape[0]
        indices = torch.randperm(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Mini-batch data
            obs_batch = observations[batch_indices]
            actions_batch = actions[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            advantages_batch = advantages[batch_indices]
            returns_batch = returns[batch_indices]
            
            # Forward pass
            action_dist, values = self.policy(obs_batch)
            
            # Calculate policy loss
            new_log_probs = action_dist.log_prob(actions_batch)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            
            # PPO clipped objective
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = nn.MSELoss()(values.squeeze(-1), returns_batch)
            
            # Entropy loss (for exploration)
            entropy = action_dist.entropy().mean()
            entropy_loss = -self.config.entropy_coefficient * entropy
            
            # Total loss
            total_loss = policy_loss + self.config.value_loss_coefficient * value_loss + entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # Store losses
            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())
            epoch_entropy_losses.append(entropy_loss.item())
        
        return {
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses),
            'entropy_loss': np.mean(epoch_entropy_losses),
            'total_loss': np.mean(epoch_policy_losses) + 
                         self.config.value_loss_coefficient * np.mean(epoch_value_losses) + 
                         np.mean(epoch_entropy_losses)
        }
    
    def evaluate_observation(self, observation: np.ndarray) -> float:
        """
        Get value estimate for observation.
        
        Args:
            observation: 35D observation vector
            
        Returns:
            Value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            _, value = self.policy(obs_tensor)
            return value.cpu().item()
    

    def get_value(self, observation: np.ndarray) -> float:
        """
        Get value estimate for observation (alias for evaluate_observation).
        
        Args:
            observation: Observation array
            
        Returns:
            Value estimate
        """
        return self.evaluate_observation(observation)

    def train_episode(self, episode_reward: float, episode_length: int):
        """
        Record episode completion for training statistics.
        
        Args:
            episode_reward: Total episode reward
            episode_length: Episode length in steps
        """
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Update learning metrics
        if len(self.episode_rewards) >= 10:  # Wait for some episodes
            self.learning_metrics['mean_reward'].append(np.mean(list(self.episode_rewards)[-10:]))
            self.learning_metrics['mean_episode_length'].append(np.mean(list(self.episode_lengths)[-10:]))
    
    def save_model(self, filepath: str):
        """
        Save PPO model to file.
        
        Args:
            filepath: Model save path
        """
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
        """
        Load PPO model from file.
        
        Args:
            filepath: Model load path
        """
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
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
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
                'recent_rewards': list(self.episode_rewards)[-10:] if len(self.episode_rewards) >= 10 else list(self.episode_rewards)
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
        """Update learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.config.learning_rate = new_lr
        self.logger.info(f"Learning rate updated to {new_lr}")
    
    def get_action_statistics(self, actions: np.ndarray) -> Dict[str, Any]:
        """
        Analyze action distribution statistics.
        
        Args:
            actions: Batch of actions [N, 4]
            
        Returns:
            Action statistics
        """
        if len(actions.shape) != 2 or actions.shape[1] != 4:
            return {'error': 'Invalid action shape'}
        
        # Per-dimension statistics
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
        
        # Overall action magnitude
        action_magnitudes = np.linalg.norm(actions[:, :3], axis=1)  # Exclude yaw rate
        action_stats['overall'] = {
            'mean_magnitude': float(np.mean(action_magnitudes)),
            'max_magnitude': float(np.max(action_magnitudes)),
            'action_diversity': float(np.std(action_magnitudes))
        }
        
        return action_stats
    
    def is_ready_for_update(self) -> bool:
        """Check if ready for policy update."""
        return self.buffer.full or self.buffer.pos >= self.config.rollout_length
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Could save model checkpoint here
        pass
