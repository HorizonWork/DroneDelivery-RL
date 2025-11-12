"""
Main PPO Trainer
Core training loop with 5 million steps as specified in Section 5.2.
Implements exact training procedure from report.
"""

import torch
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import json

# Try wandb import (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - logging will be limited")

from src.rl.agents.ppo_agent import PPOAgent
from src.rl.evaluation.evaluator import DroneEvaluator


@dataclass
class TrainingConfig:
    """Training configuration matching Section 5.2."""

    total_timesteps: int = 5_000_000  # 5M steps as per report
    eval_frequency: int = 50_000  # Evaluate every 50k steps
    save_frequency: int = 100_000  # Save model every 100k steps
    log_frequency: int = 1000  # Log every 1k steps

    # Early stopping
    early_stopping_patience: int = 500_000  # Stop if no improvement for 500k steps
    target_success_rate: float = 0.96  # 96% success rate target

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "ppo_energy_aware"
    resume_from_checkpoint: bool = False
    checkpoint_path: str = ""


@dataclass
class TrainingState:
    """Current training state."""

    timestep: int = 0
    episode: int = 0
    best_success_rate: float = 0.0
    best_energy_efficiency: float = float("inf")
    steps_since_improvement: int = 0
    current_lr: float = 3e-4
    training_start_time: float = 0.0


class PPOTrainer:
    """
    Main PPO trainer implementing exact procedure from Section 5.2.
    Supports curriculum learning and comprehensive monitoring.
    """

    def __init__(self, agent: PPOAgent, environment, config: Dict[str, Any]):
        self.agent = agent
        self.environment = environment
        self.training_config = TrainingConfig(**config.get("training", {}))
        self.logger = logging.getLogger(__name__)

        # Training state
        self.state = TrainingState()

        # Evaluator for periodic assessment
        evaluator_config = config.get("evaluation", {})
        evaluator_config["num_episodes"] = 10  # Quick evaluation during training
        self.evaluator = DroneEvaluator(evaluator_config)

        # Monitoring
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE
        self.training_metrics = {
            "episode_rewards": deque(maxlen=100),
            "episode_lengths": deque(maxlen=100),
            "success_rates": deque(maxlen=50),
            "energy_consumptions": deque(maxlen=100),
            "policy_losses": deque(maxlen=1000),
            "value_losses": deque(maxlen=1000),
        }

        # Callbacks
        self.episode_callbacks: List[Callable] = []
        self.update_callbacks: List[Callable] = []

        # Setup directories
        self.checkpoint_dir = Path(self.training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        if self.use_wandb:
            self._initialize_wandb(config.get("wandb", {}))

        self.logger.info("PPO Trainer initialized")
        self.logger.info(f"Total timesteps: {self.training_config.total_timesteps:,}")
        self.logger.info(
            f"Target success rate: {self.training_config.target_success_rate}"
        )

    def _initialize_wandb(self, wandb_config: Dict[str, Any]):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=wandb_config.get("project", "drone-delivery-rl"),
                name=self.training_config.experiment_name,
                config={
                    "total_timesteps": self.training_config.total_timesteps,
                    "agent_config": (
                        self.agent.config.__dict__
                        if hasattr(self.agent.config, "__dict__")
                        else self.agent.config
                    ),
                    "training_config": self.training_config.__dict__,
                },
            )
            self.logger.info("Wandb logging initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False

    def train(self) -> Dict[str, Any]:
        """
        Execute complete training procedure.

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting PPO training")
        self.logger.info(f"Environment: {self.environment}")

        self.state.training_start_time = time.time()

        # Resume from checkpoint if specified
        if (
            self.training_config.resume_from_checkpoint
            and self.training_config.checkpoint_path
        ):
            self._load_checkpoint(self.training_config.checkpoint_path)

        # Training loop
        while self.state.timestep < self.training_config.total_timesteps:
            # Run episode
            episode_result = self._run_training_episode()

            # Update training state
            self.state.episode += 1
            self.state.timestep += episode_result["episode_length"]

            # Record metrics
            self._record_episode_metrics(episode_result)

            # Policy update when buffer is full
            if self.agent.is_ready_for_update():
                update_result = self._perform_policy_update()
                self._record_update_metrics(update_result)

            # Periodic evaluation
            if (
                self.state.timestep % self.training_config.eval_frequency == 0
                or self.state.timestep >= self.training_config.total_timesteps
            ):
                eval_result = self._periodic_evaluation()
                self._check_early_stopping(eval_result)

            # Periodic saving
            if self.state.timestep % self.training_config.save_frequency == 0:
                self._save_checkpoint()

            # Logging
            if self.state.episode % self.training_config.log_frequency == 0:
                self._log_training_progress()

            # Early stopping check
            if self._should_stop_early():
                self.logger.info("Early stopping triggered")
                break

        # Final evaluation
        final_evaluation = self._final_evaluation()

        # Save final model
        self._save_final_model()

        training_time = time.time() - self.state.training_start_time

        results = {
            "training_completed": True,
            "total_timesteps": self.state.timestep,
            "total_episodes": self.state.episode,
            "training_time": training_time,
            "final_evaluation": final_evaluation,
            "best_success_rate": self.state.best_success_rate,
            "best_energy_efficiency": self.state.best_energy_efficiency,
        }

        self.logger.info(f"Training completed in {training_time/3600:.1f} hours")
        self.logger.info(
            f"Final success rate: {final_evaluation.get('success_rate', 0):.1f}%"
        )
        self.logger.info(
            f"Final energy efficiency: {final_evaluation.get('energy_efficiency', 0):.0f}J"
        )

        if self.use_wandb:
            wandb.finish()

        return results

    def _run_training_episode(self) -> Dict[str, Any]:
        """
        Run single training episode.

        Returns:
            Episode result dictionary
        """
        # Reset environment
        observation = self.environment.reset()

        episode_reward = 0.0
        episode_length = 0
        episode_energy = 0.0
        collision_occurred = False

        done = False

        while not done:
            # Select action
            action, log_prob = self.agent.select_action(observation)
            value = self.agent.evaluate_observation(observation)

            # Execute action
            next_observation, reward, done, info = self.environment.step(action)

            # Record experience
            self.agent.add_experience(
                observation, action, reward, value, log_prob, done
            )

            # Update episode metrics
            episode_reward += reward
            episode_length += 1
            episode_energy += info.get("energy_consumption", 0.0)

            if info.get("collision", False):
                collision_occurred = True

            observation = next_observation

        # Track last observation for bootstrap value
        self.agent.last_observation = observation

        # Final position check
        final_position = info.get("position", (0, 0, 0))
        goal_position = info.get("goal_position", (0, 0, 0))
        final_distance = np.linalg.norm(
            np.array(final_position) - np.array(goal_position)
        )

        success = final_distance <= 0.5 and not collision_occurred  # 0.5m tolerance

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "episode_energy": episode_energy,
            "success": success,
            "collision": collision_occurred,
            "final_distance": final_distance,
            "info": info,
        }

    def _perform_policy_update(self) -> Dict[str, float]:
        """Perform PPO policy update."""
        # Get final observation for bootstrap
        final_obs = getattr(
            self.agent, "last_observation", np.zeros(self.agent.observation_dim)
        )
        final_done = False  # Episode ongoing

        # Update policy
        update_metrics = self.agent.update_policy(final_obs, final_done)

        return update_metrics

    def _periodic_evaluation(self) -> Dict[str, Any]:
        """Perform periodic evaluation during training."""
        self.logger.info(f"Evaluating at timestep {self.state.timestep:,}")

        # FIX: Use .policy attribute (as defined in PPOAgent)
        self.agent.policy.eval()  # Set to evaluation mode

        evaluation_results = []
        for _ in range(10):  # Quick 10-episode evaluation
            obs = self.environment.reset()
            episode_reward = 0.0
            episode_energy = 0.0
            done = False

            while not done:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, info = self.environment.step(action)
                episode_reward += reward
                episode_energy += info.get("energy_consumption", 0.0)

            # Check success
            final_pos = info.get("position", (0, 0, 0))
            goal_pos = info.get("goal_position", (0, 0, 0))
            final_dist = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
            success = final_dist <= 0.5 and not info.get("collision", False)

            evaluation_results.append(
                {"success": success, "energy": episode_energy, "reward": episode_reward}
            )

        self.agent.policy.train()  # Back to training mode

        # Calculate evaluation metrics
        success_rate = np.mean([r["success"] for r in evaluation_results]) * 100
        successful_episodes = [r for r in evaluation_results if r["success"]]
        avg_energy = (
            np.mean([r["energy"] for r in successful_episodes])
            if successful_episodes
            else float("inf")
        )

        eval_summary = {
            "timestep": self.state.timestep,
            "success_rate": success_rate,
            "energy_efficiency": avg_energy,
            "avg_reward": np.mean([r["reward"] for r in evaluation_results]),
        }

        self.logger.info(
            f"Evaluation: {success_rate:.1f}% success, {avg_energy:.0f}J energy"
        )

        return eval_summary

    def _check_early_stopping(self, eval_result: Dict[str, Any]):
        """Check early stopping criteria."""
        success_rate = eval_result.get("success_rate", 0.0)
        energy_efficiency = eval_result.get("energy_efficiency", float("inf"))

        # Check for improvement
        improvement = False

        if success_rate > self.state.best_success_rate:
            self.state.best_success_rate = success_rate
            improvement = True

        if energy_efficiency < self.state.best_energy_efficiency:
            self.state.best_energy_efficiency = energy_efficiency
            improvement = True

        if improvement:
            self.state.steps_since_improvement = 0
            self.logger.info(
                f"New best performance: {success_rate:.1f}% success, {energy_efficiency:.0f}J energy"
            )
            # Save best model
            self._save_checkpoint(is_best=True)
        else:
            self.state.steps_since_improvement += self.training_config.eval_frequency

    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered."""
        # Target achieved
        if (
            self.state.best_success_rate
            >= self.training_config.target_success_rate * 100
        ):
            self.logger.info(
                f"Target success rate {self.training_config.target_success_rate*100}% achieved"
            )
            return True

        # No improvement for too long
        if (
            self.state.steps_since_improvement
            >= self.training_config.early_stopping_patience
        ):
            self.logger.info(
                f"No improvement for {self.training_config.early_stopping_patience:,} steps"
            )
            return True

        return False

    def _record_episode_metrics(self, episode_result: Dict[str, Any]):
        """Record episode metrics for monitoring."""
        self.training_metrics["episode_rewards"].append(
            episode_result["episode_reward"]
        )
        self.training_metrics["episode_lengths"].append(
            episode_result["episode_length"]
        )

        if episode_result["success"]:
            self.training_metrics["energy_consumptions"].append(
                episode_result["episode_energy"]
            )

        # Calculate recent success rate
        recent_episodes = list(self.training_metrics["episode_rewards"])[-20:]
        if len(recent_episodes) >= 20:
            # Heuristic: Positive reward indicates success
            recent_successes = [1 if r > -100 else 0 for r in recent_episodes]
            success_rate = np.mean(recent_successes)
            self.training_metrics["success_rates"].append(success_rate)

        # Execute episode callbacks
        for callback in self.episode_callbacks:
            try:
                callback(episode_result)
            except Exception as e:
                self.logger.error(f"Episode callback error: {e}")

    def _record_update_metrics(self, update_result: Dict[str, float]):
        """Record policy update metrics."""
        if "policy_loss" in update_result:
            self.training_metrics["policy_losses"].append(update_result["policy_loss"])
        if "value_loss" in update_result:
            self.training_metrics["value_losses"].append(update_result["value_loss"])

        # Execute update callbacks
        for callback in self.update_callbacks:
            try:
                callback(update_result)
            except Exception as e:
                self.logger.error(f"Update callback error: {e}")

    def _log_training_progress(self):
        """Log training progress."""
        # Calculate averages
        recent_rewards = list(self.training_metrics["episode_rewards"])[-100:]
        recent_lengths = list(self.training_metrics["episode_lengths"])[-100:]
        recent_success = (
            list(self.training_metrics["success_rates"])[-10:]
            if self.training_metrics["success_rates"]
            else [0]
        )

        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        avg_length = np.mean(recent_lengths) if recent_lengths else 0
        avg_success = np.mean(recent_success) * 100

        # Training progress
        progress = (self.state.timestep / self.training_config.total_timesteps) * 100

        self.logger.info(
            f"[{progress:5.1f}%] Episode {self.state.episode:,}, "
            f"Timestep {self.state.timestep:,}/{self.training_config.total_timesteps:,}"
        )
        self.logger.info(
            f"  Reward: {avg_reward:8.1f}, Length: {avg_length:6.1f}, Success: {avg_success:5.1f}%"
        )

        # Log losses if available
        if self.training_metrics["policy_losses"]:
            recent_policy_loss = np.mean(
                list(self.training_metrics["policy_losses"])[-10:]
            )
            recent_value_loss = np.mean(
                list(self.training_metrics["value_losses"])[-10:]
            )
            self.logger.info(
                f"  Policy Loss: {recent_policy_loss:.4f}, Value Loss: {recent_value_loss:.4f}"
            )

        # Wandb logging
        if self.use_wandb:
            log_dict = {
                "timestep": self.state.timestep,
                "episode": self.state.episode,
                "avg_reward": avg_reward,
                "avg_episode_length": avg_length,
                "success_rate": avg_success,
                "learning_rate": self.state.current_lr,
            }

            if self.training_metrics["energy_consumptions"]:
                avg_energy = np.mean(
                    list(self.training_metrics["energy_consumptions"])[-50:]
                )
                log_dict["avg_energy"] = avg_energy

            if self.training_metrics["policy_losses"]:
                log_dict["policy_loss"] = recent_policy_loss
                log_dict["value_loss"] = recent_value_loss

            wandb.log(log_dict, step=self.state.timestep)

    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_data = {
            "agent_state": self.agent.policy.state_dict(),
            "optimizer_state": self.agent.optimizer.state_dict(),
            "training_state": {
                "timestep": self.state.timestep,
                "episode": self.state.episode,
                "best_success_rate": self.state.best_success_rate,
                "best_energy_efficiency": self.state.best_energy_efficiency,
                "steps_since_improvement": self.state.steps_since_improvement,
                "current_lr": self.state.current_lr,
            },
            "training_metrics": {
                "episode_rewards": list(self.training_metrics["episode_rewards"]),
                "episode_lengths": list(self.training_metrics["episode_lengths"]),
                "success_rates": list(self.training_metrics["success_rates"]),
                "energy_consumptions": list(
                    self.training_metrics["energy_consumptions"]
                ),
            },
            "training_config": self.training_config.__dict__,
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / f"best_model.pt"
        else:
            checkpoint_path = (
                self.checkpoint_dir / f"checkpoint_{self.state.timestep:08d}.pt"
            )

        torch.save(checkpoint_data, checkpoint_path)

        # Keep only recent checkpoints (except best)
        if not is_best:
            self._cleanup_old_checkpoints()

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location=self.agent.device
            )

            # Restore agent
            self.agent.policy.load_state_dict(checkpoint_data["agent_state"])
            self.agent.optimizer.load_state_dict(checkpoint_data["optimizer_state"])

            # Restore training state
            training_state = checkpoint_data["training_state"]
            self.state.timestep = training_state["timestep"]
            self.state.episode = training_state["episode"]
            self.state.best_success_rate = training_state["best_success_rate"]
            self.state.best_energy_efficiency = training_state["best_energy_efficiency"]
            self.state.steps_since_improvement = training_state[
                "steps_since_improvement"
            ]
            self.state.current_lr = training_state["current_lr"]

            # Restore metrics
            if "training_metrics" in checkpoint_data:
                metrics = checkpoint_data["training_metrics"]
                self.training_metrics["episode_rewards"].extend(
                    metrics.get("episode_rewards", [])
                )
                self.training_metrics["episode_lengths"].extend(
                    metrics.get("episode_lengths", [])
                )
                self.training_metrics["success_rates"].extend(
                    metrics.get("success_rates", [])
                )
                self.training_metrics["energy_consumptions"].extend(
                    metrics.get("energy_consumptions", [])
                )

            self.logger.info(f"Training resumed from {checkpoint_path}")
            self.logger.info(
                f"Resuming from timestep {self.state.timestep:,}, episode {self.state.episode:,}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Keep only the most recent checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))

        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                try:
                    old_checkpoint.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete old checkpoint: {e}")

    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform comprehensive final evaluation."""
        self.logger.info("Performing final evaluation...")

        # Use full evaluator with 100 episodes
        full_evaluator = DroneEvaluator({"num_episodes": 100})
        final_summary = full_evaluator.evaluate_policy(
            self.agent, self.environment, "PPO_Final"
        )

        return {
            "success_rate": final_summary.success_rate,
            "energy_efficiency": final_summary.mean_energy,
            "collision_rate": final_summary.collision_rate,
            "ate_error": final_summary.mean_ate,
            "flight_time": final_summary.mean_time,
        }

    def _save_final_model(self):
        """Save final trained model."""
        model_path = (
            self.checkpoint_dir / f"{self.training_config.experiment_name}_final.pt"
        )

        # Save using agent's save_model method
        self.agent.save_model(str(model_path))

        # Also save training summary
        summary_path = (
            self.checkpoint_dir / f"{self.training_config.experiment_name}_summary.json"
        )

        training_summary = {
            "experiment_name": self.training_config.experiment_name,
            "total_timesteps": self.state.timestep,
            "total_episodes": self.state.episode,
            "training_time_hours": (time.time() - self.state.training_start_time)
            / 3600,
            "best_success_rate": self.state.best_success_rate,
            "best_energy_efficiency": self.state.best_energy_efficiency,
            "final_learning_rate": self.state.current_lr,
            "agent_config": (
                self.agent.config.__dict__
                if hasattr(self.agent.config, "__dict__")
                else self.agent.config
            ),
            "training_config": self.training_config.__dict__,
        }

        with open(summary_path, "w") as f:
            json.dump(training_summary, f, indent=2)

        self.logger.info(f"Final model and summary saved to {self.checkpoint_dir}")

    def add_episode_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for episode completion."""
        self.episode_callbacks.append(callback)

    def add_update_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Add callback for policy updates."""
        self.update_callbacks.append(callback)

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        training_time = (
            time.time() - self.state.training_start_time
            if self.state.training_start_time > 0
            else 0
        )

        return {
            "training_progress": {
                "current_timestep": self.state.timestep,
                "current_episode": self.state.episode,
                "progress_percent": (
                    self.state.timestep / self.training_config.total_timesteps
                )
                * 100,
                "training_time_hours": training_time / 3600,
                "steps_per_second": self.state.timestep / max(1, training_time),
                "episodes_per_hour": self.state.episode / max(1, training_time / 3600),
            },
            "performance": {
                "best_success_rate": self.state.best_success_rate,
                "best_energy_efficiency": self.state.best_energy_efficiency,
                "recent_avg_reward": (
                    np.mean(list(self.training_metrics["episode_rewards"])[-20:])
                    if self.training_metrics["episode_rewards"]
                    else 0
                ),
                "recent_success_rate": (
                    np.mean(list(self.training_metrics["success_rates"])[-5:]) * 100
                    if self.training_metrics["success_rates"]
                    else 0
                ),
            },
            "training_stability": {
                "steps_since_improvement": self.state.steps_since_improvement,
                "current_learning_rate": self.state.current_lr,
                "policy_loss_trend": (
                    np.mean(list(self.training_metrics["policy_losses"])[-10:])
                    if self.training_metrics["policy_losses"]
                    else 0
                ),
                "value_loss_trend": (
                    np.mean(list(self.training_metrics["value_losses"])[-10:])
                    if self.training_metrics["value_losses"]
                    else 0
                ),
            },
        }
