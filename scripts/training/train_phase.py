import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import random

import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.training.curriculum_trainer import CurriculumTrainer
from src.rl.initialization import initialize_rl_system
from src.rl.utils.checkpoint_manager import CheckpointManager
from src.rl.utils.normalization import ObservationNormalizer
from src.rl.utils.tensorboard_logger import TensorBoardLogger
from src.utils import setup_logging, load_config, DataRecorder

def verify_airsim_connection(retries: int = 5, delay: float = 2.0) - None:

    import airsim

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            client = airsim.MultirotorClient()
            client.confirmConnection()
            logging.info("AirSim connection verified")
            return
        except Exception as exc:
            last_error = exc
            logging.warning(
                "AirSim connection attempt d/d failed: s", attempt, retries, exc
            )
            time.sleep(delay)

    logging.error("AirSim connection failed after d attempts: s", retries, last_error)
    logging.error("Make sure AirSim is running!")
    sys.exit(1)

def set_global_seeds(seed: int) - None:

    logging.info("Setting global random seed to d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MainPPOTrainer:

    def __init__(self, config_path: str, experiment_name: str = None):
        self.config = load_config(config_path)
        rl_seed = 42
        if isinstance(self.config.rl, dict):
            rl_seed = int(self.config.rl.get("seed", rl_seed))
        set_global_seeds(rl_seed)
        self.seed = rl_seed
        self.experiment_name = experiment_name or f"ppo_drone_{int(time.time())}"

        log_config = self.config.logging.copy()
        log_config["experiment_name"] = self.experiment_name
        self.logger_system = setup_logging(log_config)
        self.logger = logging.getLogger(__name__)

        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system["agent"]
        self.curriculum_manager = self.rl_system.get("curriculum_manager", None)

        checkpoint_config = (
            self.config.rl.get("checkpoints", {}).copy()
            if isinstance(self.config.rl, dict)
            else self.config.rl.checkpoints.copy()
        )
        checkpoint_config["experiment_name"] = self.experiment_name
        self.checkpoint_manager = CheckpointManager(checkpoint_config)

        self.obs_normalizer = ObservationNormalizer(
            observation_dim=40, config=self.config.rl.get("normalization", {})
        )

        tensorboard_config = self.config.rl.get(
            "logging", {"log_dir": "runs", "log_interval": 100}
        ).copy()
        tensorboard_config["experiment_name"] = self.experiment_name
        self.tensorboard_logger = TensorBoardLogger(tensorboard_config)

        self.data_recorder = DataRecorder(
            {
                "experiment_name": self.experiment_name,
                "save_trajectories": True,
                "save_energy_profiles": True,
            }
        )

        training_config = self.config.rl.get(
            "training",
            {"max_timesteps": 5000000, "eval_interval": 10000, "save_interval": 1000},
        )
        self.total_timesteps = training_config.get("total_timesteps", 5_000_000)
        self.eval_frequency = training_config.get("eval_frequency", 50_000)
        self.checkpoint_frequency = training_config.get("checkpoint_frequency", 100_000)
        self.phase_eval_frequency = training_config.get("phase_eval_frequency", 10_000)

        self.global_timestep = 0
        self.episode_count = 0
        self.training_start_time = None
        self.current_environment = None

        self.episode_rewards = []
        self.episode_energies = []
        self.success_rates = []
        self.phase_performance = {}

        self.logger.info(f"Main PPO Trainer initialized: {self.experiment_name}")
        self.logger.info(f"Target timesteps: {self.total_timesteps:,}")
        if self.curriculum_manager is not None:
            self.logger.info(
                f"Curriculum phases: {len(self.curriculum_manager.phases)}"
            )
        else:
            self.logger.info(
                "Curriculum manager not initialized - using standard training"
            )

    def train(self, resume_checkpoint: str = None) - Dict[str, Any]:

        self.logger.info("STARTING MAIN PPO TRAINING")
        self.training_start_time = time.time()

        verify_airsim_connection()
        self.logger.info("AirSim connection verified")

        if resume_checkpoint:
            self._load_checkpoint_and_resume(resume_checkpoint)

        self._setup_current_environment()

        while self.global_timestep  self.total_timesteps:
            try:
                if self._check_phase_advancement():
                    self._advance_curriculum_phase()

                episode_results = self._execute_training_episode()

                self._update_performance_metrics(episode_results)

                if self.global_timestep  self.phase_eval_frequency == 0:
                    self._run_phase_evaluation()

                if self.global_timestep  self.eval_frequency == 0:
                    self._run_comprehensive_evaluation()

                if self.global_timestep  self.checkpoint_frequency == 0:
                    self._create_checkpoint()

                if self.episode_count  50 == 0:
                    self._report_training_progress()

                    if self.episode_count  100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.debug(
                                f"GPU Memory: {torch.cuda.memory_allocated() / 10243:.2f}GB allocated"
                            )
            except KeyboardInterrupt:
                self.logger.info("Training interrupted by user")
                self._handle_training_interruption()
                break

            except Exception as e:
                self.logger.error(f"Training error: {e}", exc_info=True)
                self._handle_training_error(e)

        final_results = self._complete_training()

        self.tensorboard_logger.close()

        return final_results

    def _setup_current_environment(self):

        if self.curriculum_manager is not None:
            current_phase = self.curriculum_manager.get_current_phase()
        else:
            current_phase = {"name": "standard", "difficulty": 1}

        env_config = self.config.environment.copy()
        phase_config = current_phase.get("config", {})

        if "floors" in phase_config:
            env_config["building"]["floors"] = phase_config["floors"]

        if "obstacle_density" in phase_config:
            env_config["obstacles"]["density"] = phase_config["obstacle_density"]

        if "dynamic_obstacles" in phase_config:
            env_config["obstacles"]["dynamic_obstacles"] = phase_config[
                "dynamic_obstacles"
            ]
            env_config["obstacles"]["dynamic_count"] = phase_config.get(
                "dynamic_count", 3
            )

        self.current_environment = DroneEnvironment(env_config)

        phase_name = current_phase.get("name", "Unknown")
        self.logger.info(f"Environment setup for phase: {phase_name}")
        self.logger.info(f"Floors: {env_config['building']['floors']}")
        self.logger.info(f"Obstacle density: {env_config['obstacles']['density']}")

    def _execute_training_episode(self) - Dict[str, Any]:

        episode_start = time.time()

        observation, _ = self.current_environment.reset()
        observation = self.obs_normalizer.normalize(observation)

        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
            "energy_consumption": [],
            "positions": [],
        }

        episode_reward = 0.0
        episode_energy = 0.0
        episode_steps = 0
        done = False
        last_info: Dict[str, Any] = {}

        while not done and episode_steps  1000:
            action, log_prob, value = self.agent.select_action(observation)

            next_observation, reward, terminated, truncated, info = (
                self.current_environment.step(action)
            )
            done = terminated or truncated
            next_observation = self.obs_normalizer.normalize(next_observation)

            episode_data["observations"].append(observation)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["values"].append(value)
            episode_data["log_probs"].append(log_prob)
            episode_data["dones"].append(done)
            episode_data["energy_consumption"].append(
                info.get("energy_consumption", 0.0)
            )
            episode_data["positions"].append(info.get("position", [0, 0, 0]))
            last_info = info

            episode_reward += reward
            episode_energy += info.get("energy_consumption", 0.0)
            episode_steps += 1
            self.global_timestep += 1

            self.obs_normalizer.update_statistics(next_observation)

            observation = next_observation

        final_value = self.agent.get_value(observation) if not done else 0.0
        training_metrics = self.agent.update_policy(episode_data, final_value)
        trajectory_positions = list(episode_data["positions"])

        self.data_recorder.record_episode(
            {
                "episode": self.episode_count,
                "timestep": self.global_timestep,
                "reward": episode_reward,
                "energy": episode_energy,
                "success": last_info.get("success", False),
                "trajectory": trajectory_positions,
                "phase": (
                    self.curriculum_manager.get_current_phase()["name"]
                    if self.curriculum_manager
                    else "phase1"
                ),
            }
        )

        for data_list in episode_data.values():
            if isinstance(data_list, list):
                data_list.clear()
        episode_data.clear()
        del episode_data

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.episode_count += 1
        episode_time = time.time() - episode_start

        return {
            "reward": episode_reward,
            "energy": episode_energy,
            "steps": episode_steps,
            "success": last_info.get("success", False),
            "collision": last_info.get("collision", False),
            "training_metrics": training_metrics,
            "episode_time": episode_time,
        }

    def _check_phase_advancement(self) - bool:

        if self.curriculum_manager is None:
            return False

        recent_performance = self._calculate_recent_performance()
        return self.curriculum_manager.should_advance_phase(
            timesteps=self.total_timesteps,
            success_rate=recent_performance["success_rate"],
            avg_reward=recent_performance["avg_reward"],
        )

    def _advance_curriculum_phase(self):

        if self.curriculum_manager is None:
            self.logger.warning(
                "Requested curriculum phase advancement without a curriculum manager"
            )
            return False

        old_phase = self.curriculum_manager.get_current_phase()

        completion_metrics = self._calculate_recent_performance()
        self.phase_performance[old_phase["name"]] = {
            "completion_timestep": self.global_timestep,
            "completion_episode": self.episode_count,
            "final_success_rate": completion_metrics.get("success_rate", 0),
            "final_energy": completion_metrics.get("mean_energy", 0),
        }

        advanced = self.curriculum_manager.advance_phase()

        if advanced:
            new_phase = self.curriculum_manager.get_current_phase()
            self.logger.info(
                f" PHASE ADVANCEMENT: {old_phase['name']}  {new_phase['name']}"
            )

            self._setup_current_environment()

            self.tensorboard_logger.log_phase_transition(
                old_phase["name"],
                new_phase["name"],
                self.phase_performance[old_phase["name"]],
                self.global_timestep,
            )

        return advanced

    def _calculate_recent_performance(self) - Dict[str, float]:

        if len(self.episode_rewards)  10:
            return {"success_rate": 0.0, "mean_energy": 0.0, "avg_reward": 0.0}

        recent_count = min(50, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_count:]
        recent_energies = self.episode_energies[-recent_count:]

        successes = sum(1 for r in recent_rewards if r  400)
        success_rate = successes / len(recent_rewards)  100

        successful_energies = [
            e for r, e in zip(recent_rewards, recent_energies) if r  400
        ]
        mean_energy = np.mean(successful_energies) if successful_energies else 0.0

        avg_reward = float(np.mean(recent_rewards))

        return {
            "success_rate": success_rate,
            "mean_energy": float(mean_energy),
            "avg_reward": avg_reward
        }

    def _update_performance_metrics(self, episode_results: Dict[str, Any]):

        self.episode_rewards.append(episode_results["reward"])
        self.episode_energies.append(episode_results["energy"])

        if len(self.episode_rewards) = 20:
            recent_successes = sum(1 for r in self.episode_rewards[-20:] if r  400)
            success_rate = recent_successes / 20  100
            self.success_rates.append(success_rate)

        metrics = {
            "Episode/Reward": episode_results["reward"],
            "Episode/Energy": episode_results["energy"],
            "Episode/Steps": episode_results["steps"],
            "Episode/Success": 1.0 if episode_results["success"] else 0.0,
        }

        if self.success_rates:
            metrics["Performance/Success_Rate"] = self.success_rates[-1]

        if "training_metrics" in episode_results:
            train_metrics = episode_results["training_metrics"]
            metrics.update(
                {
                    "Training/Policy_Loss": train_metrics.get("policy_loss", 0),
                    "Training/Value_Loss": train_metrics.get("value_loss", 0),
                    "Training/Entropy": train_metrics.get("entropy", 0),
                    "Training/KL_Divergence": train_metrics.get("kl_divergence", 0),
                }
            )

        self.tensorboard_logger.log_scalars(metrics, self.global_timestep)

    def _run_phase_evaluation(self):

        if not self.current_environment:
            return

        try:
            eval_results = []
            for _ in range(5):
                try:
                    observation, _ = self.current_environment.reset()
                    observation = self.obs_normalizer.normalize(observation)

                    episode_reward = 0.0
                    episode_energy = 0.0
                    done = False
                    steps = 0
                    last_info: Dict[str, Any] = {}

                    while not done and steps  500:
                        action, _, _ = self.agent.select_action(
                            observation, deterministic=True
                        )
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = self.current_environment.step(action)
                        done = terminated or truncated
                        observation = self.obs_normalizer.normalize(next_observation)

                        episode_reward += reward
                        episode_energy += info.get("energy_consumption", 0.0)
                        steps += 1
                        last_info = info

                    eval_results.append(
                        {
                            "reward": episode_reward,
                            "energy": episode_energy,
                            "success": last_info.get("success", False),
                        }
                    )
                except Exception as episode_error:
                    self.logger.warning(
                        "Phase evaluation episode failed: s", episode_error
                    )
                    continue

            if not eval_results:
                self.logger.warning(
                    "Phase evaluation skipped because all evaluation episodes failed."
                )
                return

            successes = [r for r in eval_results if r["success"]]
            success_rate = len(successes) / len(eval_results)  100
            mean_energy = (
                np.mean([r["energy"] for r in successes]) if successes else 0
            )

            self.tensorboard_logger.log_scalars(
                {
                    "Phase_Eval/Success_Rate": success_rate,
                    "Phase_Eval/Mean_Energy": mean_energy,
                },
                self.global_timestep,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as error:
            self.logger.error(
                f"Phase evaluation failed due to unexpected error: {error}",
                exc_info=True,
            )

    def _run_comprehensive_evaluation(self):

        if not self.current_environment:
            self.logger.warning("Cannot run evaluation - environment is not initialized")
            return {
                "success_rate": 0.0,
                "mean_energy": 0.0,
                "collision_rate": 0.0,
            }

        self.logger.info(
            f"Running comprehensive evaluation at timestep {self.global_timestep:,}"
        )

        try:
            eval_results = []
            for episode in range(20):
                try:
                    observation, _ = self.current_environment.reset()
                    observation = self.obs_normalizer.normalize(observation)

                    episode_reward = 0.0
                    episode_energy = 0.0
                    trajectory = []
                    done = False
                    steps = 0
                    last_info: Dict[str, Any] = {}

                    while not done and steps  1000:
                        action, _, _ = self.agent.select_action(
                            observation, deterministic=True
                        )
                        (
                            next_observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ) = self.current_environment.step(action)
                        done = terminated or truncated
                        observation = self.obs_normalizer.normalize(next_observation)

                        episode_reward += reward
                        episode_energy += info.get("energy_consumption", 0.0)
                        trajectory.append(info.get("position", [0, 0, 0]))
                        steps += 1
                        last_info = info

                    eval_results.append(
                        {
                            "reward": episode_reward,
                            "energy": episode_energy,
                            "success": last_info.get("success", False),
                            "collision": last_info.get("collision", False),
                            "trajectory": trajectory,
                            "steps": steps,
                        }
                    )
                except Exception as episode_error:
                    self.logger.warning(
                        "Evaluation episode d failed: s", episode, episode_error
                    )
                    continue

            if not eval_results:
                self.logger.warning(
                    "Comprehensive evaluation skipped because all episodes failed."
                )
                return {
                    "success_rate": 0.0,
                    "mean_energy": 0.0,
                    "collision_rate": 0.0,
                }

            successes = [r for r in eval_results if r["success"]]
            success_rate = len(successes) / len(eval_results)  100
            mean_energy = np.mean([r["energy"] for r in successes]) if successes else 0
            collision_rate = (
                np.mean([r["collision"] for r in eval_results])  100
                if eval_results
                else 0
            )
            mean_steps = np.mean([r["steps"] for r in successes]) if successes else 0

            eval_metrics = {
                "Eval/Success_Rate": success_rate,
                "Eval/Mean_Energy": mean_energy,
                "Eval/Collision_Rate": collision_rate,
                "Eval/Mean_Steps": mean_steps,
                "Eval/Episodes": len(eval_results),
            }

            self.tensorboard_logger.log_scalars(eval_metrics, self.global_timestep)

            self.logger.info(
                f"Evaluation: {success_rate:.1f} success, {mean_energy:.0f}J energy, "
                f"{collision_rate:.1f} collisions"
            )

            return {
                "success_rate": success_rate,
                "mean_energy": mean_energy,
                "collision_rate": collision_rate,
            }
        except Exception as error:
            self.logger.error(
                f"Comprehensive evaluation failed: {error}", exc_info=True
            )
            return {
                "success_rate": 0.0,
                "mean_energy": 0.0,
                "collision_rate": 0.0,
            }

    def _create_checkpoint(self):

        recent_performance = self._calculate_recent_performance()
        curriculum_phase_index = (
            self.curriculum_manager.current_phase_index
            if self.curriculum_manager is not None
            else -1
        )
        phase_name = (
            self.curriculum_manager.get_current_phase()["name"]
            if self.curriculum_manager is not None
            else "standard"
        )

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.agent,
            self.global_timestep,
            self.episode_count,
            recent_performance["success_rate"],
            recent_performance["mean_energy"],
            additional_data={
                "curriculum_phase": curriculum_phase_index,
                "phase_name": phase_name,
                "normalization_stats": {
                    "mean": self.obs_normalizer.mean.tolist(),
                    "variance": self.obs_normalizer.variance.tolist(),
                    "count": self.obs_normalizer.count,
                },
                "phase_performance": self.phase_performance,
                "experiment_name": self.experiment_name,
            },
        )

        self.logger.info(f" Checkpoint saved: {Path(checkpoint_path).name}")

    def _report_training_progress(self):

        elapsed_time = time.time() - self.training_start_time
        progress = self.global_timestep / self.total_timesteps  100

        if progress  0:
            total_estimated_time = elapsed_time / (progress / 100)
            remaining_time = total_estimated_time - elapsed_time
        else:
            remaining_time = 0

        recent_performance = self._calculate_recent_performance()
        if self.curriculum_manager is not None:
            current_phase = self.curriculum_manager.get_current_phase()
        else:
            current_phase = {"name": "standard", "difficulty": 1}

        self.logger.info(f" TRAINING PROGRESS")
        self.logger.info(
            f"Progress: {progress:.1f} ({self.global_timestep:,}/{self.total_timesteps:,})"
        )
        self.logger.info(f"Episode: {self.episode_count:,}")
        self.logger.info(
            f"Time: {elapsed_time/3600:.1f}h elapsed, {remaining_time/3600:.1f}h remaining"
        )
        self.logger.info(f"Phase: {current_phase['name']}")
        self.logger.info(f"Success: {recent_performance['success_rate']:.1f}")
        self.logger.info(f"Energy: {recent_performance['mean_energy']:.0f}J")

    def _complete_training(self) - Dict[str, Any]:

        total_time = time.time() - self.training_start_time

        final_evaluation = self._run_comprehensive_evaluation()

        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            self.agent,
            self.global_timestep,
            self.episode_count,
            final_evaluation["success_rate"],
            final_evaluation["mean_energy"],
            additional_data={
                "training_completed": True,
                "final_model": True,
                "experiment_name": self.experiment_name,
                "checkpoint_type": "final"
            }
        )

        training_history = {
            "episode_rewards": self.episode_rewards,
            "episode_energies": self.episode_energies,
            "success_rates": self.success_rates,
            "phase_performance": self.phase_performance,
        }

        history_path = (
            Path(final_checkpoint).parent / f"{self.experiment_name}_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2, default=str)

        phases_completed = (
            self.curriculum_manager.current_phase_index + 1
            if self.curriculum_manager is not None
            else 0
        )

        final_results = {
            "training_completed": True,
            "experiment_name": self.experiment_name,
            "total_timesteps": self.global_timestep,
            "total_episodes": self.episode_count,
            "total_training_time_hours": total_time / 3600,
            "phases_completed": phases_completed,
            "final_checkpoint": final_checkpoint,
            "final_evaluation": final_evaluation,
            "phase_performance_summary": self.phase_performance,
            "target_achievements": {
                "success_rate_96_percent": final_evaluation["success_rate"] = 96.0,
                "energy_efficiency": final_evaluation["mean_energy"] = 700.0,
                "collision_safety": final_evaluation["collision_rate"] = 2.0,
            },
        }

        self.logger.info(" MAIN PPO TRAINING COMPLETED!")
        self.logger.info(f"Total time: {total_time/3600:.1f} hours")
        self.logger.info(f"Final success rate: {final_evaluation['success_rate']:.1f}")
        self.logger.info(f"Final energy: {final_evaluation['mean_energy']:.0f}J")
        self.logger.info(f"Model ready for evaluation!")

        if self.current_environment:
            try:
                self.current_environment.reset()
            except Exception as cleanup_error:
                self.logger.warning(
                    "Environment reset during cleanup failed: s", cleanup_error
                )

            if hasattr(self.current_environment, "close"):
                try:
                    self.current_environment.close()
                except Exception as cleanup_error:
                    self.logger.warning(
                        "Environment close during cleanup failed: s", cleanup_error
                    )

            self.current_environment = None

        return final_results

    def _load_checkpoint_and_resume(self, checkpoint_path: str):

        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location=self.agent.device
            )

            self.agent.policy.load_state_dict(checkpoint_data["agent_state_dict"])
            if "optimizer_state_dict" in checkpoint_data:
                self.agent.optimizer.load_state_dict(
                    checkpoint_data["optimizer_state_dict"]
                )

            training_info = checkpoint_data["training_info"]
            self.global_timestep = training_info.get("timestep", 0)
            self.episode_count = training_info.get("episode", 0)

            additional_data = checkpoint_data.get("additional_data", {})
            curriculum_phase = additional_data.get("curriculum_phase")
            if (
                curriculum_phase is not None
                and self.curriculum_manager is not None
            ):
                self.curriculum_manager.current_phase_index = curriculum_phase
            elif curriculum_phase is not None:
                self.logger.warning(
                    "Checkpoint specified curriculum phase but no curriculum manager is configured"
                )

            if "normalization_stats" in additional_data:
                norm_stats = additional_data["normalization_stats"]
                self.obs_normalizer.mean = torch.tensor(norm_stats["mean"])
                self.obs_normalizer.variance = torch.tensor(norm_stats["variance"])
                self.obs_normalizer.count = norm_stats["count"]

            self.logger.info(
                f"Resumed from checkpoint at timestep {self.global_timestep:,}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def _handle_training_interruption(self):

        self.logger.info("Saving emergency checkpoint...")

        try:
            recent_performance = self._calculate_recent_performance()
            emergency_checkpoint = self.checkpoint_manager.save_checkpoint(
                self.agent,
                self.global_timestep,
                self.episode_count,
                recent_performance["success_rate"],
                recent_performance["mean_energy"],
                additional_data={
                    "emergency_save": True,
                    "checkpoint_type": "emergency"
                }
            )

            self.logger.info(f"Emergency checkpoint saved: {emergency_checkpoint}")

        except Exception as e:
            self.logger.error(f"Emergency save failed: {e}")

    def _handle_training_error(self, error: Exception):

        self.logger.error(f"Training error occurred: {error}")

        try:
            self._create_checkpoint()
            self.logger.info("Current state saved before error handling")
        except:
            self.logger.warning("Could not save checkpoint during error handling")

        raise error

def main():
    parser = argparse.ArgumentParser(
        description="Main PPO training for drone navigation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/main_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total timesteps"
    )

    args = parser.parse_args()

    if not args.name:
        args.name = f"ppo_drone_{int(time.time())}"

    trainer = MainPPOTrainer(args.config, args.name)

    if args.timesteps:
        trainer.total_timesteps = args.timesteps

    try:
        results = trainer.train(resume_checkpoint=args.resume)

        print("\n" + "="  70)
        print(" PPO TRAINING COMPLETED SUCCESSFULLY!")
        print("="  70)
        print(f"Experiment: {results['experiment_name']}")
        print(f"Total timesteps: {results['total_timesteps']:,}")
        print(f"Total episodes: {results['total_episodes']:,}")
        print(f"Training time: {results['total_training_time_hours']:.1f} hours")
        print(f"Phases completed: {results['phases_completed']}/3")
        print(f"Final success rate: {results['final_evaluation']['success_rate']:.1f}")
        print(
            f"Final energy consumption: {results['final_evaluation']['mean_energy']:.0f}J"
        )
        print(f"Target achievements:")
        for target, achieved in results["target_achievements"].items():
            status = "" if achieved else ""
            print(f"  {status} {target}")
        print(f"Final model: {Path(results['final_checkpoint']).name}")
        print("="  70)
        print(" Ready for comprehensive evaluation!")

    except KeyboardInterrupt:
        print("\n Training interrupted by user")
    except Exception as e:
        print(f"\n Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
