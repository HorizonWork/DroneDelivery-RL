import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

class Phase3Trainer:

    def __init__(self, agent, environment, config: Dict[str, Any]):
        self.agent = agent
        self.environment = environment
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.target_success_rate = 0.96
        self.target_energy_savings = 0.25
        self.max_episode_length = 2000
        self.obstacle_density = 0.20
        self.dynamic_obstacle_count = 5

        self.reward_weights = {
            "goal_reached": 500.0,
            "energy_penalty": -0.5,
            "collision_penalty": -200.0,
            "progress_reward": 3.0,
            "time_penalty": -0.05,
            "floor_transition_bonus": 20.0,
            "dynamic_avoidance_bonus": 10.0,
            "energy_efficiency_bonus": 50.0,
            "exploration_penalty": -1.0,
            "smoothness_bonus": 5.0,
        }

        self.all_floors_enabled = True
        self.complex_obstacles = True
        self.advanced_dynamics = True

        self.target_energy_per_meter = 100.0
        self.baseline_energy = 2800.0

        self.phase_timesteps = 0
        self.target_timesteps = 2_000_000

        self.energy_efficiency_history = deque(maxlen=100)
        self.multi_floor_success_history = deque(maxlen=100)

        self.logger.info("Phase 3 Trainer initialized")
        self.logger.info("Training scope: Five floors + full complexity")
        self.logger.info(
            f"Target: {self.target_success_rate100} success, {self.target_energy_savings100} energy savings"
        )
        self.logger.info(f"Max dynamic obstacles: {self.dynamic_obstacle_count}")

    def train_phase(self) - Dict[str, Any]:

        self.logger.info("Starting Phase 3 training - FINAL PHASE")
        phase_start = time.time()

        self._configure_phase3_environment()

        episode_count = 0
        recent_successes = deque(maxlen=100)
        recent_energies = deque(maxlen=100)
        recent_multi_floor_successes = deque(maxlen=50)

        best_energy_efficiency = 0.0
        consecutive_good_episodes = 0

        while self.phase_timesteps  self.target_timesteps:
            episode_result = self._run_phase3_episode()
            episode_count += 1

            self.phase_timesteps += episode_result["episode_length"]
            recent_successes.append(episode_result["success"])

            if episode_result["success"]:
                recent_energies.append(episode_result["episode_energy"])

                path_length = episode_result.get("path_length", 1.0)
                energy_per_meter = episode_result["episode_energy"] / path_length

                if energy_per_meter = self.target_energy_per_meter:
                    consecutive_good_episodes += 1
                else:
                    consecutive_good_episodes = 0

                if episode_result.get("floor_transitions", 0)  0:
                    recent_multi_floor_successes.append(True)
                else:
                    recent_multi_floor_successes.append(False)

                self.energy_efficiency_history.append(energy_per_meter)

            if self.agent.is_ready_for_update():
                self._adaptive_learning_rate_update()

                update_result = self.agent.update_policy(
                    next_observation=self.environment.get_observation(), next_done=False
                )

                self._monitor_training_stability(update_result)

            if episode_count  50 == 0:
                self._log_phase3_progress(
                    episode_count,
                    recent_successes,
                    recent_energies,
                    recent_multi_floor_successes,
                )

            if len(recent_successes) = 100 and len(recent_energies) = 50:

                current_success = np.mean(recent_successes)
                current_energy_efficiency = self._calculate_energy_savings(
                    recent_energies
                )

                if (
                    current_success = self.target_success_rate
                    and current_energy_efficiency = self.target_energy_savings
                    and consecutive_good_episodes = 20
                ):

                    self.logger.info("Phase 3 ALL TARGETS ACHIEVED!")
                    break

        phase_time = time.time() - phase_start

        final_evaluation = self._evaluate_phase3_comprehensive()

        results = {
            "phase": "five_floors_full",
            "training_time": phase_time,
            "timesteps_trained": self.phase_timesteps,
            "episodes_trained": episode_count,
            "final_success_rate": final_evaluation["success_rate"],
            "final_energy": final_evaluation["mean_energy"],
            "energy_efficiency_improvement": final_evaluation["energy_savings_percent"],
            "final_collision_rate": final_evaluation["collision_rate"],
            "multi_floor_mastery": final_evaluation["multi_floor_success_rate"],
            "target_achieved": final_evaluation["all_targets_met"],
            "consecutive_good_episodes": consecutive_good_episodes,
        }

        self.logger.info(
            f"Phase 3 COMPLETED: {final_evaluation['success_rate']:.1f} success"
        )
        self.logger.info(
            f"Energy savings: {final_evaluation['energy_savings_percent']:.1f}"
        )
        self.logger.info(
            f"All targets met: {'' if final_evaluation['all_targets_met'] else ''}"
        )

        return results

    def _configure_phase3_environment(self):

        env_config = {
            "building_floors": 5,
            "obstacle_density": self.obstacle_density,
            "dynamic_obstacles": True,
            "dynamic_obstacle_count": self.dynamic_obstacle_count,
            "obstacle_speed_range": (0.3, 2.5),
            "max_episode_steps": self.max_episode_length,
            "reward_weights": self.reward_weights,
            "goal_spawn_floors": [1, 2, 3, 4, 5],
            "start_spawn_floors": [1, 2, 3, 4, 5],
            "floor_transition_enabled": True,
            "complex_obstacles": True,
            "weather_effects": False,
            "sensor_noise": True,
            "localization_uncertainty": True,
            "energy_modeling": "advanced",
            "emergency_scenarios": True,
        }

        if hasattr(self.environment, "configure"):
            self.environment.configure(env_config)

    def _run_phase3_episode(self) - Dict[str, Any]:

        obs = self.environment.reset()

        episode_reward = 0.0
        episode_length = 0
        episode_energy = 0.0
        floor_transitions = 0
        smoothness_score = 0.0
        path_length = 0.0

        done = False
        previous_position = obs[:3] if len(obs) = 3 else np.array([0, 0, 0])
        previous_floor = self._get_current_floor(obs)

        positions = [previous_position.copy()]

        while not done and episode_length  self.max_episode_length:
            action, log_prob = self.agent.select_action(obs)
            value = self.agent.evaluate_observation(obs)

            next_obs, reward, done, info = self.environment.step(action)

            enhanced_reward = self._calculate_phase3_reward(
                reward, info, obs, next_obs, episode_length
            )

            self.agent.add_experience(
                obs, action, enhanced_reward, value, log_prob, done
            )

            current_position = (
                next_obs[:3] if len(next_obs) = 3 else np.array([0, 0, 0])
            )
            positions.append(current_position.copy())

            step_distance = np.linalg.norm(current_position - previous_position)
            path_length += step_distance

            current_floor = self._get_current_floor(next_obs)
            if current_floor != previous_floor:
                floor_transitions += 1
                previous_floor = current_floor

            episode_reward += enhanced_reward
            episode_length += 1
            episode_energy += info.get("energy_consumption", 0.0)

            obs = next_obs
            previous_position = current_position

        if len(positions)  3:
            smoothness_score = self._calculate_trajectory_smoothness(positions)

        final_pos = info.get("position", (0, 0, 0))
        goal_pos = info.get("goal_position", (0, 0, 0))
        final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))

        success = (
            final_distance = 0.5
            and not info.get("collision", False)
            and episode_energy  self.baseline_energy  0.9
        )

        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "episode_energy": episode_energy,
            "success": success,
            "floor_transitions": floor_transitions,
            "smoothness_score": smoothness_score,
            "path_length": path_length,
            "final_distance": final_distance,
            "energy_per_meter": episode_energy / max(1.0, path_length),
        }

    def _calculate_phase3_reward(
        self,
        base_reward: float,
        info: Dict[str, Any],
        obs: np.ndarray,
        next_obs: np.ndarray,
        episode_step: int,
    ) - float:

        enhanced_reward = base_reward

        energy_consumption = info.get("energy_consumption", 0.0)
        if energy_consumption  0:
            efficiency_ratio = min(2.0, 2.0 / max(1.0, energy_consumption))
            enhanced_reward += self.reward_weights["energy_efficiency_bonus"]  (
                efficiency_ratio - 1.0
            )

        if info.get("trajectory_smooth", False):
            enhanced_reward += self.reward_weights["smoothness_bonus"]

        current_floor = self._get_current_floor(next_obs)
        goal_floor = self._get_goal_floor(info.get("goal_position", (0, 0, 0)))
        floor_distance = abs(current_floor - goal_floor)

        if floor_distance == 0:
            enhanced_reward += 10.0
        elif floor_distance == 1:
            enhanced_reward += 5.0

        if episode_step  1000 and info.get("progress_rate", 1.0)  0.1:
            enhanced_reward += self.reward_weights["exploration_penalty"]

        return enhanced_reward

    def _calculate_trajectory_smoothness(self, positions: List[np.ndarray]) - float:

        if len(positions)  4:
            return 1.0

        positions_array = np.array(positions)

        velocities = np.diff(positions_array, axis=0)

        accelerations = np.diff(velocities, axis=0)

        jerks = np.diff(accelerations, axis=0)
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)

        mean_jerk = np.mean(jerk_magnitudes)
        smoothness = 1.0 / (1.0 + mean_jerk / 2.0)

        return float(np.clip(smoothness, 0.0, 1.0))

    def _calculate_energy_savings(self, recent_energies: deque) - float:

        if not recent_energies:
            return 0.0

        current_energy = np.mean(recent_energies)
        savings_percent = (
            (self.baseline_energy - current_energy) / self.baseline_energy
        )  100

        return max(0.0, savings_percent)

    def _adaptive_learning_rate_update(self):

        if len(self.energy_efficiency_history) = 20:
            recent_efficiency = list(self.energy_efficiency_history)[-20:]
            efficiency_trend = np.polyfit(range(20), recent_efficiency, 1)[0]

            if abs(efficiency_trend)  1.0:
                new_lr = max(1e-5, self.agent.config.learning_rate  0.95)
                self.agent.set_learning_rate(new_lr)

    def _monitor_training_stability(self, update_result: Dict[str, float]):

        policy_loss = update_result.get("policy_loss", 0.0)
        value_loss = update_result.get("value_loss", 0.0)

        if policy_loss  1.0 or value_loss  10.0:
            self.logger.warning(
                f"Training instability detected: "
                f"Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}"
            )

            current_lr = self.agent.config.learning_rate
            self.agent.set_learning_rate(current_lr  0.8)

    def _log_phase3_progress(
        self,
        episode_count: int,
        recent_successes: deque,
        recent_energies: deque,
        multi_floor_successes: deque,
    ):

        current_success = np.mean(recent_successes)  100 if recent_successes else 0.0
        current_energy = np.mean(recent_energies) if recent_energies else 0.0
        energy_savings = self._calculate_energy_savings(recent_energies)
        multi_floor_success = (
            np.mean(multi_floor_successes)  100 if multi_floor_successes else 0.0
        )

        progress = (self.phase_timesteps / self.target_timesteps)  100

        self.logger.info(f"Phase 3 Episode {episode_count} [{progress:.1f}]:")
        self.logger.info(f"  Success: {current_success:.1f} (target: 96)")
        self.logger.info(
            f"  Energy: {current_energy:.0f}J (savings: {energy_savings:.1f})"
        )
        self.logger.info(f"  Multi-floor: {multi_floor_success:.1f}")

        targets_status = []
        if current_success = 96.0:
            targets_status.append(" Success")
        else:
            targets_status.append(" Success")

        if energy_savings = 25.0:
            targets_status.append(" Energy")
        else:
            targets_status.append(" Energy")

        self.logger.info(f"  Targets: {', '.join(targets_status)}")

    def _evaluate_phase3_comprehensive(self) - Dict[str, Any]:

        self.logger.info("Performing comprehensive Phase 3 evaluation...")

        self.agent.policy.eval()

        eval_results = []
        multi_floor_episodes = 0
        successful_multi_floor = 0
        energy_efficient_episodes = 0

        for eval_episode in range(50):
            obs = self.environment.reset()
            episode_energy = 0.0
            floor_transitions = 0
            path_length = 0.0
            done = False

            start_pos = obs[:3] if len(obs) = 3 else np.array([0, 0, 0])
            goal_pos = self.environment.get_goal_position()

            start_floor = max(1, min(5, int(start_pos[2]
            goal_floor = max(1, min(5, int(goal_pos[2]

            if abs(start_floor - goal_floor) = 1:
                multi_floor_episodes += 1

            previous_position = start_pos.copy()
            previous_floor = start_floor

            while not done:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, info = self.environment.step(action)

                current_position = obs[:3] if len(obs) = 3 else np.array([0, 0, 0])
                step_distance = np.linalg.norm(current_position - previous_position)
                path_length += step_distance

                current_floor = max(1, min(5, int(current_position[2]
                if current_floor != previous_floor:
                    floor_transitions += 1
                    previous_floor = current_floor

                episode_energy += info.get("energy_consumption", 0.0)
                previous_position = current_position

            final_pos = info.get("position", (0, 0, 0))
            final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
            success = final_distance = 0.5 and not info.get("collision", False)

            if abs(start_floor - goal_floor) = 1 and success:
                successful_multi_floor += 1

            energy_per_meter = episode_energy / max(1.0, path_length)
            if energy_per_meter = self.target_energy_per_meter:
                energy_efficient_episodes += 1

            eval_results.append(
                {
                    "success": success,
                    "energy": episode_energy,
                    "energy_per_meter": energy_per_meter,
                    "collision": info.get("collision", False),
                    "floor_transitions": floor_transitions,
                    "path_length": path_length,
                    "multi_floor": abs(start_floor - goal_floor) = 1,
                }
            )

        self.agent.policy.train()

        successful_results = [r for r in eval_results if r["success"]]

        success_rate = len(successful_results) / len(eval_results)  100
        mean_energy = (
            np.mean([r["energy"] for r in successful_results])
            if successful_results
            else 0
        )
        energy_savings_percent = (
            ((self.baseline_energy - mean_energy) / self.baseline_energy)  100
            if mean_energy  0
            else 0
        )
        collision_rate = np.mean([r["collision"] for r in eval_results])  100
        multi_floor_success_rate = (
            successful_multi_floor / max(1, multi_floor_episodes)
        )  100

        all_targets_met = (
            success_rate = 96.0
            and energy_savings_percent = 25.0
            and collision_rate = 2.0
            and multi_floor_success_rate = 90.0
        )

        return {
            "success_rate": success_rate,
            "mean_energy": mean_energy,
            "energy_savings_percent": energy_savings_percent,
            "collision_rate": collision_rate,
            "multi_floor_success_rate": multi_floor_success_rate,
            "energy_efficient_episodes": energy_efficient_episodes,
            "total_evaluation_episodes": len(eval_results),
            "multi_floor_episodes_evaluated": multi_floor_episodes,
            "all_targets_met": all_targets_met,
        }

    def set_phase_steps(self, steps: int):

        self.target_timesteps = steps

    def set_starting_timestep(self, timestep: int):

        self.phase_timesteps = 0
