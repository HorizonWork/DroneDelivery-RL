import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from .trainer import PPOTrainer

class Phase1Trainer:

    def __init__(self, agent, environment, config: Dict[str, Any]):
        self.agent = agent
        self.environment = environment
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.target_success_rate = 0.85
        self.max_episode_length = 1000
        self.obstacle_density = 0.1

        self.reward_weights = {
            'goal_reached': 100.0,
            'energy_penalty': -0.1,
            'collision_penalty': -50.0,
            'progress_reward': 1.0,
            'time_penalty': -0.01
        }

        self.episode_results = []
        self.phase_timesteps = 0
        self.target_timesteps = 1_000_000

        self.logger.info("Phase 1 Trainer initialized")
        self.logger.info("Training scope: Single floor + static obstacles")
        self.logger.info(f"Target: {self.target_success_rate100} success rate")

    def train_phase(self) - Dict[str, Any]:

        self.logger.info("Starting Phase 1 training")
        phase_start = time.time()

        self._configure_phase1_environment()

        episode_count = 0
        recent_successes = deque(maxlen=50)

        while self.phase_timesteps  self.target_timesteps:
            episode_result = self._run_phase1_episode()
            episode_count += 1

            self.phase_timesteps += episode_result['episode_length']
            recent_successes.append(episode_result['success'])

            if self.agent.is_ready_for_update():
                self.agent.update_policy(
                    next_observation=self.environment.get_observation(),
                    next_done=False
                )

            if episode_count  100 == 0:
                current_success_rate = np.mean(recent_successes) if recent_successes else 0.0
                self.logger.info(f"Phase 1 Episode {episode_count}: "
                               f"Success rate: {current_success_rate100:.1f}, "
                               f"Timesteps: {self.phase_timesteps:,}/{self.target_timesteps:,}")

            if len(recent_successes) = 50:
                if np.mean(recent_successes) = self.target_success_rate:
                    self.logger.info("Phase 1 target achieved early!")
                    break

        phase_time = time.time() - phase_start
        final_success_rate = np.mean(recent_successes)  100 if recent_successes else 0.0

        final_evaluation = self._evaluate_phase1()

        results = {
            'phase': 'single_floor_static',
            'training_time': phase_time,
            'timesteps_trained': self.phase_timesteps,
            'episodes_trained': episode_count,
            'final_success_rate': final_success_rate,
            'final_energy': final_evaluation.get('mean_energy', 0),
            'final_collision_rate': final_evaluation.get('collision_rate', 0),
            'target_achieved': final_success_rate = self.target_success_rate  100
        }

        self.logger.info(f"Phase 1 completed in {phase_time/3600:.1f}h")
        self.logger.info(f"Final success rate: {final_success_rate:.1f}")

        return results

    def _configure_phase1_environment(self):

        env_config = {
            'building_floors': 1,
            'obstacle_density': self.obstacle_density,
            'dynamic_obstacles': False,
            'max_episode_steps': self.max_episode_length,
            'reward_weights': self.reward_weights,
            'goal_spawn_distance': (2.0, 8.0),
            'obstacle_complexity': 'simple'
        }

        if hasattr(self.environment, 'configure'):
            self.environment.configure(env_config)

    def _run_phase1_episode(self) - Dict[str, Any]:

        obs = self.environment.reset()

        episode_reward = 0.0
        episode_length = 0
        episode_energy = 0.0
        done = False

        while not done and episode_length  self.max_episode_length:
            action, log_prob = self.agent.select_action(obs)
            value = self.agent.evaluate_observation(obs)

            next_obs, reward, done, info = self.environment.step(action)

            self.agent.add_experience(obs, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            episode_energy += info.get('energy_consumption', 0.0)

            obs = next_obs

        final_pos = info.get('position', (0, 0, 0))
        goal_pos = info.get('goal_position', (0, 0, 0))
        final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
        success = final_distance = 0.5 and not info.get('collision', False)

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_energy': episode_energy,
            'success': success,
            'final_distance': final_distance
        }

    def _evaluate_phase1(self) - Dict[str, Any]:

        self.agent.policy.eval()

        eval_results = []
        for _ in range(20):
            obs = self.environment.reset()
            episode_energy = 0.0
            done = False

            while not done:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, info = self.environment.step(action)
                episode_energy += info.get('energy_consumption', 0.0)

            final_pos = info.get('position', (0, 0, 0))
            goal_pos = info.get('goal_position', (0, 0, 0))
            final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
            success = final_distance = 0.5 and not info.get('collision', False)

            eval_results.append({
                'success': success,
                'energy': episode_energy,
                'collision': info.get('collision', False)
            })

        self.agent.policy.train()

        return {
            'success_rate': np.mean([r['success'] for r in eval_results])  100,
            'mean_energy': np.mean([r['energy'] for r in eval_results if r['success']]),
            'collision_rate': np.mean([r['collision'] for r in eval_results])  100
        }

    def set_phase_steps(self, steps: int):

        self.target_timesteps = steps

    def set_starting_timestep(self, timestep: int):

        self.phase_timesteps = 0
