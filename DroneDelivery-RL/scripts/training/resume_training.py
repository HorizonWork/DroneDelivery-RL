import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.environment import DroneEnvironment
from src.rl import PPOAgent, CurriculumManager, initialize_rl_system
from src.rl.utils import CheckpointManager, ObservationNormalizer, TensorBoardLogger
from src.utils import setup_logging, load_config

class TrainingResumer:

    def __init__(self, config_path: str):
        self.config = load_config(config_path)

        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system['agent']
        self.curriculum_manager = self.rl_system['curriculum_manager']

        self.checkpoint_manager = CheckpointManager(self.config.rl.checkpoints)
        self.obs_normalizer = ObservationNormalizer(35, self.config.rl.normalization)
        self.tensorboard_logger = TensorBoardLogger(self.config.rl.logging)

        self.restored_state = {}

        self.logger.info("Training Resumer initialized")

    def resume_training(self, checkpoint_path: str,
                       target_timesteps: Optional[int] = None) - Dict[str, Any]:

        self.logger.info("=== RESUMING PPO TRAINING ===")
        self.logger.info(f"Checkpoint: {checkpoint_path}")

        if not self._load_checkpoint(checkpoint_path):
            raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")

        self._restore_training_state()

        results = self._continue_training(target_timesteps)

        self.logger.info("=== TRAINING RESUMED AND COMPLETED ===")
        return results

    def _load_checkpoint(self, checkpoint_path: str) - bool:

        try:
            checkpoint_file = Path(checkpoint_path)
            if not checkpoint_file.exists():
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            checkpoint_data = torch.load(checkpoint_path, map_location=self.agent.device)

            required_keys = ['agent_state_dict', 'training_info']
            if not all(key in checkpoint_data for key in required_keys):
                self.logger.error(f"Invalid checkpoint format. Required keys: {required_keys}")
                return False

            self.agent.policy.load_state_dict(checkpoint_data['agent_state_dict'])

            if 'optimizer_state_dict' in checkpoint_data:
                self.agent.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

            self.restored_state = {
                'training_info': checkpoint_data['training_info'],
                'agent_config': checkpoint_data.get('agent_config', {}),
                'additional_data': checkpoint_data.get('additional_data', {}),
                'checkpoint_metadata': checkpoint_data.get('checkpoint_metadata', {})
            }

            self.logger.info(" Checkpoint loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _restore_training_state(self):

        training_info = self.restored_state['training_info']
        additional_data = self.restored_state['additional_data']

        self.global_timestep = training_info.get('timestep', 0)
        self.episode_count = training_info.get('episode', 0)

        self.logger.info(f"Restored timestep: {self.global_timestep:,}")
        self.logger.info(f"Restored episode: {self.episode_count:,}")

        if 'curriculum_phase' in additional_data:
            phase_index = additional_data['curriculum_phase']
            self.curriculum_manager.current_phase_index = phase_index
            self.logger.info(f"Restored curriculum phase: {phase_index}")

        if 'normalization_stats' in additional_data:
            try:
                norm_stats = additional_data['normalization_stats']
                self.obs_normalizer.mean = norm_stats.get('mean', self.obs_normalizer.mean)
                self.obs_normalizer.variance = norm_stats.get('variance', self.obs_normalizer.variance)
                self.obs_normalizer.count = norm_stats.get('count', self.obs_normalizer.count)
                self.logger.info(" Normalization statistics restored")
            except Exception as e:
                self.logger.warning(f"Failed to restore normalization stats: {e}")

        if 'recent_rewards' in additional_data:
            self.episode_rewards = additional_data['recent_rewards']
            self.logger.info(f"Restored {len(self.episode_rewards)} recent rewards")

    def _continue_training(self, target_timesteps: Optional[int] = None) - Dict[str, Any]:

        if target_timesteps is None:
            target_timesteps = self.config.rl.training.get('total_timesteps', 5_000_000)

        remaining_timesteps = target_timesteps - self.global_timestep

        if remaining_timesteps = 0:
            self.logger.info("Training already completed!")
            return {
                'training_completed': True,
                'timesteps_trained': 0,
                'resumed_from': self.global_timestep
            }

        self.logger.info(f"Continuing training for {remaining_timesteps:,} more timesteps")

        current_phase = self.curriculum_manager.get_current_phase()
        environment = self._create_environment_for_phase(current_phase)

        training_start = time.time()
        episodes_trained = 0

        episode_rewards = getattr(self, 'episode_rewards', [])
        recent_performance = []

        while self.global_timestep  target_timesteps:
            episode_result = self._run_training_episode(environment)

            episode_rewards.append(episode_result['reward'])
            episodes_trained += 1

            if len(episode_rewards) = 20:
                recent_successes = sum(1 for r in episode_rewards[-20:] if r  400)
                success_rate = recent_successes / 20  100
                recent_performance.append(success_rate)

            if self.global_timestep  50_000 == 0:
                self._run_evaluation_and_checkpoint(environment)

            if episodes_trained  100 == 0:
                self._log_resume_progress(episodes_trained, recent_performance)

            if self.curriculum_manager.should_advance_phase(
                self.global_timestep, self.episode_count,
                {'success_rate': recent_performance[-1] if recent_performance else 0}
            ):
                self.logger.info("Phase advancement detected - updating environment")
                current_phase = self.curriculum_manager.get_current_phase()
                environment = self._create_environment_for_phase(current_phase)

        training_time = time.time() - training_start

        final_performance = self._run_final_evaluation(environment)
        final_checkpoint = self._save_final_checkpoint(final_performance)

        resume_results = {
            'training_completed': True,
            'resumed_from_timestep': self.global_timestep - remaining_timesteps,
            'final_timestep': self.global_timestep,
            'episodes_trained_in_resume': episodes_trained,
            'resume_training_time_hours': training_time / 3600,
            'final_performance': final_performance,
            'final_checkpoint': final_checkpoint,
            'curriculum_phases_completed': self.curriculum_manager.current_phase_index + 1
        }

        self.logger.info(f"Resume training completed in {training_time/3600:.1f} hours")
        self.logger.info(f"Final performance: {final_performance['success_rate']:.1f} success")

        return resume_results

    def _create_environment_for_phase(self, phase_config: Dict[str, Any]) - DroneEnvironment:

        env_config = self.config.environment.copy()

        if 'config' in phase_config:
            phase_settings = phase_config['config']

            if 'floors' in phase_settings:
                env_config['building']['floors'] = phase_settings['floors']

            if 'obstacle_density' in phase_settings:
                env_config['obstacles']['density'] = phase_settings['obstacle_density']

            if 'dynamic_obstacles' in phase_settings:
                env_config['obstacles']['dynamic_obstacles'] = phase_settings['dynamic_obstacles']

        return DroneEnvironment(env_config)

    def _run_training_episode(self, environment: DroneEnvironment) - Dict[str, Any]:

        observation = environment.reset()
        observation = self.obs_normalizer.normalize(observation)

        episode_reward = 0.0
        episode_energy = 0.0
        episode_steps = 0
        done = False

        rollout = {'observations': [], 'actions': [], 'rewards': [], 'values': [], 'log_probs': [], 'dones': []}

        while not done and episode_steps  self.agent.rollout_length:
            action, log_prob, value = self.agent.select_action(observation, training=True)
            next_observation, reward, done, info = environment.step(action)
            next_observation = self.obs_normalizer.normalize(next_observation)

            rollout['observations'].append(observation)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['values'].append(value)
            rollout['log_probs'].append(log_prob)
            rollout['dones'].append(done)

            episode_reward += reward
            episode_energy += info.get('energy_consumption', 0.0)
            episode_steps += 1
            self.global_timestep += 1

            observation = next_observation

        final_value = self.agent.get_value(observation) if not done else 0.0
        training_metrics = self.agent.update_policy(rollout, final_value)

        self.episode_count += 1

        return {
            'reward': episode_reward,
            'energy': episode_energy,
            'steps': episode_steps,
            'success': info.get('success', False),
            'training_metrics': training_metrics
        }

    def _run_evaluation_and_checkpoint(self, environment: DroneEnvironment):

        eval_results = self._quick_evaluation(environment)

        self.checkpoint_manager.save_checkpoint(
            self.agent,
            self.global_timestep,
            self.episode_count,
            eval_results['success_rate'],
            eval_results['mean_energy'],
            additional_data={
                'curriculum_phase': self.curriculum_manager.current_phase_index,
                'phase_name': self.curriculum_manager.get_current_phase()['name'],
                'normalization_stats': {
                    'mean': self.obs_normalizer.mean.tolist(),
                    'variance': self.obs_normalizer.variance.tolist(),
                    'count': self.obs_normalizer.count
                }
            }
        )

        self.logger.info(f"Checkpoint saved at timestep {self.global_timestep:,}")

    def _quick_evaluation(self, environment: DroneEnvironment, num_episodes: int = 10) - Dict[str, Any]:

        eval_results = []

        for _ in range(num_episodes):
            observation = environment.reset()
            observation = self.obs_normalizer.normalize(observation)

            episode_reward = 0.0
            episode_energy = 0.0
            done = False
            steps = 0

            while not done and steps  500:
                action, _, _ = self.agent.select_action(observation, deterministic=True)
                observation, reward, done, info = environment.step(action)
                observation = self.obs_normalizer.normalize(observation)

                episode_reward += reward
                episode_energy += info.get('energy_consumption', 0.0)
                steps += 1

            eval_results.append({
                'reward': episode_reward,
                'energy': episode_energy,
                'success': info.get('success', False)
            })

        successes = [r for r in eval_results if r['success']]

        return {
            'success_rate': len(successes) / len(eval_results)  100,
            'mean_energy': sum(r['energy'] for r in successes) / len(successes) if successes else 0
        }

    def _run_final_evaluation(self, environment: DroneEnvironment) - Dict[str, Any]:

        return self._quick_evaluation(environment, num_episodes=50)

    def _save_final_checkpoint(self, performance: Dict[str, Any]) - str:

        return self.checkpoint_manager.save_checkpoint(
            self.agent,
            self.global_timestep,
            self.episode_count,
            performance['success_rate'],
            performance['mean_energy'],
            additional_data={
                'training_completed': True,
                'resume_completed': True,
                'final_evaluation': performance
            }
        )

    def _log_resume_progress(self, episodes_trained: int, recent_performance: list):

        success_rate = recent_performance[-1] if recent_performance else 0
        progress = self.global_timestep / 5_000_000  100

        self.logger.info(f"Resume Progress: {progress:.1f} ({self.global_timestep:,}/5,000,000)")
        self.logger.info(f"Episodes trained in resume: {episodes_trained:,}")
        self.logger.info(f"Recent success rate: {success_rate:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Resume PPO training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config/main_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override target timesteps')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f" Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    resumer = TrainingResumer(args.config)

    results = resumer.resume_training(args.checkpoint, args.timesteps)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_file = output_dir / 'resume_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f" Resume results saved to {results_file}")

    print("\n" + "="60)
    print(" TRAINING RESUME COMPLETED!")
    print("="60)
    print(f"Resumed from: {results['resumed_from_timestep']:,} timesteps")
    print(f"Final timestep: {results['final_timestep']:,}")
    print(f"Episodes trained: {results['episodes_trained_in_resume']:,}")
    print(f"Resume time: {results['resume_training_time_hours']:.1f} hours")
    print(f"Final success rate: {results['final_performance']['success_rate']:.1f}")
    print(f"Final checkpoint: {Path(results['final_checkpoint']).name}")
    print("="60)

if __name__ == "__main__":
    main()
