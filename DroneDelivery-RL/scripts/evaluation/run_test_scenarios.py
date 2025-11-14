import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.environment.airsim_env import AirSimEnvironment as DroneEnvironment
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.initialization import initialize_rl_system
from src.utils import setup_logging, load_config

class TestScenariosRunner:

    def __init__(self, config_path: str, model_path: str):
        self.config = load_config(config_path)
        self.model_path = Path(model_path)

        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)

        self.rl_system = initialize_rl_system(self.config.rl)
        self.agent = self.rl_system['agent']
        self._load_model()

        self.test_scenarios = {
            'nominal': {
                'description': 'Standard operating conditions',
                'episodes': 20,
                'config_overrides': {}
            },
            'high_obstacle_density': {
                'description': 'Dense obstacle environment (30 occupancy)',
                'episodes': 15,
                'config_overrides': {
                    'obstacle_density': 0.30,
                    'dynamic_obstacles': True,
                    'dynamic_obstacle_count': 8
                }
            },
            'long_distance': {
                'description': 'Long-range delivery (max building distance)',
                'episodes': 10,
                'config_overrides': {
                    'force_long_distance': True,
                    'min_goal_distance': 35.0
                }
            },
            'multi_floor_stress': {
                'description': 'Complex multi-floor navigation',
                'episodes': 15,
                'config_overrides': {
                    'force_multi_floor': True,
                    'min_floor_transitions': 3,
                    'complex_floor_layouts': True
                }
            },
            'low_battery': {
                'description': 'Low battery simulation',
                'episodes': 10,
                'config_overrides': {
                    'initial_battery': 0.3,
                    'energy_penalty_weight': 2.0
                }
            },
            'sensor_noise': {
                'description': 'High sensor noise conditions',
                'episodes': 15,
                'config_overrides': {
                    'slam_noise_factor': 3.0,
                    'imu_noise_factor': 2.0,
                    'localization_uncertainty': True
                }
            },
            'dynamic_environment': {
                'description': 'Highly dynamic environment',
                'episodes': 15,
                'config_overrides': {
                    'dynamic_obstacles': True,
                    'dynamic_obstacle_count': 10,
                    'obstacle_speed_range': [1.0, 3.0],
                    'unpredictable_movement': True
                }
            },
            'emergency_scenarios': {
                'description': 'Emergency landing and recovery',
                'episodes': 10,
                'config_overrides': {
                    'emergency_events': True,
                    'system_failures': True,
                    'recovery_testing': True
                }
            }
        }

        self.scenario_results = {}

        self.logger.info("Test Scenarios Runner initialized")
        self.logger.info(f"Configured {len(self.test_scenarios)} test scenarios")

    def _load_model(self):

        try:
            checkpoint = torch.load(self.model_path, map_location=self.agent.device)

            if 'agent_state_dict' in checkpoint:
                self.agent.policy.load_state_dict(checkpoint['agent_state_dict'])
            else:
                self.agent.policy.load_state_dict(checkpoint)

            self.agent.policy.eval()
            self.logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def run_all_scenarios(self) - Dict[str, Any]:

        self.logger.info("Starting comprehensive scenario testing")
        total_start = time.time()

        total_episodes = sum(scenario['episodes'] for scenario in self.test_scenarios.values())
        completed_episodes = 0

        for scenario_name, scenario_config in self.test_scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")

            scenario_results = self._run_scenario(scenario_name, scenario_config)
            self.scenario_results[scenario_name] = scenario_results

            completed_episodes += scenario_config['episodes']
            progress = (completed_episodes / total_episodes)  100

            self.logger.info(f"Scenario {scenario_name} completed: "
                           f"{scenario_results['success_rate']:.1f} success "
                           f"[{progress:.1f} total progress]")

        total_time = time.time() - total_start

        overall_analysis = self._analyze_scenario_results()

        final_results = {
            'testing_completed': True,
            'total_testing_time': total_time,
            'total_episodes': total_episodes,
            'scenarios_tested': len(self.test_scenarios),
            'scenario_results': self.scenario_results,
            'robustness_analysis': overall_analysis,
            'summary': self._generate_robustness_summary()
        }

        self.logger.info(f"All scenarios completed in {total_time/60:.1f} minutes")
        return final_results

    def _run_scenario(self, scenario_name: str, scenario_config: Dict[str, Any]) - Dict[str, Any]:

        num_episodes = scenario_config['episodes']
        config_overrides = scenario_config['config_overrides']

        scenario_env_config = self.config.environment.copy()
        scenario_env_config.update(config_overrides)
        scenario_environment = DroneEnvironment(scenario_env_config)

        episode_results = []

        for episode in range(num_episodes):
            episode_result = self._run_scenario_episode(scenario_environment)
            episode_results.append(episode_result)

        return self._aggregate_scenario_results(episode_results, scenario_name)

    def _run_scenario_episode(self, environment) - Dict[str, Any]:

        observation = environment.reset()

        total_energy = 0.0
        trajectory = []
        done = False
        steps = 0
        max_steps = 6000

        collision_occurred = False
        emergency_landing = False
        localization_loss = False

        while not done and steps  max_steps:
            try:
                action, _ = self.agent.select_action(observation, deterministic=True)

                observation, reward, done, info = environment.step(action)

                trajectory.append(info.get('position', [0, 0, 0]))
                total_energy += info.get('energy_consumption', 0.0)

                if info.get('collision', False):
                    collision_occurred = True

                if info.get('emergency_landing', False):
                    emergency_landing = True

                if info.get('localization_lost', False):
                    localization_loss = True

                steps += 1

            except Exception as e:
                self.logger.warning(f"Episode exception: {e}")
                done = True

        goal_position = environment.get_goal_position()
        final_position = trajectory[-1] if trajectory else [0, 0, 0]
        final_distance = np.linalg.norm(np.array(final_position) - np.array(goal_position))

        success = (final_distance = 0.5 and
                  not collision_occurred and
                  not emergency_landing and
                  not localization_loss)

        return {
            'success': success,
            'energy': total_energy,
            'flight_time': steps / 20.0,
            'collision': collision_occurred,
            'emergency_landing': emergency_landing,
            'localization_loss': localization_loss,
            'final_distance': final_distance,
            'trajectory_length': len(trajectory),
            'steps': steps
        }

    def _aggregate_scenario_results(self, episode_results: List[Dict[str, Any]],
                                   scenario_name: str) - Dict[str, Any]:

        successful_episodes = [ep for ep in episode_results if ep['success']]

        aggregated = {
            'scenario_name': scenario_name,
            'total_episodes': len(episode_results),
            'successful_episodes': len(successful_episodes),
            'success_rate': len(successful_episodes) / len(episode_results)  100,
            'collision_rate': np.mean([ep['collision'] for ep in episode_results])  100,
            'emergency_rate': np.mean([ep['emergency_landing'] for ep in episode_results])  100,
            'localization_loss_rate': np.mean([ep['localization_loss'] for ep in episode_results])  100
        }

        if successful_episodes:
            energies = [ep['energy'] for ep in successful_episodes]
            times = [ep['flight_time'] for ep in successful_episodes]

            aggregated.update({
                'mean_energy': float(np.mean(energies)),
                'std_energy': float(np.std(energies)),
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times))
            })
        else:
            aggregated.update({
                'mean_energy': 0.0, 'std_energy': 0.0,
                'mean_time': 0.0, 'std_time': 0.0
            })

        return aggregated

    def _analyze_scenario_results(self) - Dict[str, Any]:

        if not self.scenario_results:
            return {}

        success_rates = {name: results['success_rate']
                        for name, results in self.scenario_results.items()}

        best_scenario = max(success_rates.items(), key=lambda x: x[1])
        worst_scenario = min(success_rates.items(), key=lambda x: x[1])

        success_rate_variance = np.var(list(success_rates.values()))
        mean_success_rate = np.mean(list(success_rates.values()))

        robustness_score = mean_success_rate - (success_rate_variance  0.1)

        return {
            'overall_robustness_score': float(robustness_score),
            'mean_success_rate_across_scenarios': float(mean_success_rate),
            'success_rate_variance': float(success_rate_variance),
            'most_robust_scenario': best_scenario[0],
            'most_robust_success_rate': best_scenario[1],
            'least_robust_scenario': worst_scenario[0],
            'least_robust_success_rate': worst_scenario[1],
            'robustness_grade': self._grade_robustness(robustness_score)
        }

    def _grade_robustness(self, robustness_score: float) - str:

        if robustness_score = 90:
            return 'Excellent'
        elif robustness_score = 80:
            return 'Good'
        elif robustness_score = 70:
            return 'Fair'
        else:
            return 'Poor'

    def _generate_robustness_summary(self) - Dict[str, Any]:

        analysis = self._analyze_scenario_results()

        return {
            'overall_grade': analysis.get('robustness_grade', 'Unknown'),
            'stress_test_passed': analysis.get('mean_success_rate_across_scenarios', 0) = 80,
            'failure_modes_identified': self._identify_failure_modes(),
            'robustness_recommendations': self._generate_robustness_recommendations()
        }

    def _identify_failure_modes(self) - List[str]:

        failure_modes = []

        for scenario_name, results in self.scenario_results.items():
            if results['success_rate']  80:
                failure_modes.append(f"{scenario_name}: {results['success_rate']:.1f} success")

        return failure_modes

    def _generate_robustness_recommendations(self) - List[str]:

        recommendations = []
        analysis = self._analyze_scenario_results()

        if analysis.get('success_rate_variance', 0)  100:
            recommendations.append("Improve consistency across different scenarios")

        if self.scenario_results.get('sensor_noise', {}).get('success_rate', 100)  85:
            recommendations.append("Enhance robustness to sensor noise and uncertainty")

        if self.scenario_results.get('emergency_scenarios', {}).get('success_rate', 100)  90:
            recommendations.append("Improve emergency handling and recovery procedures")

        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive test scenarios')
    parser.add_argument('--config', type=str, default='config/evaluation_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='results/scenario_testing.json',
                       help='Output results file')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       help='Specific scenarios to run (default: all)')

    args = parser.parse_args()

    runner = TestScenariosRunner(args.config, args.model)

    if args.scenarios:
        runner.test_scenarios = {
            name: config for name, config in runner.test_scenarios.items()
            if name in args.scenarios
        }

    results = runner.run_all_scenarios()

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nTEST SCENARIOS SUMMARY")
    print("="  50)

    for scenario_name, scenario_result in results['scenario_results'].items():
        status = "PASS" if scenario_result['success_rate'] = 80 else "FAIL"
        print(f"{scenario_name:25}: {scenario_result['success_rate']:5.1f} [{status}]")

    print(f"\nOverall Robustness: {results['robustness_analysis']['robustness_grade']}")
    print(f"Mean Success Rate: {results['robustness_analysis']['mean_success_rate_across_scenarios']:.1f}")

if __name__ == "__main__":
    main()
