"""
Random Baseline Evaluator
Evaluates random exploration performance as lower bound baseline.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import time
from dataclasses import dataclass
import json

# Import shared data structures
from ..astar_baseline.evaluator import EpisodeResult, EvaluationMetrics

class RandomEvaluator:
    """
    Evaluates random exploration baseline performance.
    Provides lower bound for comparison with intelligent methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.episode_results: List[EpisodeResult] = []
        
        # Evaluation parameters
        self.num_episodes = config.get('num_episodes', 200)
        self.max_episode_time = config.get('max_episode_time', 300.0)  # seconds
        self.goal_tolerance = config.get('goal_tolerance', 0.5)        # meters
        
        # Energy calculation parameters
        self.control_dt = config.get('control_dt', 0.05)               # seconds
        self.drone_mass = config.get('drone_mass', 1.5)                # kg
        
    def evaluate_episode(self, random_agent, environment) -> EpisodeResult:
        """
        Evaluate single episode with random exploration.
        
        Args:
            random_agent: Random exploration agent
            environment: Simulation environment
            
        Returns:
            EpisodeResult with performance metrics
        """
        result = EpisodeResult()
        
        # Reset environment and agent
        obs = environment.reset()
        random_agent.reset()
        
        start_time = time.time()
        total_energy = 0.0
        path_length = 0.0
        previous_pos = None
        
        # Get start and goal positions
        current_pos = environment.get_drone_position()
        goal_pos = environment.get_goal_position()
        random_agent.set_goal(goal_pos)
        
        # Episode loop
        step = 0
        max_steps = int(self.max_episode_time / self.control_dt)
        
        while step < max_steps:
            # Get current state
            current_pos = environment.get_drone_position()
            current_yaw = environment.get_drone_yaw()
            obstacles = environment.get_obstacles()
            
            # Check goal reached
            distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(goal_pos))
            if distance_to_goal < self.goal_tolerance:
                result.success = True
                break
            
            # Check collision
            if environment.check_collision():
                result.collision = True
                break
            
            # Get random action
            vx, vy, vz, yaw_rate = random_agent.get_action(
                current_pos, current_yaw, obstacles
            )
            
            # Execute action
            action = np.array([vx, vy, vz, yaw_rate])
            obs, reward, done, info = environment.step(action)
            
            # Track energy consumption (simplified model)
            velocity_magnitude = np.linalg.norm([vx, vy, vz])
            # Energy ∝ thrust² ∝ (acceleration + gravity_compensation)²
            thrust_estimate = self.drone_mass * (velocity_magnitude + 9.81)
            energy_step = (thrust_estimate ** 2) * self.control_dt
            total_energy += energy_step
            
            # Track path length
            if previous_pos is not None:
                path_length += np.linalg.norm(np.array(current_pos) - np.array(previous_pos))
            previous_pos = current_pos
            
            step += 1
        
        # Calculate final metrics
        result.flight_time = time.time() - start_time
        result.energy_consumed = total_energy
        result.path_length = path_length
        result.final_distance_to_goal = distance_to_goal
        result.ate_error = environment.get_ate_error() if hasattr(environment, 'get_ate_error') else 0.0
        
        if step >= max_steps and not result.success and not result.collision:
            result.timeout = True
        
        return result
    
    def evaluate_multiple_episodes(self, random_agent, environment, 
                                 num_episodes: int = None) -> EvaluationMetrics:
        """
        Evaluate multiple episodes and compute aggregate metrics.
        
        Args:
            random_agent: Random exploration agent
            environment: Simulation environment
            num_episodes: Number of episodes (default: from config)
            
        Returns:
            EvaluationMetrics with aggregated results
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        
        self.episode_results = []
        
        print(f"Evaluating Random baseline over {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            if episode % 50 == 0:
                print(f"Episode {episode}/{num_episodes}")
            
            result = self.evaluate_episode(random_agent, environment)
            self.episode_results.append(result)
        
        return self._compute_metrics()
    
    def _compute_metrics(self) -> EvaluationMetrics:
        """Compute aggregate metrics from episode results."""
        if not self.episode_results:
            return EvaluationMetrics()
        
        metrics = EvaluationMetrics()
        metrics.episodes_completed = len(self.episode_results)
        
        # Set target values for random baseline (lower bound expectations)
        metrics.target_success_rate = 5.0       # Very low success rate expected
        metrics.target_energy_mean = 2000.0     # High energy due to inefficiency
        metrics.target_time_mean = 280.0        # Close to timeout usually
        metrics.target_collision_rate = 15.0    # High collision rate
        metrics.target_ate_mean = 0.20          # High trajectory error
        
        # Success and failure rates
        successes = [r for r in self.episode_results if r.success]
        collisions = [r for r in self.episode_results if r.collision]
        timeouts = [r for r in self.episode_results if r.timeout]
        
        metrics.success_rate = len(successes) / len(self.episode_results) * 100
        metrics.collision_rate = len(collisions) / len(self.episode_results) * 100
        metrics.timeout_rate = len(timeouts) / len(self.episode_results) * 100
        
        # Energy metrics (for all episodes, since most won't succeed)
        all_energies = [r.energy_consumed for r in self.episode_results]
        if all_energies:
            metrics.mean_energy = np.mean(all_energies)
            metrics.std_energy = np.std(all_energies)
        
        # Time metrics (for all episodes)
        all_times = [r.flight_time for r in self.episode_results]
        if all_times:
            metrics.mean_time = np.mean(all_times)
            metrics.std_time = np.std(all_times)
        
        # ATE metrics
        ate_errors = [r.ate_error for r in self.episode_results if r.ate_error > 0]
        if ate_errors:
            metrics.mean_ate = np.mean(ate_errors)
            metrics.std_ate = np.std(ate_errors)
        
        return metrics
    
    def print_results(self, metrics: EvaluationMetrics):
        """Print evaluation results in Table 3 format."""
        print("\n" + "="*60)
        print("RANDOM EXPLORATION BASELINE EVALUATION RESULTS")
        print("="*60)
        
        print(f"Episodes completed: {metrics.episodes_completed}")
        print(f"Success rate: {metrics.success_rate:.1f}% (expected: ~{metrics.target_success_rate:.1f}%)")
        print(f"Collision rate: {metrics.collision_rate:.1f}% (expected: ~{metrics.target_collision_rate:.1f}%)")
        print(f"Timeout rate: {metrics.timeout_rate:.1f}%")
        
        if metrics.mean_energy > 0:
            print(f"Energy consumption: {metrics.mean_energy:.0f}±{metrics.std_energy:.0f} J "
                  f"(expected: ~{metrics.target_energy_mean:.0f} J)")
        
        if metrics.mean_time > 0:
            print(f"Flight time: {metrics.mean_time:.0f}±{metrics.std_time:.0f} s "
                  f"(expected: ~{metrics.target_time_mean:.0f} s)")
        
        if metrics.mean_ate > 0:
            print(f"ATE error: {metrics.mean_ate:.3f}±{metrics.std_ate:.3f} m "
                  f"(expected: ~{metrics.target_ate_mean:.3f} m)")
        
        print("\nNote: Random baseline provides lower bound for comparison.")
        print("Low performance is expected and indicates proper baseline functioning.")
