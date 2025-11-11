"""
Phase 2 Trainer  
Two floors + dynamic obstacles training.
Intermediate complexity with multi-floor navigation.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

class Phase2Trainer:
    """
    Phase 2: Two-floor navigation with dynamic obstacles.
    Target: 90% success rate, improved energy efficiency.
    """
    
    def __init__(self, agent, environment, config: Dict[str, Any]):
        self.agent = agent
        self.environment = environment  # Two-floor environment
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Phase 2 specific parameters
        self.target_success_rate = 0.90           # 90% to advance
        self.max_episode_length = 1500            # Longer episodes for multi-floor
        self.obstacle_density = 0.15              # Increased obstacle density
        self.dynamic_obstacle_count = 3           # 3 dynamic obstacles
        
        # Enhanced reward structure
        self.reward_weights = {
            'goal_reached': 200.0,                # Higher goal reward
            'energy_penalty': -0.3,               # Increased energy awareness
            'collision_penalty': -100.0,          # Stronger collision penalty
            'progress_reward': 2.0,               # Higher progress reward
            'time_penalty': -0.02,                # Moderate time pressure
            'floor_transition_bonus': 10.0,       # Bonus for successful floor changes
            'dynamic_avoidance_bonus': 5.0        # Bonus for avoiding dynamic obstacles
        }
        
        # Multi-floor navigation parameters
        self.floor_transition_tolerance = 1.0     # meters
        self.vertical_navigation_bonus = 15.0     # Bonus for efficient vertical movement
        
        # Dynamic obstacle prediction parameters
        self.prediction_horizon = 2.0             # seconds
        self.obstacle_speed_range = (0.5, 2.0)   # m/s dynamic obstacle speeds
        
        # Training tracking
        self.phase_timesteps = 0
        self.target_timesteps = 2_000_000         # 2M steps
        
        self.logger.info("Phase 2 Trainer initialized")
        self.logger.info("Training scope: Two floors + dynamic obstacles")
        self.logger.info(f"Target: {self.target_success_rate*100}% success rate")
        self.logger.info(f"Dynamic obstacles: {self.dynamic_obstacle_count}")
    
    def train_phase(self) -> Dict[str, Any]:
        """
        Execute Phase 2 training.
        
        Returns:
            Phase 2 training results
        """
        self.logger.info("Starting Phase 2 training")
        phase_start = time.time()
        
        # Configure environment for Phase 2
        self._configure_phase2_environment()
        
        episode_count = 0
        recent_successes = deque(maxlen=50)
        recent_energies = deque(maxlen=50)
        
        # Curriculum within Phase 2
        sub_phase_progress = 0.0
        
        while self.phase_timesteps < self.target_timesteps:
            # Gradually increase difficulty within Phase 2
            self._adjust_phase2_difficulty(sub_phase_progress)
            
            # Run training episode
            episode_result = self._run_phase2_episode()
            episode_count += 1
            
            self.phase_timesteps += episode_result['episode_length']
            recent_successes.append(episode_result['success'])
            
            if episode_result['success']:
                recent_energies.append(episode_result['episode_energy'])
            
            # Update agent
            if self.agent.is_ready_for_update():
                self.agent.update_policy(
                    next_observation=self.environment.get_observation(),
                    next_done=False
                )
            
            # Update sub-phase progress
            sub_phase_progress = self.phase_timesteps / self.target_timesteps
            
            # Log progress
            if episode_count % 100 == 0:
                current_success_rate = np.mean(recent_successes) if recent_successes else 0.0
                avg_energy = np.mean(recent_energies) if recent_energies else 0.0
                
                self.logger.info(f"Phase 2 Episode {episode_count}: "
                               f"Success: {current_success_rate*100:.1f}%, "
                               f"Energy: {avg_energy:.0f}J, "
                               f"Progress: {sub_phase_progress*100:.1f}%")
            
            # Check early phase completion
            if (len(recent_successes) >= 50 and 
                np.mean(recent_successes) >= self.target_success_rate and
                sub_phase_progress > 0.5):  # Must complete at least 50% of phase
                self.logger.info("Phase 2 target achieved early!")
                break
        
        phase_time = time.time() - phase_start
        final_success_rate = np.mean(recent_successes) * 100 if recent_successes else 0.0
        final_energy = np.mean(recent_energies) if recent_energies else 0.0
        
        # Final Phase 2 evaluation
        final_evaluation = self._evaluate_phase2()
        
        results = {
            'phase': 'two_floors_dynamic',
            'training_time': phase_time,
            'timesteps_trained': self.phase_timesteps,
            'episodes_trained': episode_count,
            'final_success_rate': final_success_rate,
            'final_energy': final_energy,
            'final_collision_rate': final_evaluation.get('collision_rate', 0),
            'multi_floor_success_rate': final_evaluation.get('multi_floor_success', 0),
            'dynamic_avoidance_rate': final_evaluation.get('dynamic_avoidance', 0),
            'target_achieved': final_success_rate >= self.target_success_rate * 100
        }
        
        self.logger.info(f"Phase 2 completed: {final_success_rate:.1f}% success, {final_energy:.0f}J energy")
        
        return results
    
    def _configure_phase2_environment(self):
        """Configure environment for Phase 2."""
        env_config = {
            'building_floors': 2,                         # Two floors
            'obstacle_density': self.obstacle_density,
            'dynamic_obstacles': True,                    # Enable dynamic obstacles
            'dynamic_obstacle_count': self.dynamic_obstacle_count,
            'obstacle_speed_range': self.obstacle_speed_range,
            'max_episode_steps': self.max_episode_length,
            'reward_weights': self.reward_weights,
            'goal_spawn_floors': [1, 2],                 # Goals on both floors
            'start_spawn_floors': [1, 2],                # Start on both floors
            'floor_transition_enabled': True,
            'staircase_positions': [(3, 3), (17, 3)],   # Staircase locations
            'elevator_positions': [(3, 37), (17, 37)]   # Elevator locations
        }
        
        if hasattr(self.environment, 'configure'):
            self.environment.configure(env_config)
    
    def _adjust_phase2_difficulty(self, progress: float):
        """
        Gradually increase difficulty within Phase 2.
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        # Increase obstacle density
        current_density = self.obstacle_density + (progress * 0.05)  # Up to 20% density
        
        # Increase dynamic obstacle count
        current_dynamic_count = min(5, int(self.dynamic_obstacle_count + progress * 2))
        
        # Increase obstacle speed
        max_speed = self.obstacle_speed_range[1] + (progress * 0.5)  # Up to 2.5 m/s
        
        # Update environment difficulty
        difficulty_config = {
            'obstacle_density': current_density,
            'dynamic_obstacle_count': current_dynamic_count,
            'obstacle_speed_range': (self.obstacle_speed_range[0], max_speed),
            'goal_spawn_distance': (3.0 + progress * 5.0, 15.0 + progress * 5.0)  # Further goals
        }
        
        if hasattr(self.environment, 'update_difficulty'):
            self.environment.update_difficulty(difficulty_config)
    
    def _run_phase2_episode(self) -> Dict[str, Any]:
        """Run single Phase 2 episode with multi-floor challenges."""
        obs = self.environment.reset()
        
        episode_reward = 0.0
        episode_length = 0
        episode_energy = 0.0
        floor_transitions = 0
        dynamic_avoidances = 0
        
        done = False
        previous_floor = self._get_current_floor(obs)
        
        while not done and episode_length < self.max_episode_length:
            # Agent action
            action, log_prob = self.agent.select_action(obs)
            value = self.agent.evaluate_observation(obs)
            
            # Environment step
            next_obs, reward, done, info = self.environment.step(action)
            
            # Enhanced reward shaping for Phase 2
            enhanced_reward = self._calculate_phase2_reward(reward, info, obs, next_obs)
            
            # Record experience with enhanced reward
            self.agent.add_experience(obs, action, enhanced_reward, value, log_prob, done)
            
            # Track Phase 2 specific metrics
            current_floor = self._get_current_floor(next_obs)
            if current_floor != previous_floor:
                floor_transitions += 1
                previous_floor = current_floor
            
            if info.get('dynamic_obstacle_avoided', False):
                dynamic_avoidances += 1
            
            # Update metrics
            episode_reward += enhanced_reward
            episode_length += 1
            episode_energy += info.get('energy_consumption', 0.0)
            
            obs = next_obs
        
        # Success determination
        final_pos = info.get('position', (0, 0, 0))
        goal_pos = info.get('goal_position', (0, 0, 0))
        final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
        success = final_distance <= 0.5 and not info.get('collision', False)
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'episode_energy': episode_energy,
            'success': success,
            'floor_transitions': floor_transitions,
            'dynamic_avoidances': dynamic_avoidances,
            'final_distance': final_distance
        }
    
    def _calculate_phase2_reward(self, base_reward: float, info: Dict[str, Any],
                               obs: np.ndarray, next_obs: np.ndarray) -> float:
        """Calculate Phase 2 specific reward with multi-floor bonuses."""
        enhanced_reward = base_reward
        
        # Floor transition bonus
        current_floor = self._get_current_floor(obs)
        next_floor = self._get_current_floor(next_obs)
        
        if next_floor != current_floor:
            # Check if transition brings closer to goal
            goal_floor = self._get_goal_floor(info.get('goal_position', (0, 0, 0)))
            
            if abs(next_floor - goal_floor) < abs(current_floor - goal_floor):
                enhanced_reward += self.reward_weights['floor_transition_bonus']
        
        # Dynamic obstacle avoidance bonus
        if info.get('dynamic_obstacle_avoided', False):
            enhanced_reward += self.reward_weights['dynamic_avoidance_bonus']
        
        # Energy efficiency bonus (comparative to Phase 1 baseline)
        energy_consumption = info.get('energy_consumption', 0.0)
        if energy_consumption < 2.0:  # Efficient step
            enhanced_reward += 0.5
        
        return enhanced_reward
    
    def _get_current_floor(self, observation: np.ndarray) -> int:
        """Extract current floor from observation."""
        # Assume position is in observation[0:3]
        if len(observation) >= 3:
            altitude = observation[2]
            floor = max(1, min(2, int(altitude // 3.0) + 1))  # Floors 1-2
            return floor
        return 1
    
    def _get_goal_floor(self, goal_position: Tuple[float, float, float]) -> int:
        """Get floor number for goal position."""
        return max(1, min(2, int(goal_position[2] // 3.0) + 1))
    
    def _evaluate_phase2(self) -> Dict[str, Any]:
        """Comprehensive Phase 2 evaluation."""
        self.agent.policy.eval()
        
        eval_results = []
        multi_floor_episodes = 0
        successful_multi_floor = 0
        
        for _ in range(30):  # More episodes for multi-floor evaluation
            obs = self.environment.reset()
            episode_energy = 0.0
            floor_transitions = 0
            dynamic_avoidances = 0
            done = False
            
            start_floor = self._get_current_floor(obs)
            goal_floor = self._get_goal_floor(self.environment.get_goal_position())
            
            if start_floor != goal_floor:
                multi_floor_episodes += 1
            
            previous_floor = start_floor
            
            while not done:
                action, _ = self.agent.select_action(obs, deterministic=True)
                obs, reward, done, info = self.environment.step(action)
                
                episode_energy += info.get('energy_consumption', 0.0)
                
                current_floor = self._get_current_floor(obs)
                if current_floor != previous_floor:
                    floor_transitions += 1
                    previous_floor = current_floor
                
                if info.get('dynamic_obstacle_avoided', False):
                    dynamic_avoidances += 1
            
            final_pos = info.get('position', (0, 0, 0))
            goal_pos = info.get('goal_position', (0, 0, 0))
            final_distance = np.linalg.norm(np.array(final_pos) - np.array(goal_pos))
            success = final_distance <= 0.5 and not info.get('collision', False)
            
            if start_floor != goal_floor and success:
                successful_multi_floor += 1
            
            eval_results.append({
                'success': success,
                'energy': episode_energy,
                'collision': info.get('collision', False),
                'floor_transitions': floor_transitions,
                'dynamic_avoidances': dynamic_avoidances,
                'multi_floor': start_floor != goal_floor
            })
        
        self.agent.policy.train()
        
        # Multi-floor success rate
        multi_floor_success_rate = (successful_multi_floor / max(1, multi_floor_episodes)) * 100
        
        return {
            'success_rate': np.mean([r['success'] for r in eval_results]) * 100,
            'mean_energy': np.mean([r['energy'] for r in eval_results if r['success']]),
            'collision_rate': np.mean([r['collision'] for r in eval_results]) * 100,
            'multi_floor_success': multi_floor_success_rate,
            'avg_floor_transitions': np.mean([r['floor_transitions'] for r in eval_results]),
            'dynamic_avoidance': np.mean([r['dynamic_avoidances'] for r in eval_results if r['success']]),
            'multi_floor_episodes': multi_floor_episodes
        }
    
    def set_phase_steps(self, steps: int):
        """Set target timesteps for phase."""
        self.target_timesteps = steps
    
    def set_starting_timestep(self, timestep: int):
        """Set starting timestep offset."""
        self.phase_timesteps = 0
