#!/usr/bin/env python3
"""
Hyperparameter Search Script
Automated hyperparameter optimization for PPO training.
Uses Optuna for efficient hyperparameter search with pruning.
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from environment import DroneEnvironment
from rl import PPOAgent, initialize_rl_system
from utils import setup_logging, load_config

class HyperparameterSearcher:
    """
    Automated hyperparameter optimization for PPO.
    Optimizes learning rate, batch size, entropy coefficient, etc.
    """
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Setup logging
        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)
        
        # Search configuration
        self.search_config = {
            'n_trials': 50,
            'trial_timesteps': 500_000,  # 500k timesteps per trial
            'evaluation_episodes': 20,
            'pruning_enabled': True,
            'timeout_hours': 24  # Max 24 hours total search
        }
        
        # Parameter search spaces
        self.search_space = {
            'learning_rate': (1e-5, 1e-3),
            'batch_size': [64, 128, 256, 512],
            'rollout_length': [1024, 2048, 4096],
            'entropy_coef': (0.001, 0.1),
            'value_loss_coef': (0.25, 1.0),
            'clip_range': (0.1, 0.3),
            'gamma': (0.95, 0.999),
            'gae_lambda': (0.9, 0.98)
        }
        
        # Best parameters tracking
        self.best_params = None
        self.best_score = -float('inf')
        self.trial_results = []
        
        self.logger.info("Hyperparameter Searcher initialized")
        self.logger.info(f"Search space: {self.search_space}")
    
    def search_hyperparameters(self, output_dir: str) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Search results with best parameters
        """
        self.logger.info("=== STARTING HYPERPARAMETER SEARCH ===")
        search_start = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10) if self.search_config['pruning_enabled'] else None,
            sampler=TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(
            lambda trial: self._objective_function(trial),
            n_trials=self.search_config['n_trials'],
            timeout=self.search_config['timeout_hours'] * 3600
        )
        
        search_time = time.time() - search_start
        
        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Compile results
        search_results = {
            'search_completed': True,
            'search_time_hours': search_time / 3600,
            'n_trials_completed': len(study.trials),
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'trial_results': self.trial_results,
            'study_statistics': {
                'best_trial_number': study.best_trial.number,
                'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }
        
        # Save results
        self._save_search_results(search_results, output_path)
        
        self.logger.info("=== HYPERPARAMETER SEARCH COMPLETED ===")
        return search_results
    
    def _objective_function(self, trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function - returns score to maximize.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Performance score (higher is better)
        """
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial)
        
        self.logger.info(f"Trial {trial.number}: Testing parameters {params}")
        
        try:
            # Train with sampled parameters
            score = self._train_and_evaluate(params, trial)
            
            # Store trial result
            self.trial_results.append({
                'trial_number': trial.number,
                'parameters': params,
                'score': score,
                'completed': True
            })
            
            self.logger.info(f"Trial {trial.number} completed: Score = {score:.3f}")
            
            return score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            
            # Store failed trial
            self.trial_results.append({
                'trial_number': trial.number,
                'parameters': params,
                'score': -1000.0,  # Very low score for failed trials
                'completed': False,
                'error': str(e)
            })
            
            return -1000.0
    
    def _sample_hyperparameters(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}
        
        # Learning rate (log scale)
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 
            self.search_space['learning_rate'][0], 
            self.search_space['learning_rate'][1], 
            log=True
        )
        
        # Batch size (categorical)
        params['batch_size'] = trial.suggest_categorical('batch_size', self.search_space['batch_size'])
        
        # Rollout length (categorical)
        params['rollout_length'] = trial.suggest_categorical('rollout_length', self.search_space['rollout_length'])
        
        # Entropy coefficient (log scale)
        params['entropy_coef'] = trial.suggest_float(
            'entropy_coef',
            self.search_space['entropy_coef'][0],
            self.search_space['entropy_coef'][1],
            log=True
        )
        
        # Value loss coefficient
        params['value_loss_coef'] = trial.suggest_float(
            'value_loss_coef',
            self.search_space['value_loss_coef'][0],
            self.search_space['value_loss_coef'][1]
        )
        
        # Clip range
        params['clip_range'] = trial.suggest_float(
            'clip_range',
            self.search_space['clip_range'][0],
            self.search_space['clip_range'][1]
        )
        
        # Gamma (discount factor)
        params['gamma'] = trial.suggest_float(
            'gamma',
            self.search_space['gamma'][0],
            self.search_space['gamma'][1]
        )
        
        # GAE lambda
        params['gae_lambda'] = trial.suggest_float(
            'gae_lambda',
            self.search_space['gae_lambda'][0],
            self.search_space['gae_lambda'][1]
        )
        
        return params
    
    def _train_and_evaluate(self, params: Dict[str, Any], trial: optuna.trial.Trial) -> float:
        """
        Train agent with given parameters and evaluate performance.
        
        Args:
            params: Hyperparameters to test
            trial: Optuna trial for pruning
            
        Returns:
            Performance score
        """
        # Update config with sampled parameters
        trial_config = self.config.copy()
        trial_config['rl']['ppo'].update(params)
        
        # Initialize environment and agent
        environment = DroneEnvironment(trial_config['environment'])
        
        # Initialize RL system with trial parameters
        rl_system = initialize_rl_system(trial_config['rl'])
        agent = rl_system['agent']
        
        # Training loop
        timesteps = 0
        episode_rewards = []
        success_rates = []
        target_timesteps = self.search_config['trial_timesteps']
        
        # Intermediate evaluation intervals
        eval_interval = target_timesteps // 10  # 10 evaluations per trial
        
        while timesteps < target_timesteps:
            # Run training episode
            observation = environment.reset()
            episode_reward = 0.0
            done = False
            episode_steps = 0
            
            while not done and episode_steps < 1000:
                action, _, _ = agent.select_action(observation, training=True)
                observation, reward, done, info = environment.step(action)
                
                episode_reward += reward
                episode_steps += 1
                timesteps += 1
            
            episode_rewards.append(episode_reward)
            
            # Intermediate evaluation for pruning
            if timesteps % eval_interval == 0:
                # Quick evaluation
                eval_score = self._quick_evaluation(agent, environment)
                success_rates.append(eval_score)
                
                # Report to Optuna for pruning
                trial.report(eval_score, timesteps // eval_interval)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        # Final evaluation
        final_score = self._evaluate_agent(agent, environment)
        
        return final_score
    
    def _quick_evaluation(self, agent: PPOAgent, environment: DroneEnvironment, 
                         num_episodes: int = 5) -> float:
        """Quick evaluation for intermediate pruning."""
        scores = []
        
        for _ in range(num_episodes):
            observation = environment.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < 500:  # Shorter episodes for speed
                action, _, _ = agent.select_action(observation, deterministic=True)
                observation, reward, done, info = environment.step(action)
                episode_reward += reward
                steps += 1
            
            # Convert to success-based score
            success_score = 1.0 if episode_reward > 400 else 0.0
            scores.append(success_score)
        
        return np.mean(scores) * 100  # Return as percentage
    
    def _evaluate_agent(self, agent: PPOAgent, environment: DroneEnvironment) -> float:
        """
        Comprehensive agent evaluation.
        
        Returns:
            Combined score considering success rate, energy efficiency, and safety
        """
        eval_results = []
        num_episodes = self.search_config['evaluation_episodes']
        
        for episode in range(num_episodes):
            observation = environment.reset()
            
            episode_reward = 0.0
            episode_energy = 0.0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action, _, _ = agent.select_action(observation, deterministic=True)
                observation, reward, done, info = environment.step(action)
                
                episode_reward += reward
                episode_energy += info.get('energy_consumption', 0.0)
                steps += 1
            
            eval_results.append({
                'reward': episode_reward,
                'energy': episode_energy,
                'success': info.get('success', False),
                'collision': info.get('collision', False),
                'steps': steps
            })
        
        # Calculate metrics
        successes = [r for r in eval_results if r['success']]
        success_rate = len(successes) / len(eval_results) * 100
        
        # Energy efficiency (only for successful episodes)
        if successes:
            avg_energy = np.mean([r['energy'] for r in successes])
            # Normalize energy (lower is better, so invert)
            energy_score = max(0, 100 - (avg_energy / 30.0))  # Scale assuming ~3000J baseline
        else:
            energy_score = 0
        
        # Safety score (collision rate)
        collision_rate = np.mean([r['collision'] for r in eval_results]) * 100
        safety_score = max(0, 100 - collision_rate)
        
        # Combined score (weighted)
        combined_score = (
            success_rate * 0.6 +      # 60% weight on success
            energy_score * 0.3 +      # 30% weight on energy efficiency  
            safety_score * 0.1        # 10% weight on safety
        )
        
        return combined_score
    
    def _save_search_results(self, results: Dict[str, Any], output_path: Path):
        """Save hyperparameter search results."""
        # Save complete results
        results_file = output_path / 'hyperparameter_search_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save best parameters for easy use
        best_params_file = output_path / 'best_hyperparameters.json'
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_parameters': results['best_parameters'],
                'best_score': results['best_score'],
                'search_completed': results['search_completed']
            }, f, indent=2)
        
        # Save parameter importance analysis
        if len(self.trial_results) > 10:
            importance_analysis = self._analyze_parameter_importance()
            importance_file = output_path / 'parameter_importance.json'
            with open(importance_file, 'w') as f:
                json.dump(importance_analysis, f, indent=2)
        
        self.logger.info(f"Search results saved to {output_path}")
    
    def _analyze_parameter_importance(self) -> Dict[str, Any]:
        """Analyze parameter importance from completed trials."""
        completed_trials = [t for t in self.trial_results if t['completed']]
        
        if len(completed_trials) < 10:
            return {'error': 'Not enough completed trials for analysis'}
        
        # Simple correlation analysis
        param_correlations = {}
        scores = [t['score'] for t in completed_trials]
        
        for param_name in self.search_space.keys():
            param_values = [t['parameters'][param_name] for t in completed_trials]
            
            # Calculate correlation with scores
            correlation = np.corrcoef(param_values, scores)[0, 1]
            param_correlations[param_name] = {
                'correlation_with_performance': correlation,
                'importance_rank': 0  # Will be filled later
            }
        
        # Rank parameters by absolute correlation
        sorted_params = sorted(param_correlations.items(), 
                             key=lambda x: abs(x[1]['correlation_with_performance']), 
                             reverse=True)
        
        for rank, (param_name, _) in enumerate(sorted_params):
            param_correlations[param_name]['importance_rank'] = rank + 1
        
        return {
            'parameter_importance': param_correlations,
            'most_important_parameter': sorted_params[0][0],
            'analysis_based_on_trials': len(completed_trials)
        }

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for PPO')
    parser.add_argument('--config', type=str, default='config/main_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, default='results/hyperparameter_search',
                       help='Output directory for results')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials to run')
    parser.add_argument('--timeout', type=int, default=24,
                       help='Timeout in hours')
    
    args = parser.parse_args()
    
    # Create searcher
    searcher = HyperparameterSearcher(args.config)
    
    # Update search configuration
    searcher.search_config['n_trials'] = args.trials
    searcher.search_config['timeout_hours'] = args.timeout
    
    # Run search
    results = searcher.search_hyperparameters(args.output)
    
    # Print results
    print("\n" + "="*60)
    print("🔍 HYPERPARAMETER SEARCH COMPLETED!")
    print("="*60)
    print(f"Trials completed: {results['n_trials_completed']}")
    print(f"Search time: {results['search_time_hours']:.1f} hours")
    print(f"Best score: {results['best_score']:.3f}")
    print("\nBest parameters:")
    for param, value in results['best_parameters'].items():
        print(f"  {param}: {value}")
    print("="*60)

if __name__ == "__main__":
    main()
