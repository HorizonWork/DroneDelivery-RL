"""
Reinforcement Learning Module
PPO-based energy-aware control for indoor drone delivery.

This module implements the complete RL system including:
- PPO agent with exact Table 2 hyperparameters
- Actor-Critic networks (256→128→64 hidden layers)
- GAE advantage estimation (λ=0.95)
- Comprehensive evaluation system
- Training utilities and monitoring
"""

import logging

# Core RL components
from .agents import PPOAgent, ActorCriticNetwork, PolicyNetwork, ValueNetwork, GAECalculator
from .evaluation import DroneEvaluator, MetricsCollector, BaselineComparator, EnergyAnalyzer, TrajectoryAnalyzer

# Training components (would be implemented next)
# from .training import PPOTrainer, ExperienceBuffer, TrainingMonitor

# Utility components
# from .utils import RewardFunction, ObservationProcessor, ActionProcessor

__version__ = "1.0.0"

# Configure module logger
logger = logging.getLogger(__name__)
logger.info("DroneDelivery-RL module loaded")

# Export main components
__all__ = [
    # Core agents
    'PPOAgent',
    'ActorCriticNetwork', 
    'PolicyNetwork',
    'ValueNetwork',
    'GAECalculator',
    
    # Evaluation system
    'DroneEvaluator',
    'MetricsCollector',
    'BaselineComparator', 
    'EnergyAnalyzer',
    'TrajectoryAnalyzer',
    
    # Training (to be implemented)
    # 'PPOTrainer',
    # 'ExperienceBuffer',
    # 'TrainingMonitor',
    
    # Utilities (to be implemented)
    # 'RewardFunction',
    # 'ObservationProcessor', 
    # 'ActionProcessor'
]

# System information
SYSTEM_INFO = {
    'algorithm': 'Proximal Policy Optimization (PPO)',
    'observation_space': '35D continuous (position, velocity, goal, obstacles, energy)',
    'action_space': '4D continuous (vx, vy, vz, yaw_rate)',
    'network_architecture': '256→128→64 hidden layers with tanh activation',
    'hyperparameters': {
        'learning_rate': '3e-4',
        'rollout_length': '2048',
        'batch_size': '64', 
        'epochs_per_update': '10',
        'clip_range': '0.2',
        'discount_factor': '0.99',
        'gae_lambda': '0.95'
    },
    'evaluation_metrics': [
        'Success Rate (%)',
        'Energy Consumption (J)',
        'Flight Time (s)', 
        'Collision Rate (%)',
        'ATE Error (m)'
    ],
    'performance_targets': {
        'success_rate': '≥96%',
        'energy_savings': '≥25% vs A* Only',
        'ate_accuracy': '≤5cm',
        'collision_rate': '≤2%'
    }
}

def get_system_info() -> dict:
    """Get RL system information."""
    return SYSTEM_INFO.copy()

def create_ppo_agent(observation_dim: int = 35, action_dim: int = 4, 
                    config: dict = None) -> PPOAgent:
    """
    Factory function to create PPO agent with default configuration.
    
    Args:
        observation_dim: Observation space dimension
        action_dim: Action space dimension
        config: Custom configuration
        
    Returns:
        Configured PPO agent
    """
    default_config = {
        'ppo': {
            'learning_rate': 3e-4,
            'rollout_length': 2048,
            'batch_size': 64,
            'epochs_per_update': 10,
            'clip_range': 0.2,
            'discount_factor': 0.99,
            'gae_lambda': 0.95,
            'entropy_coefficient': 0.01,
            'value_loss_coefficient': 0.5,
            'max_grad_norm': 0.5
        },
        'hidden_sizes': [256, 128, 64],
        'activation': 'tanh'
    }
    
    if config:
        # Merge custom config with defaults
        for key, value in config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return PPOAgent(observation_dim, action_dim, default_config)

def create_evaluator(config: dict = None) -> DroneEvaluator:
    """
    Factory function to create drone evaluator with default configuration.
    
    Args:
        config: Custom evaluation configuration
        
    Returns:
        Configured drone evaluator
    """
    default_config = {
        'num_episodes': 100,
        'episode_timeout': 300.0,
        'goal_tolerance': 0.5,
        'success_threshold': 0.5,
        'save_trajectories': True,
        'save_detailed_logs': True,
        'output_dir': 'evaluation_results'
    }
    
    if config:
        default_config.update(config)
    
    return DroneEvaluator(default_config)

# Module initialization
def initialize_rl_system(config: dict = None) -> dict:
    """
    Initialize complete RL system.
    
    Args:
        config: System configuration
        
    Returns:
        Initialized system components
    """
    logger.info("Initializing DroneDelivery-RL system...")
    
    # Create default components
    agent = create_ppo_agent(config=config)
    evaluator = create_evaluator(config=config)
    
    system = {
        'agent': agent,
        'evaluator': evaluator,
        'system_info': get_system_info(),
        'initialized_at': time.time()
    }
    
    logger.info("RL system initialization completed")
    return system
