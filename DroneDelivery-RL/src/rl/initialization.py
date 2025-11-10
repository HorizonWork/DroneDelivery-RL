"""
RL System Initialization
Helper functions for initializing RL components.
"""

import torch
import logging
from typing import Dict, Any

from src.rl.agents.ppo_agent import PPOAgent, PPOConfig
from src.rl.agents.actor_critic import ActorCriticNetwork, NetworkConfig


def initialize_rl_system(rl_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize complete RL system with agent and components.
    
    Args:
        rl_config: RL configuration dictionary
        
    Returns:
        Dictionary containing initialized RL components:
            - 'agent': PPOAgent instance
            - 'config': PPO configuration
            - 'device': torch device
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing RL system...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Extract dimensions from config
    observation_dim = rl_config.get('observation_dim', 37)  # Default from report
    action_dim = rl_config.get('action_dim', 4)  # [vx, vy, vz, yaw_rate]
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        learning_rate=rl_config.get('learning_rate', 3e-4),
        rollout_length=rl_config.get('rollout_length', 2048),
        batch_size=rl_config.get('batch_size', 64),
        epochs_per_update=rl_config.get('epochs_per_update', 10),
        gamma=rl_config.get('gamma', 0.99),
        gae_lambda=rl_config.get('gae_lambda', 0.95),
        clip_epsilon=rl_config.get('clip_epsilon', 0.2),
        value_loss_coef=rl_config.get('value_loss_coef', 0.5),
        entropy_coef=rl_config.get('entropy_coef', 0.01),
        max_grad_norm=rl_config.get('max_grad_norm', 0.5),
        use_gae=rl_config.get('use_gae', True),
        normalize_advantages=rl_config.get('normalize_advantages', True)
    )
    
    # Create network configuration
    network_config = NetworkConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dims=rl_config.get('hidden_dims', [256, 256]),  # Table 2
        activation=rl_config.get('activation', 'relu'),
        use_layer_norm=rl_config.get('use_layer_norm', True)
    )
    
    # Initialize agent
    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim=action_dim,
        ppo_config=ppo_config,
        network_config=network_config,
        device=device
    )
    
    logger.info(f"PPO Agent initialized with {observation_dim}D obs, {action_dim}D action")
    logger.info(f"Network: {network_config.hidden_dims}, Device: {device}")
    
    return {
        'agent': agent,
        'config': ppo_config,
        'network_config': network_config,
        'device': device,
        'observation_dim': observation_dim,
        'action_dim': action_dim
    }


def create_agent_from_config(config_path: str) -> PPOAgent:
    """
    Create PPOAgent from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Initialized PPOAgent
    """
    from src.utils.config_loader import load_config
    
    config = load_config(config_path)
    rl_system = initialize_rl_system(config.get('rl', {}))
    
    return rl_system['agent']
