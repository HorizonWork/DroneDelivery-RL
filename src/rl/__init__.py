import logging
import time

from src.rl.agents.ppo_agent import PPOAgent, PPOConfig
from src.rl.agents.actor_critic import ActorCriticNetwork, NetworkConfig
from src.rl.agents.policy_networks import PolicyNetwork, ValueNetwork
from src.rl.evaluation.evaluator import DroneEvaluator
from src.rl.agents.gae_calculator import GAECalculator
from src.rl.training.trainer import PPOTrainer
from src.rl.initialization import initialize_rl_system, create_agent_from_config

__all__ = [
    "PPOAgent",
    "PPOConfig",
    "GAECalculator",
    "PolicyNetwork",
    "ActorCriticNetwork",
    "ValueNetwork",
    "NetworkConfig",
    "DroneEvaluator",
    "PPOTrainer",
    "initialize_rl_system",
    "create_agent_from_config",
]

SYSTEM_INFO = {
    "algorithm": "Proximal Policy Optimization (PPO)",
    "observation_space": "35D continuous",
    "action_space": "4D continuous",
    "hyperparameters": {
        "learning_rate": "3e-4",
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
    },
}

def get_system_info() - dict:

    return SYSTEM_INFO.copy()

def create_ppo_agent(
    observation_dim: int = 35, action_dim: int = 4, config: dict = None
) - PPOAgent:

    if config is None:
        config = {}

    default_config = {
        "ppo": {
            "learning_rate": 3e-4,
            "rollout_length": 2048,
            "batch_size": 64,
            "epochs_per_update": 10,
            "clip_range": 0.2,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "entropy_coefficient": 0.01,
            "value_loss_coefficient": 0.5,
            "max_grad_norm": 0.5,
        },
        "hidden_sizes": [256, 128, 64],
        "activation": "tanh",
    }

    for key, value in config.items():
        if key == "ppo" and isinstance(value, dict):
            default_config["ppo"].update(value)
        else:
            default_config[key] = value

    try:
        return PPOAgent(observation_dim, action_dim, default_config)
    except Exception as e:
        logging.error(f"Failed to create PPOAgent: {e}")
        return None
