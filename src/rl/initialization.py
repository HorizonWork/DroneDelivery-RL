import torch
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

from src.rl.agents.ppo_agent import PPOAgent, PPOConfig
from src.rl.agents.actor_critic import ActorCriticNetwork, NetworkConfig

def initialize_rl_system(rl_config: Dict[str, Any]) - Dict[str, Any]:

    logger = logging.getLogger(__name__)
    logger.info("Initializing RL system...")

    hyperparams, hyper_source = _load_hyperparameters(
        rl_config.get("hyperparameters_path")
    )
    resolved_rl_config: Dict[str, Any] = {}
    if hyperparams:
        logger.info(f"Applying PPO hyperparameters from {hyper_source}")
        resolved_rl_config = _deep_merge_dicts(resolved_rl_config, hyperparams)
    resolved_rl_config = _deep_merge_dicts(resolved_rl_config, rl_config)
    rl_config = resolved_rl_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    observation_dim = rl_config.get("observation_dim", 40)
    action_dim = rl_config.get("action_dim", 4)

    ppo_config = PPOConfig(
        learning_rate=rl_config.get("learning_rate", 3e-4),
        rollout_length=rl_config.get("rollout_length", 2048),
        batch_size=rl_config.get("batch_size", 64),
        epochs_per_update=rl_config.get("epochs_per_update", 10),
        discount_factor=rl_config.get("gamma", 0.99),
        gae_lambda=rl_config.get("gae_lambda", 0.95),
        clip_range=rl_config.get("clip_epsilon", 0.2),
        value_loss_coefficient=rl_config.get("value_loss_coef", 0.5),
        entropy_coefficient=rl_config.get("entropy_coef", 0.01),
        max_grad_norm=rl_config.get("max_grad_norm", 0.5),
    )

    network_config = NetworkConfig(
        observation_dim=40,
        action_dim=4,
        hidden_sizes=[256, 128, 64],
        activation="tanh",
    )

    agent = PPOAgent(
        observation_dim=observation_dim, action_dim=action_dim, config=rl_config
    )

    logger.info(
        f"PPO Agent initialized with {observation_dim}D obs, {action_dim}D action"
    )
    logger.info(f"Network: {network_config.hidden_dims}, Device: {device}")

    return {
        "agent": agent,
        "config": ppo_config,
        "network_config": network_config,
        "device": device,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
    }

def create_agent_from_config(config_path: str) - PPOAgent:

    from src.utils.config_loader import load_config

    config = load_config(config_path)
    rl_system = initialize_rl_system(config.get("rl", {}))

    return rl_system["agent"]

def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) - Dict[str, Any]:

    result = dict(base) if isinstance(base, dict) else {}
    for key, value in (override or {}).items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def _load_hyperparameters(
    path_value: Optional[str],
) - Tuple[Dict[str, Any], Optional[str]]:

    if not path_value:
        return {}, None

    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path_value

    if not path.exists():
        logging.getLogger(__name__).warning(
            "Hyperparameter file not found at s. Using inline RL config.", path
        )
        return {}, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data, str(path)
    except Exception as exc:
        logging.getLogger(__name__).error(
            "Failed to read hyperparameter file s: s", path, exc
        )
        return {}, None
