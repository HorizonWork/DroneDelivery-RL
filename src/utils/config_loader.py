"""
Configuration Management
Centralized configuration loading and validation for all system components.
Supports YAML, JSON, and environment variable overrides.
"""

import os
import yaml
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import copy


@dataclass
class SystemConfig:
    """Complete system configuration."""

    # Core modules
    localization: Dict[str, Any] = field(default_factory=dict)
    planning: Dict[str, Any] = field(default_factory=dict)
    rl: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    # System settings
    logging: Dict[str, Any] = field(default_factory=dict)
    data_paths: Dict[str, str] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)

    # Hardware settings
    hardware: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Advanced configuration management system.
    Handles loading, validation, environment overrides, and nested configurations.
    """

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Configuration cache
        self._config_cache: Dict[str, Any] = {}

        # Default configuration paths
        self.default_configs = {
            "main": self.config_dir / "main_config.yaml",
            "localization": self.config_dir / "localization_config.yaml",
            "planning": self.config_dir / "planning_config.yaml",
            "rl": self.config_dir / "rl_config.yaml",
            "environment": self.config_dir / "environment_config.yaml",
        }

        # Environment variable prefix
        self.env_prefix = "DRONE_DELIVERY_"

        self.logger.info(f"Config Manager initialized: {config_dir}")

    def load_config(self, config_name: str = "main") -> SystemConfig:
        """
        Load configuration from file with environment overrides.

        Args:
            config_name: Configuration name to load

        Returns:
            Loaded system configuration
        """
        if config_name in self._config_cache:
            return self._config_cache[config_name]

        # Load base configuration
        config_path = self.default_configs.get(
            config_name, self.config_dir / f"{config_name}_config.yaml"
        )

        if not config_path.exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return SystemConfig()

        # Load from file
        config_data = self._load_config_file(config_path)

        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)

        # Create SystemConfig object
        system_config = SystemConfig(**config_data)

        # Cache configuration
        self._config_cache[config_name] = system_config

        self.logger.info(f"Configuration loaded: {config_name}")
        return system_config

    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

        except Exception as e:
            self.logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        overrides = {}

        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Convert environment variable to config path
                config_key = key[len(self.env_prefix) :].lower()
                config_path = config_key.split("_")

                # Parse value
                parsed_value = self._parse_env_value(value)

                # Set nested value
                self._set_nested_value(overrides, config_path, parsed_value)

        # Merge overrides
        if overrides:
            config_data = self._merge_configs(config_data, overrides)
            self.logger.info(f"Applied {len(overrides)} environment overrides")

        return config_data

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try JSON for complex types
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Return as string
        return value

    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any):
        """Set value in nested dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[path[-1]] = value

    def _merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = copy.deepcopy(base_config)

        for key, value in override_config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def save_config(self, config: SystemConfig, output_path: str):
        """Save configuration to file."""
        output_path = Path(output_path)

        # Convert to dictionary
        config_dict = {
            "localization": config.localization,
            "planning": config.planning,
            "rl": config.rl,
            "environment": config.environment,
            "evaluation": config.evaluation,
            "logging": config.logging,
            "data_paths": config.data_paths,
            "output_paths": config.output_paths,
            "hardware": config.hardware,
            "simulation": config.simulation,
        }

        # Save based on extension
        with open(output_path, "w") as f:
            if output_path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_dict, f, indent=2)

        self.logger.info(f"Configuration saved to {output_path}")


# Convenience functions
def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """Load system configuration."""
    manager = ConfigManager()

    if config_path:
        # Load specific config file
        custom_path = Path(config_path)
        if custom_path.exists():
            config_data = manager._load_config_file(custom_path)
            return SystemConfig(**config_data)

    return manager.load_config("main")


def validate_config(config: SystemConfig) -> Dict[str, List[str]]:
    """
    Validate system configuration.

    Returns:
        Dictionary of validation errors by component
    """
    errors = {}

    # Validate RL configuration
    rl_errors = []
    if not config.rl:
        rl_errors.append("RL configuration is empty")
    else:
        # Check PPO hyperparameters
        ppo_config = config.rl.get("ppo", {})
        if ppo_config.get("learning_rate", 0) <= 0:
            rl_errors.append("Learning rate must be positive")
        if ppo_config.get("rollout_length", 0) <= 0:
            rl_errors.append("Rollout length must be positive")

    if rl_errors:
        errors["rl"] = rl_errors

    # Validate environment configuration
    env_errors = []
    if not config.environment:
        env_errors.append("Environment configuration is empty")
    else:
        building = config.environment.get("building", {})
        if not building.get("floors", 0):
            env_errors.append("Building must have at least one floor")

    if env_errors:
        errors["environment"] = env_errors

    # Validate paths
    path_errors = []
    for path_name, path_value in config.data_paths.items():
        if path_value and not Path(path_value).exists():
            path_errors.append(f"Path does not exist: {path_name} = {path_value}")

    if path_errors:
        errors["paths"] = path_errors

    return errors
