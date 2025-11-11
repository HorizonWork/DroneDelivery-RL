"""
DroneDelivery-RL: Energy-Aware Indoor Drone Navigation System
Main package initialization.

WARNING: This file intentionally left minimal to avoid circular imports.
Use direct imports from submodules instead.

Example:
    from src.environment.airsim_env import AirSimEnvironment
    from src.rl.agents.ppo_agent import PPOAgent
"""

import logging
import sys
import warnings

# Package version
__version__ = "1.0.0"
__author__ = "FPT University Team"
__description__ = "Energy-Aware Indoor Drone Delivery System"

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

# Setup basic logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Package logger
logger = logging.getLogger(__name__)
logger.info(f"DroneDelivery-RL v{__version__} package loaded")
logger.warning("Use direct imports from submodules (e.g., 'from src.rl import PPOAgent')")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT: DO NOT ADD ANY OTHER IMPORTS HERE TO AVOID CIRCULAR DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

__all__ = ['__version__', '__author__', '__description__']