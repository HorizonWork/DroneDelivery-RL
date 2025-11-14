import logging
import sys
import warnings

__version__ = "1.0.0"
__author__ = "FPT University Team"
__description__ = "Energy-Aware Indoor Drone Delivery System"

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='(asctime)s - (name)s - (levelname)s - (message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

logger = logging.getLogger(__name__)
logger.info(f"DroneDelivery-RL v{__version__} package loaded")
logger.warning("Use direct imports from submodules (e.g., 'from src.rl import PPOAgent')")

__all__ = ['__version__', '__author__', '__description__']