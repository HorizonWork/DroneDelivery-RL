import logging
from typing import Optional

AIRSIM_AVAILABLE = False
AirSimBridge = None
try:
    from src.bridges.airsim_bridge import AirSimBridge

    AIRSIM_AVAILABLE = True
except ImportError as e:
    logging.debug(f"AirSim bridge not available: {e}")

    class AirSimBridge:
        def __init__(self, args, kwargs):
            raise RuntimeError(
                "AirSimBridge requires 'airsim' package. "
                "Install with: pip install airsim"
            )

ROS2_AVAILABLE = False
ROSBridge = None
try:
    from src.bridges.ros_bridge import ROSBridge, ROS2_AVAILABLE as _ROS2_CHECK

    ROS2_AVAILABLE = _ROS2_CHECK
    if not ROS2_AVAILABLE:
        logging.info("ROSBridge module loaded but ROS2 dependencies not available")
except ImportError as e:
    logging.debug(f"ROS2 bridge not available: {e}")

    class ROSBridge:
        def __init__(self, args, kwargs):
            raise RuntimeError(
                "ROSBridge requires ROS2 packages. "
                "Install ROS2 and required packages: "
                "https:
            )

SLAM_AVAILABLE = False
SLAMBridge = None
try:
    from src.bridges.slam_bridge import SLAMBridge

    SLAM_AVAILABLE = True
except ImportError as e:
    logging.debug(f"SLAM bridge not available: {e}")

    class SLAMBridge:
        def __init__(self, args, kwargs):
            raise RuntimeError(
                "SLAMBridge requires ORB-SLAM3 or compatible SLAM system. "
                "See documentation for SLAM setup instructions."
            )

SENSOR_AVAILABLE = False
SensorBridge = None
try:
    from src.bridges.sensor_bridge import SensorBridge

    SENSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Sensor bridge import failed: {e}")

    class SensorBridge:
        def __init__(self, args, kwargs):
            raise RuntimeError("SensorBridge failed to import. Check installation.")

def get_bridge_status() - dict:

    return {
        "airsim": AIRSIM_AVAILABLE,
        "ros2": ROS2_AVAILABLE,
        "slam": SLAM_AVAILABLE,
        "sensor": SENSOR_AVAILABLE,
    }

def print_bridge_status():

    status = get_bridge_status()

    print("\n" + "="  50)
    print("DroneDelivery-RL Bridge Status")
    print("="  50)

    for bridge_name, available in status.items():
        status_str = " AVAILABLE" if available else " NOT AVAILABLE"
        print(f"{bridge_name.upper():12} : {status_str}")

    print("="  50 + "\n")

    if not status["sensor"]:
        logging.error("SensorBridge is critical but not available!")

    if not status["airsim"]:
        logging.info(
            "AirSim not available - simulation features disabled. "
            "Install with: pip install airsim"
        )

    if not status["ros2"]:
        logging.info(
            "ROS2 not available - ROS integration disabled. "
            "This is optional for basic training."
        )

    if not status["slam"]:
        logging.info(
            "SLAM not available - external SLAM features disabled. "
            "This is optional if using simulated localization."
        )

__all__ = [
    "AirSimBridge",
    "ROSBridge",
    "SLAMBridge",
    "SensorBridge",
    "AIRSIM_AVAILABLE",
    "ROS2_AVAILABLE",
    "SLAM_AVAILABLE",
    "SENSOR_AVAILABLE",
    "get_bridge_status",
    "print_bridge_status",
]

logging.info("DroneDelivery-RL Bridges Module initialized")
_status = get_bridge_status()
_available_count = sum(_status.values())
logging.info(f"Bridges available: {_available_count}/4")

if _available_count  4:
    for name, available in _status.items():
        if not available:
            logging.debug(f"  - {name} bridge: NOT AVAILABLE")
