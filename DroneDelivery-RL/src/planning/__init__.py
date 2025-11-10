"""
Path Planning Module for DroneDelivery-RL
Hierarchical planning system with global A* and local S-RRT.
"""

import logging

# Module-level availability checks
_PLANNING_AVAILABLE = True
_IMPORT_ERRORS = []

try:
    from src.planning.global_planner.astar_planner import AStarPlanner
    from src.planning.global_planner.occupancy_grid import OccupancyGrid
    from src.planning.global_planner.heuristics import Heuristics
    from src.planning.global_planner.path_optimizer import PathOptimizer
except ImportError as e:
    _IMPORT_ERRORS.append(f"global_planner: {e}")
    AStarPlanner = None
    OccupancyGrid = None
    Heuristics = None
    PathOptimizer = None

try:
    from src.planning.local_planner.srrt_planner import SRRTPlanner
    from src.planning.local_planner.cost_functions import CostFunctions
    from src.planning.local_planner.dynamic_obstacles import DynamicObstacleHandler
    from src.planning.local_planner.safety_checker import SafetyChecker
except ImportError as e:
    _IMPORT_ERRORS.append(f"local_planner: {e}")
    SRRTPlanner = None
    CostFunctions = None
    DynamicObstacleHandler = None
    SafetyChecker = None

try:
    from src.planning.integration.planner_manager import PlannerManager
    from src.planning.integration.path_smoother import PathSmoother
    from src.planning.integration.execution_monitor import ExecutionMonitor
except ImportError as e:
    _IMPORT_ERRORS.append(f"integration: {e}")
    PlannerManager = None
    PathSmoother = None
    ExecutionMonitor = None

# Log any import errors
if _IMPORT_ERRORS:
    logger = logging.getLogger(__name__)
    logger.warning(f"Planning module partial import: {len(_IMPORT_ERRORS)} errors")
    for error in _IMPORT_ERRORS:
        logger.debug(f"  - {error}")


# Helper function to create planner with safety checks
def create_hierarchical_planner(config: dict):
    """
    Create hierarchical planner with proper error handling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PlannerManager instance or None if unavailable
    """
    if PlannerManager is None:
        raise RuntimeError(
            "PlannerManager not available. Check planning module imports."
        )
    
    try:
        return PlannerManager(config)
    except Exception as e:
        logging.error(f"Failed to create PlannerManager: {e}")
        return None


def create_global_planner(config: dict):
    """
    Create A* global planner.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AStarPlanner instance or None if unavailable
    """
    if AStarPlanner is None:
        raise RuntimeError(
            "AStarPlanner not available. Check global planner imports."
        )
    
    try:
        return AStarPlanner(config)
    except Exception as e:
        logging.error(f"Failed to create AStarPlanner: {e}")
        return None


def create_local_planner(config: dict):
    """
    Create S-RRT local planner.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SRRTPlanner instance or None if unavailable
    """
    if SRRTPlanner is None:
        raise RuntimeError(
            "SRRTPlanner not available. Check local planner imports."
        )
    
    try:
        return SRRTPlanner(config)
    except Exception as e:
        logging.error(f"Failed to create SRRTPlanner: {e}")
        return None


__all__ = [
    # Global Planner
    'AStarPlanner',
    'OccupancyGrid',
    'Heuristics',
    'PathOptimizer',
    
    # Local Planner
    'SRRTPlanner',
    'CostFunctions',
    'DynamicObstacleHandler',
    'SafetyChecker',
    
    # Integration
    'PlannerManager',
    'PathSmoother',
    'ExecutionMonitor',
    
    # Factory functions
    'create_hierarchical_planner',
    'create_global_planner',
    'create_local_planner',
]
