"""
Path Planning Module for DroneDelivery-RL
Hierarchical planning system with global A* and local S-RRT.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

# Module-level availability checks
_PLANNING_AVAILABLE = True
_IMPORT_ERRORS = []

# Global Planner
AStarPlanner = None
OccupancyGrid = None
Heuristics = None
PathOptimizer = None

try:
    from src.planning.global_planner.astar_planner import AStarPlanner
    from src.planning.global_planner.occupancy_grid import OccupancyGrid
    from src.planning.global_planner.heuristics import Heuristics
    from src.planning.global_planner.path_optimizer import PathOptimizer
except ImportError as e:
    _IMPORT_ERRORS.append(f"global_planner: {e}")

# Local Planner
SRRTPlanner = None
CostFunctions = None
DynamicObstacleHandler = None
SafetyChecker = None

try:
    from src.planning.local_planner.srrt_planner import SRRTPlanner
    from src.planning.local_planner.cost_functions import CostFunctions
    from src.planning.local_planner.dynamic_obstacles import DynamicObstacleHandler
    from src.planning.local_planner.safety_checker import SafetyChecker
except ImportError as e:
    _IMPORT_ERRORS.append(f"local_planner: {e}")

# Integration
PlannerManager = None
PathSmoother = None
ExecutionMonitor = None

try:
    from src.planning.integration.planner_manager import PlannerManager
    from src.planning.integration.path_smoother import PathSmoother
    from src.planning.integration.execution_monitor import ExecutionMonitor
except ImportError as e:
    _IMPORT_ERRORS.append(f"integration: {e}")

# Log any import errors
if _IMPORT_ERRORS:
    logger = logging.getLogger(__name__)
    logger.warning(f"Planning module partial import: {len(_IMPORT_ERRORS)} errors")
    for error in _IMPORT_ERRORS:
        logger.debug(f"  - {error}")


__all__ = [
    'AStarPlanner',
    'OccupancyGrid',
    'Heuristics',
    'PathOptimizer',
    'SRRTPlanner',
    'CostFunctions',
    'DynamicObstacleHandler',
    'SafetyChecker',
    'PlannerManager',
    'PathSmoother',
    'ExecutionMonitor',
]
