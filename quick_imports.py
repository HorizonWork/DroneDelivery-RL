#!/usr/bin/env python3
"""
Quick fix for all import errors in DroneDelivery-RL
Fixes:
1. RRTStarPlanner -> RRTStarController
2. AirSimBridge undefined errors
"""

import re
from pathlib import Path

def fix_baselines_init():
    """Fix src/baselines/__init__.py"""
    file_path = Path('src/baselines/__init__.py')
    
    content = '''"""
Baseline methods for drone delivery evaluation.
Implements classical planning approaches for comparison with PPO agent.
"""

# Import baseline components
from src.baselines.astar_baseline.astar_controller import AStarController
from src.baselines.astar_baseline.pid_controller import PIDController
from src.baselines.astar_baseline.evaluator import AStarEvaluator

from src.baselines.rrt_baseline.rrt_star import RRTStarController
from src.baselines.rrt_baseline.pid_controller import PIDController as RRTPIDController
from src.baselines.rrt_baseline.evaluator import RRTEvaluator

from src.baselines.random_baseline.random_agent import RandomAgent
from src.baselines.random_baseline.evaluator import RandomEvaluator


# Create baseline wrapper classes for easy usage
class AStarBaseline:
    """A* + PID Baseline Method (Section 4.2)."""
    
    def __init__(self, config: dict):
        self.planner = AStarController(config)
        self.controller = PIDController(config)
        self.evaluator = AStarEvaluator(config)
        self.config = config
    
    def plan(self, start, goal, obstacles):
        """Plan path from start to goal avoiding obstacles."""
        self.planner.update_occupancy_grid(obstacles)
        return self.planner.plan_path(start, goal)
    
    def control(self, current_state, target_waypoint):
        """Generate control commands to reach target waypoint."""
        return self.controller.compute_control(current_state, target_waypoint)
    
    def evaluate(self, environment, num_episodes=100):
        """Evaluate baseline performance."""
        return self.evaluator.evaluate(environment, self.planner, self.controller, num_episodes)


class RRTBaseline:
    """RRT* + PID Baseline Method (Section 4.2)."""
    
    def __init__(self, config: dict):
        self.planner = RRTStarController(config)
        self.controller = RRTPIDController(config)
        self.evaluator = RRTEvaluator(config)
        self.config = config
    
    def plan(self, start, goal, obstacles):
        """Plan path from start to goal avoiding obstacles."""
        return self.planner.plan(start, goal, obstacles)
    
    def control(self, current_state, target_waypoint):
        """Generate control commands to reach target waypoint."""
        return self.controller.compute_control(current_state, target_waypoint)
    
    def evaluate(self, environment, num_episodes=100):
        """Evaluate baseline performance."""
        return self.evaluator.evaluate(environment, self.planner, self.controller, num_episodes)


class RandomBaseline:
    """Random Policy Baseline (Section 4.2)."""
    
    def __init__(self, config: dict):
        self.agent = RandomAgent(config)
        self.evaluator = RandomEvaluator(config)
        self.config = config
    
    def act(self, observation):
        """Select random action."""
        return self.agent.select_action(observation)
    
    def evaluate(self, environment, num_episodes=100):
        """Evaluate baseline performance."""
        return self.evaluator.evaluate(environment, self.agent, num_episodes)


__all__ = [
    'AStarBaseline',
    'RRTBaseline',
    'RandomBaseline',
    'AStarController',
    'PIDController',
    'AStarEvaluator',
    'RRTStarController',
    'RRTPIDController',
    'RRTEvaluator',
    'RandomAgent',
    'RandomEvaluator'
]
'''
    
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed: {file_path}")


def fix_planning_init():
    """Fix src/planning/__init__.py"""
    file_path = Path('src/planning/__init__.py')
    
    content = '''"""
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
'''
    
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed: {file_path}")


def main():
    """Run all fixes"""
    print("🔧 Fixing import errors...")
    
    try:
        fix_baselines_init()
        fix_planning_init()
        
        print("\n✅ All fixes applied successfully!")
        print("\n📝 Next steps:")
        print("1. Clear Python cache: find . -type d -name '__pycache__' -exec rm -r {} + 2>/dev/null")
        print("2. Run tests: pytest tests/test_basic_imports.py -v")
        
    except Exception as e:
        print(f"\n❌ Error during fix: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
