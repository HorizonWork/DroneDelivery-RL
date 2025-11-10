"""
Baseline methods for drone delivery evaluation.
Implements classical planning approaches for comparison with PPO agent.
"""

# Import baseline components
from src.baselines.astar_baseline.astar_controller import AStarController
from src.baselines.astar_baseline.pid_controller import PIDController
from src.baselines.astar_baseline.evaluator import AStarEvaluator

from src.baselines.rrt_baseline.rrt_star import RRTStarPlanner
from src.baselines.rrt_baseline.pid_controller import PIDController as RRTPIDController
from src.baselines.rrt_baseline.evaluator import RRTEvaluator

from src.baselines.random_baseline.random_agent import RandomAgent
from src.baselines.random_baseline.evaluator import RandomEvaluator


# Create baseline wrapper classes for easy usage
class AStarBaseline:
    """
    A* + PID Baseline Method (Section 4.2).
    Uses A* for global planning and PID for local control.
    """
    
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
    """
    RRT* + PID Baseline Method (Section 4.2).
    Uses RRT* for global planning and PID for local control.
    """
    
    def __init__(self, config: dict):
        self.planner = RRTStarPlanner(config)
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
    """
    Random Policy Baseline (Section 4.2).
    Takes random actions for comparison.
    """
    
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
    # Wrapper classes (recommended for use)
    'AStarBaseline',
    'RRTBaseline',
    'RandomBaseline',
    
    # Individual components (for advanced usage)
    'AStarController',
    'PIDController',
    'AStarEvaluator',
    'RRTStarPlanner',
    'RRTPIDController',
    'RRTEvaluator',
    'RandomAgent',
    'RandomEvaluator'
]
