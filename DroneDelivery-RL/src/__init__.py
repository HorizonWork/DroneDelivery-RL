"""
DroneDelivery-RL: Energy-Aware Indoor Drone Navigation System
Main package initialization for the complete indoor drone delivery system.

This package implements the complete system described in the research report:
"Indoor Multi-Floor UAV Delivery Energy-Aware Navigation through A*-RRT and Reinforcement Learning"

System Architecture:
├── bridges/         - Hardware and simulation interfaces
├── environment/     - Drone environment and physics simulation  
├── localization/    - VI-SLAM and pose estimation
├── planning/        - A* global planning and S-RRT local replanning
├── rl/             - PPO reinforcement learning system
└── utils/          - Common utilities and tools

Key Features:
- 5-floor indoor navigation (20m × 40m × 15m building)
- Centimeter-scale VI-SLAM localization (≤5cm ATE)
- Energy-aware path planning and control
- PPO-based reinforcement learning (96% success rate target)
- 25% energy efficiency improvement vs A* Only baseline
- Real-time obstacle avoidance with S-RRT
- Comprehensive evaluation and monitoring system

Usage:
    from src import DroneDeliverySystem
    
    # Initialize complete system
    system = DroneDeliverySystem(config_path='config/main_config.yaml')
    
    # Run autonomous navigation
    result = system.navigate_to_goal(start_pos, goal_pos)
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Any

# Package version
__version__ = "1.0.0"
__author__ = "FPT University Team"
__description__ = "Energy-Aware Indoor Drone Delivery Navigation System"

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")

# Setup basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Package logger
logger = logging.getLogger(__name__)

# Core system components
from .utils import setup_logging, ConfigManager, load_config
from .localization import VISLAMLocalizer, PoseEstimator
from .planning import GlobalPlanner, LocalPlanner, PathPlanner
from .rl import PPOAgent, DroneEvaluator, initialize_rl_system
from .environment import DroneEnvironment, BuildingEnvironment

# System integration
class DroneDeliverySystem:
    """
    Complete drone delivery system integration.
    Provides high-level interface for autonomous navigation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize complete drone delivery system.
        
        Args:
            config_path: Path to system configuration file
        """
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = load_config()
        
        # Setup system logging
        self.logger_system = setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.logger.info("Initializing Drone Delivery System...")
        
        # Localization system
        self.localizer = VISLAMLocalizer(self.config.localization)
        
        # Planning system  
        self.global_planner = GlobalPlanner(self.config.planning)
        self.local_planner = LocalPlanner(self.config.planning)
        
        # RL system
        self.rl_system = initialize_rl_system(self.config.rl)
        self.rl_agent = self.rl_system['agent']
        
        # Environment
        self.environment = DroneEnvironment(self.config.environment)
        
        # System state
        self.is_initialized = True
        self.current_mission = None
        
        self.logger.info("Drone Delivery System initialized successfully")
        self.logger.info(f"System version: {__version__}")
    
    def navigate_to_goal(self, start_position: list, goal_position: list, 
                        use_rl: bool = True) -> Dict[str, Any]:
        """
        Execute autonomous navigation from start to goal.
        
        Args:
            start_position: Starting position [x, y, z]
            goal_position: Goal position [x, y, z]  
            use_rl: Whether to use RL agent for control
            
        Returns:
            Navigation result dictionary
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        self.logger.info(f"Starting navigation: {start_position} → {goal_position}")
        
        try:
            # Global path planning
            global_path = self.global_planner.plan_path(start_position, goal_position)
            
            if not global_path:
                return {'success': False, 'error': 'Global path planning failed'}
            
            # Execute navigation
            if use_rl:
                result = self._execute_rl_navigation(global_path, goal_position)
            else:
                result = self._execute_classical_navigation(global_path, goal_position)
            
            self.logger.info(f"Navigation completed: {result['success']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _execute_rl_navigation(self, global_path: list, goal_position: list) -> Dict[str, Any]:
        """Execute navigation using RL agent."""
        # Reset environment with global path
        observation = self.environment.reset(
            global_path=global_path,
            goal_position=goal_position
        )
        
        total_energy = 0.0
        trajectory = []
        done = False
        steps = 0
        max_steps = 5000  # Safety limit
        
        while not done and steps < max_steps:
            # Get action from RL agent
            action, _ = self.rl_agent.select_action(observation, deterministic=True)
            
            # Execute action
            observation, reward, done, info = self.environment.step(action)
            
            # Record trajectory and metrics
            trajectory.append(info.get('position', [0, 0, 0]))
            total_energy += info.get('energy_consumption', 0.0)
            steps += 1
            
            # Check for collision
            if info.get('collision', False):
                return {
                    'success': False,
                    'reason': 'collision',
                    'trajectory': trajectory,
                    'energy': total_energy,
                    'steps': steps
                }
        
        # Check success
        final_position = info.get('position', [0, 0, 0])
        distance_to_goal = ((final_position[0] - goal_position[0])**2 + 
                           (final_position[1] - goal_position[1])**2 + 
                           (final_position[2] - goal_position[2])**2)**0.5
        
        success = distance_to_goal <= 0.5  # 0.5m tolerance
        
        return {
            'success': success,
            'trajectory': trajectory,
            'energy': total_energy,
            'flight_time': steps / 20.0,  # 20Hz
            'final_distance': distance_to_goal,
            'steps': steps,
            'method': 'RL_PPO'
        }
    
    def _execute_classical_navigation(self, global_path: list, goal_position: list) -> Dict[str, Any]:
        """Execute navigation using classical planning only."""
        # Simplified classical navigation implementation
        trajectory = global_path.copy()
        
        # Estimate energy and time (simplified)
        path_length = sum(
            ((trajectory[i][0] - trajectory[i-1][0])**2 + 
             (trajectory[i][1] - trajectory[i-1][1])**2 + 
             (trajectory[i][2] - trajectory[i-1][2])**2)**0.5
            for i in range(1, len(trajectory))
        )
        
        estimated_energy = path_length * 120.0  # J/m (rough estimate)
        estimated_time = path_length / 2.0      # Assume 2 m/s average speed
        
        return {
            'success': True,
            'trajectory': trajectory,
            'energy': estimated_energy,
            'flight_time': estimated_time,
            'final_distance': 0.0,
            'method': 'Classical_AStar'
        }
    
    def evaluate_system(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate complete system performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results
        """
        evaluator = DroneEvaluator({'num_episodes': num_episodes})
        results = evaluator.evaluate_policy(self.rl_agent, self.environment)
        
        return {
            'success_rate': results.success_rate,
            'mean_energy': results.mean_energy,
            'mean_time': results.mean_time,
            'collision_rate': results.collision_rate,
            'ate_error': results.mean_ate
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'initialized': self.is_initialized,
            'version': __version__,
            'components': {
                'localizer': hasattr(self, 'localizer'),
                'global_planner': hasattr(self, 'global_planner'),
                'local_planner': hasattr(self, 'local_planner'),
                'rl_agent': hasattr(self, 'rl_agent'),
                'environment': hasattr(self, 'environment')
            },
            'current_mission': self.current_mission
        }

# Package-level convenience functions
def create_system(config_path: Optional[str] = None) -> DroneDeliverySystem:
    """Create and initialize drone delivery system."""
    return DroneDeliverySystem(config_path)

def get_system_info() -> Dict[str, str]:
    """Get package information."""
    return {
        'name': 'DroneDelivery-RL',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

# Export main components
__all__ = [
    # Main system
    'DroneDeliverySystem',
    'create_system',
    'get_system_info',
    
    # Core components
    'VISLAMLocalizer', 'PoseEstimator',
    'GlobalPlanner', 'LocalPlanner', 'PathPlanner', 
    'PPOAgent', 'DroneEvaluator',
    'DroneEnvironment', 'BuildingEnvironment',
    
    # Utilities
    'setup_logging', 'ConfigManager', 'load_config',
    
    # Package info
    '__version__', '__author__', '__description__'
]

# Log package initialization
logger.info(f"DroneDelivery-RL v{__version__} package loaded")
logger.info("Indoor Multi-Floor UAV Delivery System Ready")
