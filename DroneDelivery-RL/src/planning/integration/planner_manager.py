"""
Planner Manager
Coordinates global A* and local S-RRT planners with seamless integration.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from src.bridges.airsim_bridge import AirSimBridge
from src.environment.sensor_interface import SensorInterface
from src.planning.integration.execution_monitor import ExecutionMonitor
from src.planning.global_planner.astar_planner import AStarPlanner, PlanningResult
from src.planning.local_planner.srrt_planner import SRRTPlanner

class PlanningMode(Enum):
    """Planning execution modes."""
    GLOBAL_ONLY = "global_only"
    LOCAL_ONLY = "local_only"
    HIERARCHICAL = "hierarchical"  # Global + Local (default)

@dataclass
class PlanningState:
    """Current planning system state."""
    mode: PlanningMode = PlanningMode.HIERARCHICAL
    global_path: List[Tuple[float, float, float]] = field(default_factory=list)
    local_path: List[Tuple[float, float, float]] = field(default_factory=list)
    active_path: List[Tuple[float, float, float]] = field(default_factory=list)
    current_waypoint_index: int = 0
    global_plan_valid: bool = False
    local_plan_active: bool = False
    last_replan_time: float = 0.0

class PlannerManager:
    """
    Hierarchical planning manager.
    Coordinates A* global planning with S-RRT local replanning.
    """
    
    def __init__(self, config: Dict[str, Any],
                    airsim_bridge: AirSimBridge,
                    sensor_interface: SensorInterface):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize planners
        self.global_planner = AStarPlanner(config.get('global_planner', {}))
        self.local_planner = SRRTPlanner(config.get('local_planner', {}))
        
        # Planning parameters
        self.replanning_threshold = config.get('replanning_threshold', 2.0)    # meters (obstacle proximity)
        self.replan_frequency = config.get('replan_frequency', 5.0)            # Hz
        self.global_replan_interval = config.get('global_replan_interval', 30.0) # seconds
        
        # Waypoint following
        self.waypoint_tolerance = config.get('waypoint_tolerance', 1.0)        # meters
        self.lookahead_distance = config.get('lookahead_distance', 3.0)        # meters
        
        # State management
        self.planning_state = PlanningState()
        self.planning_lock = threading.Lock()
        
        # Performance monitoring
        self.planning_statistics = {
            'global_plans': 0,
            'local_replans': 0,
            'total_planning_time': 0.0,
            'average_planning_time': 0.0,
            'replan_triggers': 0
        }
        
        # Initialize execution monitor with dependencies
        self.execution_monitor = ExecutionMonitor(
            config.get('execution_monitor', {}),
            airsim_bridge,
            sensor_interface        
    )
    
        self.logger.info("Planner Manager initialized")
        self.logger.info(f"Mode: {self.planning_state.mode.value}")
        self.logger.info(f"Replanning threshold: {self.replanning_threshold}m")
        self.logger.info(f"Waypoint tolerance: {self.waypoint_tolerance}m")
    
    def plan_mission(self, start_position: Tuple[float, float, float],
                    goal_position: Tuple[float, float, float],
                    obstacles: List[Tuple[float, float, float]] = None) -> bool:
        """
        Plan complete mission from start to goal.
        
        Args:
            start_position: Mission start position
            goal_position: Mission goal position
            obstacles: Current obstacle positions
            
        Returns:
            Success status
        """
        planning_start = time.time()
        
        with self.planning_lock:
            self.logger.info(f"Planning mission: {start_position} → {goal_position}")
            
            # Update obstacles in global planner
            if obstacles:
                self.global_planner.update_occupancy(obstacles)
            
            # Execute global planning
            global_result = self.global_planner.plan_path(start_position, goal_position)
            
            if not global_result.success:
                self.logger.error("Global planning failed")
                return False
            
            # Update planning state
            self.planning_state.global_path = global_result.path
            self.planning_state.active_path = global_result.path.copy()
            self.planning_state.current_waypoint_index = 0
            self.planning_state.global_plan_valid = True
            self.planning_state.local_plan_active = False
            
            # Update statistics
            planning_time = time.time() - planning_start
            self.planning_statistics['global_plans'] += 1
            self.planning_statistics['total_planning_time'] += planning_time
            self.planning_statistics['average_planning_time'] = (
                self.planning_statistics['total_planning_time'] / 
                self.planning_statistics['global_plans']
            )
            
            self.logger.info(f"Mission planned successfully in {planning_time:.3f}s")
            self.logger.info(f"Global path: {len(global_result.path)} waypoints, "
                           f"cost: {global_result.total_cost:.2f}")
            
            return True
    
    def update_execution(self, current_position: Tuple[float, float, float],
                        current_obstacles: List[Tuple[float, float, float]],
                        goal_position: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """
        Update path execution with dynamic replanning.
        
        Args:
            current_position: Current drone position
            current_obstacles: Current obstacle positions
            goal_position: Mission goal position
            
        Returns:
            Active path to follow or None if no valid path
        """
        current_time = time.time()
        
        with self.planning_lock:
            # Check if global replan is needed
            if (current_time - self.planning_state.last_replan_time > self.global_replan_interval):
                self.logger.info("Periodic global replanning")
                success = self.plan_mission(current_position, goal_position, current_obstacles)
                if success:
                    self.planning_state.last_replan_time = current_time
            
            # Check for local replanning triggers
            replan_needed = self._check_replan_triggers(current_position, current_obstacles)
            
            if replan_needed:
                local_path = self._execute_local_replan(current_position, current_obstacles)
                
                if local_path:
                    self.planning_state.local_path = local_path
                    self.planning_state.active_path = local_path
                    self.planning_state.local_plan_active = True
                    
                    self.planning_statistics['local_replans'] += 1
                    self.logger.info(f"Local replan executed: {len(local_path)} waypoints")
                else:
                    self.logger.warning("Local replanning failed")
            
            # Update waypoint following
            self._update_waypoint_following(current_position)
            
            return self.planning_state.active_path.copy()
    
    def _check_replan_triggers(self, current_position: Tuple[float, float, float],
                             obstacles: List[Tuple[float, float, float]]) -> bool:
        """
        Check if local replanning should be triggered.
        
        Args:
            current_position: Current position
            obstacles: Current obstacle positions
            
        Returns:
            True if replanning needed
        """
        if not self.planning_state.global_plan_valid:
            return False
        
        # Check obstacle proximity to current path
        for obstacle_pos in obstacles:
            distance_to_path = self._distance_to_path(obstacle_pos, self.planning_state.active_path)
            
            if distance_to_path < self.replanning_threshold:
                self.planning_statistics['replan_triggers'] += 1
                self.logger.info(f"Replan trigger: obstacle within {distance_to_path:.2f}m of path")
                return True
        
        # Check if current path is still valid
        if not self.global_planner._validate_path(self.planning_state.active_path):
            self.logger.info("Replan trigger: current path invalid")
            return True
        
        return False
    
    def _distance_to_path(self, point: Tuple[float, float, float],
                         path: List[Tuple[float, float, float]]) -> float:
        """
        Calculate minimum distance from point to path.
        
        Args:
            point: Point to check
            path: Path segments
            
        Returns:
            Minimum distance to path
        """
        if not path:
            return float('inf')
        
        point_array = np.array(point)
        min_distance = float('inf')
        
        # Check distance to each path segment
        for i in range(len(path) - 1):
            seg_start = np.array(path[i])
            seg_end = np.array(path[i + 1])
            
            # Distance to line segment
            seg_vec = seg_end - seg_start
            seg_length = np.linalg.norm(seg_vec)
            
            if seg_length > 0:
                # Project point onto line segment
                t = np.dot(point_array - seg_start, seg_vec) / (seg_length ** 2)
                t = np.clip(t, 0, 1)
                
                closest_point = seg_start + t * seg_vec
                distance = np.linalg.norm(point_array - closest_point)
                
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _execute_local_replan(self, current_position: Tuple[float, float, float],
                            obstacles: List[Tuple[float, float, float]]) -> Optional[List[Tuple[float, float, float]]]:
        """
        Execute S-RRT local replanning.
        
        Args:
            current_position: Current position
            obstacles: Dynamic obstacles
            
        Returns:
            Local path or None if failed
        """
        if not self.planning_state.global_path:
            return None
        
        # Determine local goal (next few waypoints from global path)
        local_goal = self._get_local_goal(current_position)
        
        if local_goal is None:
            return None
        
        # Execute S-RRT local planning
        local_result = self.local_planner.plan_safe_path(
            start=current_position,
            goal=local_goal,
            obstacles=obstacles,
            global_path_hint=self.planning_state.global_path
        )
        
        if local_result and local_result.get('success', False):
            return local_result['path']
        
        return None
    
    def _get_local_goal(self, current_position: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Get local goal for S-RRT planning.
        
        Args:
            current_position: Current position
            
        Returns:
            Local goal position
        """
        if not self.planning_state.global_path:
            return None
        
        current_pos_array = np.array(current_position)
        
        # Find waypoint ahead at lookahead distance
        for i, waypoint in enumerate(self.planning_state.global_path[self.planning_state.current_waypoint_index:]):
            distance = np.linalg.norm(np.array(waypoint) - current_pos_array)
            
            if distance >= self.lookahead_distance:
                return waypoint
        
        # If no waypoint at lookahead distance, use final goal
        if self.planning_state.global_path:
            return self.planning_state.global_path[-1]
        
        return None
    
    def _update_waypoint_following(self, current_position: Tuple[float, float, float]):
        """
        Update current waypoint following progress.
        
        Args:
            current_position: Current drone position
        """
        if not self.planning_state.active_path:
            return
        
        current_waypoint_idx = self.planning_state.current_waypoint_index
        
        if current_waypoint_idx >= len(self.planning_state.active_path):
            return  # Already at end of path
        
        # Check if current waypoint is reached
        current_waypoint = self.planning_state.active_path[current_waypoint_idx]
        distance_to_waypoint = np.linalg.norm(
            np.array(current_position) - np.array(current_waypoint)
        )
        
        if distance_to_waypoint <= self.waypoint_tolerance:
            # Advance to next waypoint
            self.planning_state.current_waypoint_index += 1
            self.logger.debug(f"Waypoint {current_waypoint_idx} reached, advancing to {self.planning_state.current_waypoint_index}")
    
    def get_current_path(self) -> List[Tuple[float, float, float]]:
        """Get current active path."""
        with self.planning_lock:
            return self.planning_state.active_path.copy()
    
    def get_next_waypoint(self, current_position: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Get next waypoint to follow.
        
        Args:
            current_position: Current position
            
        Returns:
            Next waypoint or None
        """
        with self.planning_lock:
            if not self.planning_state.active_path:
                return None
            
            # Return current target waypoint
            if self.planning_state.current_waypoint_index < len(self.planning_state.active_path):
                return self.planning_state.active_path[self.planning_state.current_waypoint_index]
            
            return None
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning statistics."""
        with self.planning_lock:
            stats = self.planning_statistics.copy()
            stats.update({
                'current_mode': self.planning_state.mode.value,
                'global_path_length': len(self.planning_state.global_path),
                'local_path_length': len(self.planning_state.local_path),
                'active_path_length': len(self.planning_state.active_path),
                'waypoint_progress': f"{self.planning_state.current_waypoint_index}/{len(self.planning_state.active_path)}",
                'global_plan_valid': self.planning_state.global_plan_valid,
                'local_plan_active': self.planning_state.local_plan_active
            })
            
            # Add planner-specific statistics
            stats['global_planner_stats'] = self.global_planner.get_statistics()
            stats['local_planner_stats'] = self.local_planner.get_statistics() if hasattr(self.local_planner, 'get_statistics') else {}
            
            return stats
    
    def reset(self):
        """Reset planning manager for new mission."""
        with self.planning_lock:
            self.planning_state = PlanningState()
        
        self.logger.info("Planner Manager reset")
