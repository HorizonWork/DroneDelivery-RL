"""
S-RRT Local Planner
Safety-oriented RRT* variant for dynamic obstacle avoidance.
Implements Equation (3) cost function with clearance and curvature.
"""

import numpy as np
import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from .cost_functions import SRRTCostFunction
from .dynamic_obstacles import DynamicObstacleTracker
from .safety_checker import SafetyChecker

@dataclass
class SRRTNode:
    """Node in S-RRT tree."""
    position: np.ndarray
    parent: Optional['SRRTNode'] = None
    children: List['SRRTNode'] = field(default_factory=list)
    cost_from_start: float = float('inf')
    clearance: float = float('inf')
    
    def __lt__(self, other):
        return self.cost_from_start < other.cost_from_start

@dataclass
class SRRTResult:
    """S-RRT planning result."""
    path: List[Tuple[float, float, float]]
    total_cost: float
    planning_time: float
    nodes_generated: int
    success: bool
    safety_clearance: float

class SRRTPlanner:
    """
    Safety-oriented RRT* (S-RRT) local planner.
    Implements Equation (3): C = ℓ + λc(1/dmin)² + λκ·κ²
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # S-RRT parameters
        self.max_iterations = config.get('max_iterations', 5000)
        self.max_planning_time = config.get('max_planning_time', 1.0)     # 1 second
        self.step_size = config.get('step_size', 0.5)                    # meters
        self.goal_bias = config.get('goal_bias', 0.1)                    # 10% goal sampling
        self.rewiring_radius = config.get('rewiring_radius', 2.0)        # meters
        
        # Cost function parameters (Equation 3)
        self.lambda_c = config.get('lambda_c', 1.0)                      # Clearance weight λc
        self.lambda_kappa = config.get('lambda_kappa', 2.0)              # Curvature weight λκ
        
        # Safety parameters
        self.min_clearance = config.get('min_clearance', 0.5)            # meters
        self.safety_margin = config.get('safety_margin', 0.3)            # meters
        self.prediction_horizon = config.get('prediction_horizon', 2.0)   # seconds
        
        # Search space
        self.search_radius = config.get('search_radius', 10.0)           # meters around current position
        
        # Initialize components
        self.cost_function = SRRTCostFunction(config.get('cost_function', {}))
        self.obstacle_tracker = DynamicObstacleTracker(config.get('obstacle_tracker', {}))
        self.safety_checker = SafetyChecker(config.get('safety_checker', {}))
        
        # Tree storage
        self.tree_nodes: List[SRRTNode] = []
        self.node_index: Dict[Tuple[float, float, float], int] = {}
        
        # Statistics
        self.planning_stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'average_planning_time': 0.0,
            'average_nodes_generated': 0.0,
            'average_cost': 0.0
        }
        
        self.logger.info("S-RRT Local Planner initialized")
        self.logger.info(f"Cost function: ℓ + {self.lambda_c}·(1/dmin)² + {self.lambda_kappa}·κ²")
        self.logger.info(f"Max iterations: {self.max_iterations}, Step size: {self.step_size}m")
        self.logger.info(f"Safety clearance: {self.min_clearance}m")
    
    def plan_safe_path(self, start: Tuple[float, float, float],
                      goal: Tuple[float, float, float],
                      obstacles: List[Tuple[float, float, float]],
                      global_path_hint: Optional[List[Tuple[float, float, float]]] = None) -> Dict[str, Any]:
        """
        Plan safe path using S-RRT algorithm.
        
        Args:
            start: Start position
            goal: Goal position
            obstacles: Dynamic obstacle positions
            global_path_hint: Global path for guidance
            
        Returns:
            Planning result dictionary
        """
        planning_start = time.time()
        
        self.logger.info(f"S-RRT planning: {start} → {goal}")
        self.logger.info(f"Dynamic obstacles: {len(obstacles)}")
        
        # Update obstacle tracker
        self.obstacle_tracker.update_obstacles(obstacles)
        
        # Initialize tree
        self._initialize_tree(start)
        
        # Define search region around current position and goal
        search_region = self._define_search_region(start, goal)
        
        nodes_generated = 0
        best_goal_node = None
        
        # S-RRT main loop
        for iteration in range(self.max_iterations):
            # Check time limit
            if time.time() - planning_start > self.max_planning_time:
                self.logger.warning("S-RRT planning timed out")
                break
            
            # Sample point in search space
            if random.random() < self.goal_bias:
                # Goal-biased sampling
                sample_point = np.array(goal)
            else:
                # Random sampling in search region
                sample_point = self._sample_in_region(search_region)
            
            # Find nearest node in tree
            nearest_node = self._find_nearest_node(sample_point)
            
            # Steer towards sample point
            new_position = self._steer(nearest_node.position, sample_point)
            
            # Check collision and safety
            if not self._is_safe_motion(nearest_node.position, new_position, obstacles):
                continue
            
            # Create new node
            new_node = SRRTNode(position=new_position)
            new_node.cost_from_start = self._calculate_cost(nearest_node, new_node, obstacles)
            new_node.parent = nearest_node
            nearest_node.children.append(new_node)
            
            self.tree_nodes.append(new_node)
            nodes_generated += 1
            
            # RRT* rewiring
            self._rewire_tree(new_node, obstacles)
            
            # Check if goal is reachable
            goal_distance = np.linalg.norm(new_node.position - np.array(goal))
            if goal_distance <= self.step_size:
                # Check direct connection to goal
                if self._is_safe_motion(new_node.position, np.array(goal), obstacles):
                    # Create goal node
                    goal_node = SRRTNode(position=np.array(goal))
                    goal_node.cost_from_start = self._calculate_cost(new_node, goal_node, obstacles)
                    goal_node.parent = new_node
                    
                    if best_goal_node is None or goal_node.cost_from_start < best_goal_node.cost_from_start:
                        best_goal_node = goal_node
        
        planning_time = time.time() - planning_start
        
        # Extract path if solution found
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            total_cost = best_goal_node.cost_from_start
            success = True
            
            # Calculate path safety clearance
            safety_clearance = self._calculate_path_clearance(path, obstacles)
            
            self.logger.info(f"S-RRT successful: {len(path)} waypoints, cost: {total_cost:.2f}")
        else:
            path = []
            total_cost = float('inf')
            success = False
            safety_clearance = 0.0
            
            self.logger.warning("S-RRT failed to find safe path")
        
        # Update statistics
        self._update_statistics(planning_time, nodes_generated, total_cost, success)
        
        result = SRRTResult(
            path=[(float(p[0]), float(p[1]), float(p[2])) for p in path] if success else [],
            total_cost=total_cost,
            planning_time=planning_time,
            nodes_generated=nodes_generated,
            success=success,
            safety_clearance=safety_clearance
        )
        
        return {
            'success': result.success,
            'path': result.path,
            'cost': result.total_cost,
            'planning_time': result.planning_time,
            'nodes_generated': result.nodes_generated,
            'safety_clearance': result.safety_clearance
        }
    
    def _initialize_tree(self, start: Tuple[float, float, float]):
        """Initialize S-RRT tree with start node."""
        self.tree_nodes.clear()
        self.node_index.clear()
        
        start_node = SRRTNode(position=np.array(start))
        start_node.cost_from_start = 0.0
        
        self.tree_nodes.append(start_node)
        self.node_index[start] = 0
    
    def _define_search_region(self, start: Tuple[float, float, float],
                            goal: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Define search region for sampling.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Search region bounds
        """
        start_array = np.array(start)
        goal_array = np.array(goal)
        
        # Create region encompassing start, goal, and search radius
        region_center = (start_array + goal_array) / 2
        region_size = np.linalg.norm(goal_array - start_array) + 2 * self.search_radius
        
        return {
            'center': region_center,
            'radius': region_size / 2,
            'bounds': {
                'x_min': region_center[0] - region_size / 2,
                'x_max': region_center[0] + region_size / 2,
                'y_min': region_center[1] - region_size / 2,
                'y_max': region_center[1] + region_size / 2,
                'z_min': region_center[2] - region_size / 2,
                'z_max': region_center[2] + region_size / 2
            }
        }
    
    def _sample_in_region(self, search_region: Dict[str, float]) -> np.ndarray:
        """Sample random point in search region."""
        bounds = search_region['bounds']
        
        sample = np.array([
            random.uniform(bounds['x_min'], bounds['x_max']),
            random.uniform(bounds['y_min'], bounds['y_max']),
            random.uniform(bounds['z_min'], bounds['z_max'])
        ])
        
        return sample
    
    def _find_nearest_node(self, sample_point: np.ndarray) -> SRRTNode:
        """Find nearest node in tree to sample point."""
        min_distance = float('inf')
        nearest_node = self.tree_nodes[0]
        
        for node in self.tree_nodes:
            distance = np.linalg.norm(node.position - sample_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """
        Steer from one position towards another with step size limit.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            New position within step size
        """
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos
        else:
            unit_direction = direction / distance
            return from_pos + unit_direction * self.step_size
    
    def _is_safe_motion(self, from_pos: np.ndarray, to_pos: np.ndarray,
                       obstacles: List[Tuple[float, float, float]]) -> bool:
        """
        Check if motion between positions is safe.
        
        Args:
            from_pos: Start position
            to_pos: End position
            obstacles: Dynamic obstacles
            
        Returns:
            True if motion is safe
        """
        return self.safety_checker.check_path_safety(
            from_pos, to_pos, obstacles, self.prediction_horizon
        )
    
    def _calculate_cost(self, parent_node: SRRTNode, child_node: SRRTNode,
                      obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Calculate S-RRT cost using Equation (3).
        
        Args:
            parent_node: Parent node
            child_node: Child node
            obstacles: Current obstacles
            
        Returns:
            Total cost from start to child
        """
        return self.cost_function.calculate_edge_cost(
            parent_node, child_node, obstacles
        ) + parent_node.cost_from_start
    
    def _rewire_tree(self, new_node: SRRTNode, obstacles: List[Tuple[float, float, float]]):
        """
        RRT* rewiring step to optimize tree.
        
        Args:
            new_node: Newly added node
            obstacles: Current obstacles
        """
        # Find nearby nodes within rewiring radius
        nearby_nodes = []
        
        for node in self.tree_nodes[:-1]:  # Exclude the new node itself
            distance = np.linalg.norm(node.position - new_node.position)
            if distance <= self.rewiring_radius:
                nearby_nodes.append(node)
        
        # Check if rewiring through new_node improves any nearby node
        for nearby_node in nearby_nodes:
            # Calculate cost if rewired through new_node
            if self._is_safe_motion(new_node.position, nearby_node.position, obstacles):
                new_cost = self._calculate_cost(new_node, nearby_node, obstacles)
                
                if new_cost < nearby_node.cost_from_start:
                    # Rewire: update parent
                    if nearby_node.parent:
                        nearby_node.parent.children.remove(nearby_node)
                    
                    nearby_node.parent = new_node
                    nearby_node.cost_from_start = new_cost
                    new_node.children.append(nearby_node)
                    
                    # Propagate cost updates to descendants
                    self._propagate_cost_updates(nearby_node, obstacles)
    
    def _propagate_cost_updates(self, node: SRRTNode, obstacles: List[Tuple[float, float, float]]):
        """
        Propagate cost updates to descendants after rewiring.
        
        Args:
            node: Node whose cost was updated
            obstacles: Current obstacles
        """
        for child in node.children:
            old_cost = child.cost_from_start
            new_cost = self._calculate_cost(node, child, obstacles)
            
            if new_cost < old_cost:
                child.cost_from_start = new_cost
                self._propagate_cost_updates(child, obstacles)
    
    def _extract_path(self, goal_node: SRRTNode) -> List[np.ndarray]:
        """
        Extract path from start to goal node.
        
        Args:
            goal_node: Goal node with parent links
            
        Returns:
            Path as list of positions
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position.copy())
            current = current.parent
        
        path.reverse()
        return path
    
    def _calculate_path_clearance(self, path: List[np.ndarray],
                                obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Calculate minimum clearance along path.
        
        Args:
            path: Path positions
            obstacles: Obstacle positions
            
        Returns:
            Minimum clearance in meters
        """
        if not path or not obstacles:
            return float('inf')
        
        min_clearance = float('inf')
        obstacle_positions = [np.array(obs) for obs in obstacles]
        
        for waypoint in path:
            for obstacle_pos in obstacle_positions:
                distance = np.linalg.norm(waypoint - obstacle_pos)
                min_clearance = min(min_clearance, distance)
        
        return min_clearance
    
    def _update_statistics(self, planning_time: float, nodes_generated: int,
                          total_cost: float, success: bool):
        """Update planning statistics."""
        self.planning_stats['total_plans'] += 1
        
        if success:
            self.planning_stats['successful_plans'] += 1
            
            # Update running averages
            n = self.planning_stats['successful_plans']
            
            self.planning_stats['average_planning_time'] = (
                (n - 1) * self.planning_stats['average_planning_time'] + planning_time
            ) / n
            
            self.planning_stats['average_nodes_generated'] = (
                (n - 1) * self.planning_stats['average_nodes_generated'] + nodes_generated
            ) / n
            
            self.planning_stats['average_cost'] = (
                (n - 1) * self.planning_stats['average_cost'] + total_cost
            ) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get S-RRT planning statistics."""
        stats = self.planning_stats.copy()
        
        if stats['total_plans'] > 0:
            stats['success_rate'] = stats['successful_plans'] / stats['total_plans']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def reset(self):
        """Reset S-RRT planner."""
        self.tree_nodes.clear()
        self.node_index.clear()
        self.obstacle_tracker.reset()
        
        self.logger.debug("S-RRT planner reset")
    
    def visualize_tree(self) -> Dict[str, Any]:
        """
        Get tree visualization data.
        
        Returns:
            Tree visualization data
        """
        if not self.tree_nodes:
            return {'nodes': [], 'edges': []}
        
        nodes = [node.position.tolist() for node in self.tree_nodes]
        edges = []
        
        for node in self.tree_nodes:
            if node.parent is not None:
                edges.append([
                    node.parent.position.tolist(),
                    node.position.tolist()
                ])
        
        return {
            'nodes': nodes,
            'edges': edges,
            'num_nodes': len(nodes),
            'tree_depth': self._calculate_tree_depth()
        }
    
    def _calculate_tree_depth(self) -> int:
        """Calculate maximum depth of S-RRT tree."""
        if not self.tree_nodes:
            return 0
        
        max_depth = 0
        
        for node in self.tree_nodes:
            depth = 0
            current = node
            
            while current.parent is not None:
                depth += 1
                current = current.parent
            
            max_depth = max(max_depth, depth)
        
        return max_depth
