"""
A* Global Planner
Implements graph-based A* planning on 5-floor building grid.
Uses 26-neighborhood with vertical transitions for floor navigation.
"""

import numpy as np
import heapq
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from src.planning.global_planner.heuristics import AStarHeuristics
from src.planning.global_planner.occupancy_grid import OccupancyGrid3D

@dataclass
class AStarNode:
    """Node in A* search tree."""
    position: Tuple[int, int, int]  # Grid coordinates
    g_cost: float = float('inf')    # Cost from start
    h_cost: float = 0.0             # Heuristic cost to goal
    f_cost: float = float('inf')    # Total cost g + h
    parent: Optional['AStarNode'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

@dataclass
class PlanningResult:
    """Result of A* planning."""
    path: List[Tuple[float, float, float]]  # World coordinates path
    grid_path: List[Tuple[int, int, int]]   # Grid coordinates path
    total_cost: float
    planning_time: float
    nodes_expanded: int
    success: bool
    floor_transitions: int

class AStarPlanner:
    """
    A* global planner for 5-floor building navigation.
    Implements graph-based search on 3D occupancy grid.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Grid configuration (exactly matching report: 20×40×5, 0.5m cells)
        self.grid_config = {
            'cell_size': config.get('cell_size', 0.5),           # meters
            'building_dims': config.get('building_dims', {
                'length': 20.0, 'width': 40.0, 'height': 15.0   # 5 floors × 3m
            })
        }
        
        # A* parameters
        self.allow_diagonal = config.get('allow_diagonal', True)
        self.allow_vertical = config.get('allow_vertical', True)
        self.floor_transition_penalty = config.get('floor_transition_penalty', 2.0)  # As per report
        
        # 26-neighborhood as specified in report
        self.neighborhood = self._create_26_neighborhood()
        
        # Safety parameters
        self.obstacle_inflation = config.get('obstacle_inflation', 0.5)  # meters
        self.min_clearance = config.get('min_clearance', 0.3)           # meters
        
        # Performance limits
        self.max_iterations = config.get('max_iterations', 100000)
        self.max_planning_time = config.get('max_planning_time', 5.0)   # seconds
        
        # Initialize components
        self.occupancy_grid = OccupancyGrid3D(self.grid_config)
        self.heuristics = AStarHeuristics(config.get('heuristics', {}))
        
        # Statistics
        self.planning_statistics = {
            'total_plans': 0,
            'successful_plans': 0,
            'average_planning_time': 0.0,
            'average_nodes_expanded': 0.0,
            'average_path_length': 0.0
        }
        
        self.logger.info("A* Planner initialized")
        self.logger.info(f"Grid: {self.occupancy_grid.dimensions} cells, {self.grid_config['cell_size']}m resolution")
        self.logger.info(f"26-neighborhood: diagonal={self.allow_diagonal}, vertical={self.allow_vertical}")
        self.logger.info(f"Floor transition penalty: {self.floor_transition_penalty}")
    
    def _create_26_neighborhood(self) -> List[Tuple[int, int, int]]:
        """
        Create 26-neighborhood for 3D A* search.
        Includes all 26 adjacent cells (face, edge, vertex neighbors).
        """
        neighborhood = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip center cell
                    
                    # Apply movement constraints
                    if not self.allow_diagonal and (abs(dx) + abs(dy) + abs(dz) > 1):
                        continue  # Skip diagonal moves if disabled
                    
                    if not self.allow_vertical and dz != 0:
                        continue  # Skip vertical moves if disabled
                    
                    neighborhood.append((dx, dy, dz))
        
        self.logger.info(f"Created {len(neighborhood)}-neighborhood")
        return neighborhood
    
    def plan_path(self, start_pos: Tuple[float, float, float],
                  goal_pos: Tuple[float, float, float],
                  occupancy_grid: Optional[np.ndarray] = None) -> PlanningResult:
        """
        Plan path using A* algorithm.
        
        Args:
            start_pos: Start position in world coordinates
            goal_pos: Goal position in world coordinates
            occupancy_grid: Optional updated occupancy grid
            
        Returns:
            Planning result with path and statistics
        """
        planning_start = time.time()
        
        # Update occupancy grid if provided
        if occupancy_grid is not None:
            self.occupancy_grid.update_grid(occupancy_grid)
        
        # Convert to grid coordinates
        start_grid = self.occupancy_grid.world_to_grid(start_pos)
        goal_grid = self.occupancy_grid.world_to_grid(goal_pos)
        
        self.logger.info(f"Planning A* path: {start_pos} → {goal_pos}")
        self.logger.debug(f"Grid coordinates: {start_grid} → {goal_grid}")
        
        # Validate start and goal
        if not self._is_valid_position(start_grid) or not self._is_valid_position(goal_grid):
            return PlanningResult(
                path=[], grid_path=[], total_cost=float('inf'),
                planning_time=time.time() - planning_start,
                nodes_expanded=0, success=False, floor_transitions=0
            )
        
        # A* search
        result = self._astar_search(start_grid, goal_grid)
        
        # Convert path to world coordinates
        if result.success and result.grid_path:
            world_path = [self.occupancy_grid.grid_to_world(grid_pos) 
                         for grid_pos in result.grid_path]
            result.path = world_path
        
        # Update statistics
        self._update_statistics(result)
        
        planning_time = time.time() - planning_start
        result.planning_time = planning_time
        
        self.logger.info(f"A* planning completed in {planning_time:.3f}s")
        self.logger.info(f"Path length: {len(result.path)} waypoints, "
                        f"Cost: {result.total_cost:.2f}, "
                        f"Success: {result.success}")
        
        return result
    
    def _astar_search(self, start: Tuple[int, int, int], 
                     goal: Tuple[int, int, int]) -> PlanningResult:
        """
        Core A* search algorithm.
        
        Args:
            start: Start position in grid coordinates
            goal: Goal position in grid coordinates
            
        Returns:
            Planning result
        """
        # Initialize search structures
        open_set = []  # Priority queue
        closed_set: Set[Tuple[int, int, int]] = set()
        nodes: Dict[Tuple[int, int, int], AStarNode] = {}
        
        # Create start node
        start_node = AStarNode(position=start)
        start_node.g_cost = 0.0
        start_node.h_cost = self.heuristics.compute_heuristic(start, goal)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        nodes[start] = start_node
        heapq.heappush(open_set, start_node)
        
        nodes_expanded = 0
        search_start = time.time()
        
        while open_set:
            # Check time limit
            if time.time() - search_start > self.max_planning_time:
                self.logger.warning("A* search timed out")
                break
            
            # Check iteration limit
            if nodes_expanded >= self.max_iterations:
                self.logger.warning("A* search hit iteration limit")
                break
            
            # Get node with lowest f-cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # Check if goal reached
            if current_pos == goal:
                path = self._reconstruct_path(current_node)
                floor_transitions = self._count_floor_transitions(path)
                
                return PlanningResult(
                    path=[], grid_path=path, total_cost=current_node.g_cost,
                    planning_time=0.0, nodes_expanded=nodes_expanded,
                    success=True, floor_transitions=floor_transitions
                )
            
            # Add to closed set
            closed_set.add(current_pos)
            nodes_expanded += 1
            
            # Expand neighbors
            for dx, dy, dz in self.neighborhood:
                neighbor_pos = (
                    current_pos[0] + dx,
                    current_pos[1] + dy,
                    current_pos[2] + dz
                )
                
                # Skip if already processed
                if neighbor_pos in closed_set:
                    continue
                
                # Skip if invalid position
                if not self._is_valid_position(neighbor_pos):
                    continue
                
                # Calculate movement cost
                move_cost = self._calculate_movement_cost(current_pos, neighbor_pos)
                tentative_g_cost = current_node.g_cost + move_cost
                
                # Get or create neighbor node
                if neighbor_pos not in nodes:
                    neighbor_node = AStarNode(position=neighbor_pos)
                    nodes[neighbor_pos] = neighbor_node
                else:
                    neighbor_node = nodes[neighbor_pos]
                
                # Update if better path found
                if tentative_g_cost < neighbor_node.g_cost:
                    neighbor_node.parent = current_node
                    neighbor_node.g_cost = tentative_g_cost
                    neighbor_node.h_cost = self.heuristics.compute_heuristic(neighbor_pos, goal)
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    
                    # Add to open set if not already there
                    if neighbor_node not in open_set:
                        heapq.heappush(open_set, neighbor_node)
        
        # No path found
        return PlanningResult(
            path=[], grid_path=[], total_cost=float('inf'),
            planning_time=0.0, nodes_expanded=nodes_expanded,
            success=False, floor_transitions=0
        )
    
    def _is_valid_position(self, grid_pos: Tuple[int, int, int]) -> bool:
        """
        Check if grid position is valid and collision-free.
        
        Args:
            grid_pos: Position in grid coordinates
            
        Returns:
            True if valid and free
        """
        x, y, z = grid_pos
        
        # Check bounds
        if not (0 <= x < self.occupancy_grid.dimensions[0] and
                0 <= y < self.occupancy_grid.dimensions[1] and
                0 <= z < self.occupancy_grid.dimensions[2]):
            return False
        
        # Check occupancy
        return self.occupancy_grid.is_free(grid_pos)
    
    def _calculate_movement_cost(self, from_pos: Tuple[int, int, int],
                               to_pos: Tuple[int, int, int]) -> float:
        """
        Calculate cost of moving between adjacent grid cells.
        
        Args:
            from_pos: Source position
            to_pos: Target position
            
        Returns:
            Movement cost
        """
        # Base cost: Euclidean distance
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dz = to_pos[2] - from_pos[2]
        
        base_cost = np.sqrt(dx**2 + dy**2 + dz**2) * self.grid_config['cell_size']
        
        # Floor transition penalty (as specified in report)
        if dz != 0:  # Vertical movement (floor change)
            base_cost += self.floor_transition_penalty
        
        # Obstacle proximity penalty
        clearance_penalty = self._calculate_clearance_penalty(to_pos)
        
        return base_cost + clearance_penalty
    
    def _calculate_clearance_penalty(self, grid_pos: Tuple[int, int, int]) -> float:
        """
        Calculate penalty based on proximity to obstacles.
        
        Args:
            grid_pos: Grid position to check
            
        Returns:
            Clearance penalty
        """
        min_clearance = self.occupancy_grid.get_min_clearance(grid_pos)
        
        if min_clearance < self.min_clearance:
            # Exponential penalty for being too close to obstacles
            penalty_factor = (self.min_clearance - min_clearance) / self.min_clearance
            return penalty_factor ** 2 * 5.0  # Max 5.0 penalty
        
        return 0.0
    
    def _reconstruct_path(self, goal_node: AStarNode) -> List[Tuple[int, int, int]]:
        """
        Reconstruct path from goal node to start.
        
        Args:
            goal_node: Goal node with parent links
            
        Returns:
            Path as list of grid coordinates
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def _count_floor_transitions(self, grid_path: List[Tuple[int, int, int]]) -> int:
        """
        Count number of floor transitions in path.
        
        Args:
            grid_path: Path in grid coordinates
            
        Returns:
            Number of floor transitions
        """
        if len(grid_path) < 2:
            return 0
        
        transitions = 0
        current_floor = grid_path[0][2]
        
        for _, _, z in grid_path[1:]:
            if z != current_floor:
                transitions += 1
                current_floor = z
        
        return transitions
    
    def update_occupancy(self, obstacles: List[Tuple[float, float, float]]):
        """
        Update occupancy grid with new obstacle positions.
        
        Args:
            obstacles: List of obstacle positions in world coordinates
        """
        self.occupancy_grid.clear_dynamic_obstacles()
        
        for obstacle_pos in obstacles:
            self.occupancy_grid.mark_obstacle(obstacle_pos, dynamic=True)
        
        self.logger.debug(f"Updated occupancy grid with {len(obstacles)} obstacles")
    
    def replan_if_invalid(self, current_path: List[Tuple[float, float, float]],
                         current_position: Tuple[float, float, float],
                         goal_position: Tuple[float, float, float]) -> Optional[PlanningResult]:
        """
        Replan if current path is no longer valid.
        
        Args:
            current_path: Current planned path
            current_position: Current drone position
            goal_position: Goal position
            
        Returns:
            New planning result if replan needed, None otherwise
        """
        if not current_path:
            return self.plan_path(current_position, goal_position)
        
        # Check if path is still valid
        path_valid = self._validate_path(current_path)
        
        if not path_valid:
            self.logger.info("Current path invalid - replanning")
            return self.plan_path(current_position, goal_position)
        
        return None
    
    def _validate_path(self, world_path: List[Tuple[float, float, float]]) -> bool:
        """
        Validate if path is collision-free.
        
        Args:
            world_path: Path in world coordinates
            
        Returns:
            True if path is valid
        """
        for waypoint in world_path:
            grid_pos = self.occupancy_grid.world_to_grid(waypoint)
            
            if not self._is_valid_position(grid_pos):
                return False
        
        return True
    
    def _update_statistics(self, result: PlanningResult):
        """Update planning statistics."""
        self.planning_statistics['total_plans'] += 1
        
        if result.success:
            self.planning_statistics['successful_plans'] += 1
            
            # Running averages
            n = self.planning_statistics['successful_plans']
            
            self.planning_statistics['average_planning_time'] = (
                (n - 1) * self.planning_statistics['average_planning_time'] + result.planning_time
            ) / n
            
            self.planning_statistics['average_nodes_expanded'] = (
                (n - 1) * self.planning_statistics['average_nodes_expanded'] + result.nodes_expanded
            ) / n
            
            self.planning_statistics['average_path_length'] = (
                (n - 1) * self.planning_statistics['average_path_length'] + len(result.path)
            ) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        stats = self.planning_statistics.copy()
        
        if stats['total_plans'] > 0:
            stats['success_rate'] = stats['successful_plans'] / stats['total_plans']
        else:
            stats['success_rate'] = 0.0
        
        stats['grid_info'] = self.occupancy_grid.get_info()
        
        return stats
    
    def visualize_path(self, result: PlanningResult):
        """
        Visualize planned path (for debugging).
        
        Args:
            result: Planning result to visualize
        """
        if not result.success:
            self.logger.info("No path to visualize")
            return
        
        self.logger.info(f"Path visualization:")
        self.logger.info(f"  Total waypoints: {len(result.path)}")
        self.logger.info(f"  Floor transitions: {result.floor_transitions}")
        self.logger.info(f"  Planning time: {result.planning_time:.3f}s")
        self.logger.info(f"  Nodes expanded: {result.nodes_expanded}")
        
        # Floor breakdown
        if result.grid_path:
            floor_counts = defaultdict(int)
            for _, _, z in result.grid_path:
                floor = z // (int(3.0 / self.grid_config['cell_size']))  # 3m per floor
                floor_counts[floor] += 1
            
            self.logger.info(f"  Waypoints per floor: {dict(floor_counts)}")
    
    def get_path_info(self, world_path: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Get information about a path.
        
        Args:
            world_path: Path in world coordinates
            
        Returns:
            Path information dictionary
        """
        if not world_path:
            return {'empty_path': True}
        
        # Calculate path metrics
        total_distance = 0.0
        floor_changes = 0
        current_floor = None
        
        for i in range(1, len(world_path)):
            # Distance
            prev_pos = np.array(world_path[i-1])
            curr_pos = np.array(world_path[i])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
            
            # Floor changes
            floor = int(curr_pos[2] // 3.0)  # 3m per floor
            if current_floor is not None and floor != current_floor:
                floor_changes += 1
            current_floor = floor
        
        return {
            'total_waypoints': len(world_path),
            'total_distance': total_distance,
            'floor_transitions': floor_changes,
            'floors_visited': len(set(int(pos[2] // 3.0) for pos in world_path)),
            'average_waypoint_distance': total_distance / max(1, len(world_path) - 1),
            'start_position': world_path[0],
            'goal_position': world_path[-1]
        }
