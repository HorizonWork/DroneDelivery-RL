"""
A* Only Baseline Controller
Follows global A* path using PID controller with no learning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import heapq
from dataclasses import dataclass
from scipy.spatial.distance import euclidean

@dataclass
class GridCell:
    """Represents a cell in the 3D occupancy grid."""
    x: int
    y: int
    z: int  # floor
    g_cost: float = float('inf')
    h_cost: float = 0.0
    f_cost: float = float('inf')
    parent: Optional['GridCell'] = None
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarController:
    """
    A* global planner for 5-floor building navigation.
    Implements exact A* algorithm as described in Section 4.2 of report.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Environment configuration
        self.floors = config.get('floors', 5)
        self.floor_length = config.get('floor_length', 20.0)  # meters
        self.floor_width = config.get('floor_width', 40.0)    # meters
        self.floor_height = config.get('floor_height', 3.0)   # meters
        self.cell_size = config.get('cell_size', 0.5)         # meters
        
        # A* parameters
        self.floor_penalty = config.get('floor_penalty', 5.0)  # φ_floor penalty
        
        # Grid dimensions
        self.grid_x = int(self.floor_length / self.cell_size)  # 40 cells
        self.grid_y = int(self.floor_width / self.cell_size)   # 80 cells
        self.grid_z = self.floors                              # 5 floors
        
        # 26-neighborhood connectivity as per report
        self.neighbors = self._generate_26_neighbors()
        
        # Occupancy grid [x, y, z] -> bool (True = occupied, False = free)
        self.occupancy_grid = np.zeros((self.grid_x, self.grid_y, self.grid_z), dtype=bool)
        
        # Current path
        self.current_path: List[Tuple[float, float, float]] = []
        self.path_index = 0
        
    def _generate_26_neighbors(self) -> List[Tuple[int, int, int]]:
        """Generate 26-neighborhood offsets for 3D grid connectivity."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))
        return neighbors
    
    def update_occupancy_grid(self, obstacles: List[Tuple[float, float, float]]):
        """Update occupancy grid with current obstacle positions."""
        # Reset grid
        self.occupancy_grid.fill(False)
        
        # Add obstacles
        for obs_x, obs_y, obs_z in obstacles:
            grid_x = int(obs_x / self.cell_size)
            grid_y = int(obs_y / self.cell_size)
            grid_z = int(obs_z / self.floor_height)
            
            if (0 <= grid_x < self.grid_x and 
                0 <= grid_y < self.grid_y and
                0 <= grid_z < self.grid_z):
                self.occupancy_grid[grid_x, grid_y, grid_z] = True
    
    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        x, y, z = world_pos
        grid_x = int(x / self.cell_size)
        grid_y = int(y / self.cell_size)
        grid_z = int(z / self.floor_height)
        return (grid_x, grid_y, grid_z)
    
    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates."""
        gx, gy, gz = grid_pos
        world_x = (gx + 0.5) * self.cell_size
        world_y = (gy + 0.5) * self.cell_size
        world_z = (gz + 0.5) * self.floor_height
        return (world_x, world_y, world_z)
    
    def heuristic(self, cell: GridCell, goal: GridCell) -> float:
        """
        A* heuristic function: straight-line distance to goal.
        As specified in Section 4.2: h(n) = straight-line distance to goal.
        """
        dx = abs(cell.x - goal.x) * self.cell_size
        dy = abs(cell.y - goal.y) * self.cell_size
        dz = abs(cell.z - goal.z) * self.floor_height
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def get_movement_cost(self, from_cell: GridCell, to_cell: GridCell) -> float:
        """
        Calculate movement cost between adjacent cells.
        Includes floor transition penalty φ_floor as per Section 4.2.
        """
        # Base Euclidean distance
        dx = abs(to_cell.x - from_cell.x) * self.cell_size
        dy = abs(to_cell.y - from_cell.y) * self.cell_size  
        dz = abs(to_cell.z - from_cell.z) * self.floor_height
        base_cost = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Add floor transition penalty
        if to_cell.z != from_cell.z:
            base_cost += self.floor_penalty
            
        return base_cost
    
    def is_valid_cell(self, x: int, y: int, z: int) -> bool:
        """Check if grid cell is within bounds and not occupied."""
        if not (0 <= x < self.grid_x and 0 <= y < self.grid_y and 0 <= z < self.grid_z):
            return False
        return not self.occupancy_grid[x, y, z]
    
    def plan_path(self, start_pos: Tuple[float, float, float], 
                  goal_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        A* path planning on 5-floor grid.
        Returns path as list of world coordinates.
        """
        # Convert world coordinates to grid
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # Create start and goal cells
        start_cell = GridCell(start_grid[0], start_grid[1], start_grid[2])
        goal_cell = GridCell(goal_grid[0], goal_grid[1], goal_grid[2])
        
        start_cell.g_cost = 0.0
        start_cell.h_cost = self.heuristic(start_cell, goal_cell)
        start_cell.f_cost = start_cell.g_cost + start_cell.h_cost
        
        # A* search
        open_list = [start_cell]
        closed_set = set()
        
        while open_list:
            # Get cell with lowest f_cost
            current_cell = heapq.heappop(open_list)
            
            # Check if goal reached
            if current_cell == goal_cell:
                return self._reconstruct_path(current_cell)
            
            closed_set.add(current_cell)
            
            # Explore 26 neighbors
            for dx, dy, dz in self.neighbors:
                neighbor_x = current_cell.x + dx
                neighbor_y = current_cell.y + dy
                neighbor_z = current_cell.z + dz
                
                # Check validity
                if not self.is_valid_cell(neighbor_x, neighbor_y, neighbor_z):
                    continue
                
                neighbor = GridCell(neighbor_x, neighbor_y, neighbor_z)
                
                if neighbor in closed_set:
                    continue
                
                # Calculate costs
                tentative_g = current_cell.g_cost + self.get_movement_cost(current_cell, neighbor)
                
                # Check if this path is better
                neighbor_in_open = None
                for cell in open_list:
                    if cell == neighbor:
                        neighbor_in_open = cell
                        break
                
                if neighbor_in_open is None:
                    # New cell
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor, goal_cell)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    neighbor.parent = current_cell
                    heapq.heappush(open_list, neighbor)
                elif tentative_g < neighbor_in_open.g_cost:
                    # Better path found
                    neighbor_in_open.g_cost = tentative_g
                    neighbor_in_open.f_cost = neighbor_in_open.g_cost + neighbor_in_open.h_cost
                    neighbor_in_open.parent = current_cell
        
        # No path found
        return []
    
    def _reconstruct_path(self, goal_cell: GridCell) -> List[Tuple[float, float, float]]:
        """Reconstruct path from goal to start using parent pointers."""
        path = []
        current = goal_cell
        
        while current is not None:
            world_pos = self.grid_to_world((current.x, current.y, current.z))
            path.append(world_pos)
            current = current.parent
        
        path.reverse()
        return path
    
    def set_path(self, path: List[Tuple[float, float, float]]):
        """Set current path to follow."""
        self.current_path = path
        self.path_index = 0
    
    def get_next_waypoint(self, current_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Get next waypoint in path based on current position."""
        if not self.current_path or self.path_index >= len(self.current_path):
            return None
        
        # Check if we're close to current waypoint
        current_waypoint = self.current_path[self.path_index]
        distance = euclidean(current_pos, current_waypoint)
        
        if distance < 0.5:  # Within 0.5m tolerance
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                return None
        
        return self.current_path[self.path_index]
    
    def is_path_valid(self, obstacles: List[Tuple[float, float, float]]) -> bool:
        """Check if current path is still collision-free."""
        if not self.current_path:
            return True
        
        # Update occupancy grid
        self.update_occupancy_grid(obstacles)
        
        # Check each path segment
        for i in range(len(self.current_path)):
            world_pos = self.current_path[i]
            grid_pos = self.world_to_grid(world_pos)
            
            if not self.is_valid_cell(grid_pos[0], grid_pos[1], grid_pos[2]):
                return False
        
        return True
