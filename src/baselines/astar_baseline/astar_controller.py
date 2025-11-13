"""
A* Only Baseline Controller
Follows global A* path using PID controller with no learning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import heapq
import json
from dataclasses import dataclass
from scipy.spatial.distance import euclidean


@dataclass
class GridCell:
    """Represents a cell in the 3D occupancy grid."""

    x: int
    y: int
    z: int  # floor
    g_cost: float = float("inf")
    h_cost: float = 0.0
    f_cost: float = float("inf")
    parent: Optional["GridCell"] = None

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
    Loads occupancy grid from AirSim map generator.
    """

    def __init__(self, config: Dict[str, Any], map_file: Optional[str] = None):
        """
        Initialize A* controller.
        
        Args:
            config: Configuration dictionary
            map_file: Path to map metadata JSON file (optional)
        """
        # A* parameters
        self.floor_penalty = config.get("floor_penalty", 5.0)  # φ_floor penalty

        # Grid will be loaded from map or initialized from config
        if map_file:
            self._load_map_from_file(map_file, config)
        else:
            self._initialize_from_config(config)

        # 26-neighborhood connectivity as per report
        self.neighbors = self._generate_26_neighbors()

        # Current path
        self.current_path: List[Tuple[float, float, float]] = []
        self.path_index = 0
    
    def _initialize_from_config(self, config: Dict[str, Any]):
        """
        Initialize grid from configuration (DEPRECATED - for testing only).
        
        WARNING: For research purposes, ALWAYS use map_file parameter!
        This fallback should only be used for unit testing.
        """
        print("⚠️  WARNING: Initializing from config (no map file provided)")
        print("⚠️  This is NOT recommended for research evaluation!")
        print("⚠️  Please generate map first: python src/environment/airsim_navigation.py")
        
        # Minimal grid for testing only
        self.cell_size = config.get("cell_size", 0.5)  # meters
        
        # Default test environment (small)
        test_bounds = config.get("test_bounds", [[0, 20], [0, 40], [0, 15]])
        self.world_bounds = np.array(test_bounds)
        
        # Calculate grid dimensions
        self.floor_length = self.world_bounds[0, 1] - self.world_bounds[0, 0]
        self.floor_width = self.world_bounds[1, 1] - self.world_bounds[1, 0]
        total_height = self.world_bounds[2, 1] - self.world_bounds[2, 0]
        self.floor_height = 3.0  # Assumed
        
        self.grid_x = int(self.floor_length / self.cell_size)
        self.grid_y = int(self.floor_width / self.cell_size)
        self.grid_z = int(total_height / self.floor_height)
        self.floors = self.grid_z

        # Empty occupancy grid (no obstacles!)
        self.occupancy_grid = np.zeros(
            (self.grid_x, self.grid_y, self.grid_z), dtype=bool
        )
        
        print(f"⚠️  Initialized EMPTY test grid: {self.grid_x}×{self.grid_y}×{self.grid_z}")
        print(f"⚠️  Bounds: {self.world_bounds.tolist()}")
    
    def _load_map_from_file(self, map_file: str, config: Dict[str, Any]):
        """Load occupancy grid from AirSim map file"""
        print(f"📂 Loading map from: {map_file}")
        
        with open(map_file, 'r') as f:
            metadata = json.load(f)
        
        # Load grid parameters from metadata
        self.cell_size = metadata['resolution']
        self.grid_x = metadata['dimensions']['x']
        self.grid_y = metadata['dimensions']['y']
        self.grid_z = metadata['dimensions']['z']
        
        # Load world bounds
        bounds = metadata['bounds']
        self.world_bounds = np.array([
            [bounds['x_min'], bounds['x_max']],
            [bounds['y_min'], bounds['y_max']],
            [bounds['z_min'], bounds['z_max']]
        ])
        
        # Calculate derived parameters
        self.floor_length = self.world_bounds[0, 1] - self.world_bounds[0, 0]
        self.floor_width = self.world_bounds[1, 1] - self.world_bounds[1, 0]
        total_height = self.world_bounds[2, 1] - self.world_bounds[2, 0]
        self.floor_height = total_height / self.grid_z if self.grid_z > 0 else 3.0
        self.floors = self.grid_z
        
        # Load occupancy grid
        grid_file = metadata['files']['grid']
        loaded_grid = np.load(grid_file)
        
        # Convert to boolean (1 = occupied, 0/-1 = free)
        self.occupancy_grid = (loaded_grid == 1)
        
        print(f"✅ Loaded map: {self.grid_x}×{self.grid_y}×{self.grid_z} cells")
        print(f"   Bounds: X[{self.world_bounds[0,0]:.1f}, {self.world_bounds[0,1]:.1f}] "
              f"Y[{self.world_bounds[1,0]:.1f}, {self.world_bounds[1,1]:.1f}] "
              f"Z[{self.world_bounds[2,0]:.1f}, {self.world_bounds[2,1]:.1f}]")
        print(f"   Resolution: {self.cell_size} m/cell")
        print(f"   Occupied cells: {np.sum(self.occupancy_grid):,}")


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

            if (
                0 <= grid_x < self.grid_x
                and 0 <= grid_y < self.grid_y
                and 0 <= grid_z < self.grid_z
            ):
                self.occupancy_grid[grid_x, grid_y, grid_z] = True

    def world_to_grid(
        self, world_pos: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Convert world coordinates to grid coordinates."""
        # Normalize to grid space using actual bounds
        x_norm = (world_pos[0] - self.world_bounds[0, 0]) / self.cell_size
        y_norm = (world_pos[1] - self.world_bounds[1, 0]) / self.cell_size
        z_norm = (world_pos[2] - self.world_bounds[2, 0]) / self.floor_height
        
        grid_x = int(x_norm)
        grid_y = int(y_norm)
        grid_z = int(z_norm)
        
        # Clamp to valid range
        grid_x = max(0, min(grid_x, self.grid_x - 1))
        grid_y = max(0, min(grid_y, self.grid_y - 1))
        grid_z = max(0, min(grid_z, self.grid_z - 1))
        
        return (grid_x, grid_y, grid_z)

    def grid_to_world(
        self, grid_pos: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates (cell center)."""
        gx, gy, gz = grid_pos
        world_x = self.world_bounds[0, 0] + (gx + 0.5) * self.cell_size
        world_y = self.world_bounds[1, 0] + (gy + 0.5) * self.cell_size
        world_z = self.world_bounds[2, 0] + (gz + 0.5) * self.floor_height
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

    def plan_path(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
    ) -> List[Tuple[float, float, float]]:
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
                tentative_g = current_cell.g_cost + self.get_movement_cost(
                    current_cell, neighbor
                )

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
                    neighbor_in_open.f_cost = (
                        neighbor_in_open.g_cost + neighbor_in_open.h_cost
                    )
                    neighbor_in_open.parent = current_cell

        # No path found
        return []

    def _reconstruct_path(
        self, goal_cell: GridCell
    ) -> List[Tuple[float, float, float]]:
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

    def get_next_waypoint(
        self, current_pos: Tuple[float, float, float]
    ) -> Optional[Tuple[float, float, float]]:
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
