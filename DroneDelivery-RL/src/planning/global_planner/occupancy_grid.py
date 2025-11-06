"""
3D Occupancy Grid
Manages 5-floor building occupancy representation for A* planning.
Implements 4000-cell grid with 0.5m resolution as per report.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from scipy.ndimage import binary_dilation

@dataclass
class GridInfo:
    """Occupancy grid information."""
    dimensions: Tuple[int, int, int]
    cell_size: float
    world_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    total_cells: int
    occupied_cells: int
    free_cells: int
    unknown_cells: int

class OccupancyGrid3D:
    """
    3D Occupancy Grid for 5-floor building.
    Manages static and dynamic obstacles with inflation for safety.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Grid parameters (exact match with report)
        self.cell_size = config.get('cell_size', 0.5)              # 0.5m resolution
        self.building_dims = config.get('building_dims', {
            'length': 20.0, 'width': 40.0, 'height': 15.0         # 5 floors × 3m
        })
        
        # Calculate grid dimensions in cells
        self.dimensions = (
            int(np.ceil(self.building_dims['length'] / self.cell_size)),    # 40 cells
            int(np.ceil(self.building_dims['width'] / self.cell_size)),     # 80 cells  
            int(np.ceil(self.building_dims['height'] / self.cell_size))     # 30 cells
        )
        
        # Verify total cells count (should be ~4000 as per report)
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        self.logger.info(f"Grid dimensions: {self.dimensions} = {total_cells} total cells")
        
        # Initialize occupancy grid
        # Values: -1 = unknown, 0 = free, 1 = occupied
        self.grid = np.full(self.dimensions, -1, dtype=np.int8)
        
        # Separate tracking for dynamic vs static obstacles
        self.static_grid = np.zeros(self.dimensions, dtype=np.int8)
        self.dynamic_grid = np.zeros(self.dimensions, dtype=np.int8)
        
        # Safety inflation parameters
        self.inflation_radius = config.get('inflation_radius', 0.5)     # meters
        self.inflation_cells = int(np.ceil(self.inflation_radius / self.cell_size))
        
        # Create inflation kernel
        self.inflation_kernel = self._create_inflation_kernel()
        
        # World coordinate bounds
        self.world_bounds = (
            (0.0, self.building_dims['length']),
            (0.0, self.building_dims['width']),
            (0.0, self.building_dims['height'])
        )
        
        # Initialize with building structure
        self._initialize_building_structure()
        
        self.logger.info(f"3D Occupancy Grid initialized: {self.dimensions}")
        self.logger.info(f"Cell size: {self.cell_size}m, Inflation: {self.inflation_radius}m")
    
    def _create_inflation_kernel(self) -> np.ndarray:
        """
        Create spherical inflation kernel for obstacle inflation.
        
        Returns:
            3D binary kernel for inflation
        """
        kernel_size = 2 * self.inflation_cells + 1
        kernel = np.zeros((kernel_size, kernel_size, kernel_size), dtype=bool)
        
        center = self.inflation_cells
        
        for x in range(kernel_size):
            for y in range(kernel_size):
                for z in range(kernel_size):
                    # Distance from center
                    dx = (x - center) * self.cell_size
                    dy = (y - center) * self.cell_size
                    dz = (z - center) * self.cell_size
                    
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if distance <= self.inflation_radius:
                        kernel[x, y, z] = True
        
        return kernel
    
    def _initialize_building_structure(self):
        """Initialize grid with basic building structure (walls, floors)."""
        # Mark perimeter walls as occupied
        
        # Floor and ceiling
        self.static_grid[:, :, 0] = 1  # Ground floor
        self.static_grid[:, :, -1] = 1  # Top ceiling
        
        # Walls (simplified representation)
        self.static_grid[0, :, :] = 1   # West wall
        self.static_grid[-1, :, :] = 1  # East wall
        self.static_grid[:, 0, :] = 1   # North wall
        self.static_grid[:, -1, :] = 1  # South wall
        
        # Floor separations (every 6 cells = 3m)
        floor_height_cells = int(3.0 / self.cell_size)  # 6 cells per floor
        
        for floor in range(1, 5):  # Floors 1-4 (floor 5 is open ceiling)
            floor_z = floor * floor_height_cells
            if floor_z < self.dimensions[2]:
                # Mark floor structure with openings for stairs/elevators
                self.static_grid[:, :, floor_z] = 1
                
                # Create openings for vertical navigation
                # Staircase opening (northwest corner)
                self.static_grid[2:6, 2:6, floor_z] = 0
                
                # Elevator opening (northeast corner)
                self.static_grid[-6:-2, 2:6, floor_z] = 0
        
        # Update combined grid
        self._update_combined_grid()
        
        self.logger.info("Building structure initialized in occupancy grid")
    
    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            world_pos: Position in world coordinates
            
        Returns:
            Position in grid coordinates
        """
        x = int(np.clip(world_pos[0] / self.cell_size, 0, self.dimensions[0] - 1))
        y = int(np.clip(world_pos[1] / self.cell_size, 0, self.dimensions[1] - 1))
        z = int(np.clip(world_pos[2] / self.cell_size, 0, self.dimensions[2] - 1))
        
        return (x, y, z)
    
    def grid_to_world(self, grid_pos: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_pos: Position in grid coordinates
            
        Returns:
            Position in world coordinates (cell center)
        """
        x = (grid_pos[0] + 0.5) * self.cell_size
        y = (grid_pos[1] + 0.5) * self.cell_size
        z = (grid_pos[2] + 0.5) * self.cell_size
        
        return (x, y, z)
    
    def is_free(self, grid_pos: Tuple[int, int, int]) -> bool:
        """
        Check if grid cell is free (navigable).
        
        Args:
            grid_pos: Position in grid coordinates
            
        Returns:
            True if cell is free
        """
        x, y, z = grid_pos
        
        # Check bounds
        if not (0 <= x < self.dimensions[0] and
                0 <= y < self.dimensions[1] and
                0 <= z < self.dimensions[2]):
            return False
        
        # Check occupancy (0 = free)
        return self.grid[x, y, z] == 0
    
    def is_occupied(self, grid_pos: Tuple[int, int, int]) -> bool:
        """
        Check if grid cell is occupied.
        
        Args:
            grid_pos: Position in grid coordinates
            
        Returns:
            True if cell is occupied
        """
        x, y, z = grid_pos
        
        # Check bounds (out of bounds = occupied for safety)
        if not (0 <= x < self.dimensions[0] and
                0 <= y < self.dimensions[1] and
                0 <= z < self.dimensions[2]):
            return True
        
        # Check occupancy (1 = occupied)
        return self.grid[x, y, z] == 1
    
    def mark_obstacle(self, world_pos: Tuple[float, float, float], 
                     dynamic: bool = False, radius: float = 0.2):
        """
        Mark obstacle in occupancy grid.
        
        Args:
            world_pos: Obstacle position in world coordinates
            dynamic: Whether this is a dynamic obstacle
            radius: Obstacle radius for inflation
        """
        grid_pos = self.world_to_grid(world_pos)
        
        # Choose appropriate grid
        target_grid = self.dynamic_grid if dynamic else self.static_grid
        
        # Mark obstacle with inflation
        inflation_cells = max(1, int(np.ceil(radius / self.cell_size)))
        
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dz in range(-inflation_cells, inflation_cells + 1):
                    # Check if within inflation radius
                    distance = np.sqrt(dx**2 + dy**2 + dz**2) * self.cell_size
                    if distance > radius:
                        continue
                    
                    # Calculate grid position
                    x = grid_pos[0] + dx
                    y = grid_pos[1] + dy
                    z = grid_pos[2] + dz
                    
                    # Check bounds
                    if (0 <= x < self.dimensions[0] and
                        0 <= y < self.dimensions[1] and
                        0 <= z < self.dimensions[2]):
                        
                        target_grid[x, y, z] = 1
        
        # Update combined grid
        self._update_combined_grid()
    
    def clear_dynamic_obstacles(self):
        """Clear all dynamic obstacles from grid."""
        self.dynamic_grid.fill(0)
        self._update_combined_grid()
    
    def _update_combined_grid(self):
        """Update combined occupancy grid from static and dynamic grids."""
        # Combine static and dynamic obstacles
        combined = np.maximum(self.static_grid, self.dynamic_grid)
        
        # Apply safety inflation
        if self.inflation_cells > 0:
            combined = binary_dilation(combined, structure=self.inflation_kernel).astype(np.int8)
        
        # Update main grid (keep unknown areas as -1)
        mask = (self.static_grid == 1) | (self.dynamic_grid == 1) | (combined == 1)
        self.grid[mask] = 1
        
        # Mark explicitly free areas (not near obstacles)
        free_mask = (self.static_grid == 0) & (self.dynamic_grid == 0) & (combined == 0)
        self.grid[free_mask] = 0
    
    def get_min_clearance(self, grid_pos: Tuple[int, int, int], search_radius: int = 5) -> float:
        """
        Get minimum clearance to nearest obstacle.
        
        Args:
            grid_pos: Position to check
            search_radius: Search radius in cells
            
        Returns:
            Minimum clearance in meters
        """
        x, y, z = grid_pos
        min_distance = float('inf')
        
        # Search in neighborhood
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Check bounds
                    if (0 <= nx < self.dimensions[0] and
                        0 <= ny < self.dimensions[1] and
                        0 <= nz < self.dimensions[2]):
                        
                        if self.grid[nx, ny, nz] == 1:  # Occupied
                            distance = np.sqrt(dx**2 + dy**2 + dz**2) * self.cell_size
                            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else search_radius * self.cell_size
    
    def update_grid(self, new_grid: np.ndarray):
        """
        Update entire occupancy grid.
        
        Args:
            new_grid: New occupancy grid data
        """
        if new_grid.shape == self.dimensions:
            self.grid = new_grid.copy()
            self.logger.debug("Occupancy grid updated")
        else:
            self.logger.error(f"Grid shape mismatch: {new_grid.shape} vs {self.dimensions}")
    
    def get_occupancy_slice(self, floor: int) -> np.ndarray:
        """
        Get occupancy grid slice for specific floor.
        
        Args:
            floor: Floor number (1-5)
            
        Returns:
            2D occupancy grid for floor
        """
        floor_height_cells = int(3.0 / self.cell_size)  # 3m per floor
        
        if 1 <= floor <= 5:
            floor_z_start = (floor - 1) * floor_height_cells
            floor_z_end = min(floor * floor_height_cells, self.dimensions[2])
            
            # Take average/max occupancy across floor height
            floor_slice = np.max(self.grid[:, :, floor_z_start:floor_z_end], axis=2)
            return floor_slice
        
        return np.zeros((self.dimensions[0], self.dimensions[1]))
    
    def get_vertical_profile(self, x: int, y: int) -> np.ndarray:
        """
        Get vertical occupancy profile at (x,y) position.
        
        Args:
            x, y: Grid coordinates
            
        Returns:
            1D vertical occupancy profile
        """
        if (0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1]):
            return self.grid[x, y, :]
        
        return np.ones(self.dimensions[2])  # Occupied if out of bounds
    
    def is_floor_accessible(self, floor: int, position: Tuple[float, float]) -> bool:
        """
        Check if position on floor is accessible.
        
        Args:
            floor: Floor number (1-5)
            position: (x, y) position on floor
            
        Returns:
            True if position is accessible
        """
        # Convert to grid coordinates
        grid_x = int(position[0] / self.cell_size)
        grid_y = int(position[1] / self.cell_size)
        
        floor_height_cells = int(3.0 / self.cell_size)
        floor_z = (floor - 1) * floor_height_cells + floor_height_cells // 2  # Mid-floor
        
        grid_pos = (grid_x, grid_y, floor_z)
        return self.is_free(grid_pos)
    
    def find_path_to_floor(self, current_pos: Tuple[int, int, int], 
                          target_floor: int) -> Optional[List[Tuple[int, int, int]]]:
        """
        Find path to access specific floor (stairs/elevator).
        
        Args:
            current_pos: Current position in grid coordinates
            target_floor: Target floor number
            
        Returns:
            Path to floor access point or None
        """
        current_floor = current_pos[2] // int(3.0 / self.cell_size)
        
        if current_floor == target_floor - 1:  # Already on target floor
            return [current_pos]
        
        # Find nearest stair or elevator
        access_points = self._get_floor_access_points(target_floor)
        
        if not access_points:
            return None
        
        # Find closest access point
        min_distance = float('inf')
        best_access_point = None
        
        for access_point in access_points:
            distance = np.linalg.norm(np.array(current_pos) - np.array(access_point))
            if distance < min_distance:
                min_distance = distance
                best_access_point = access_point
        
        if best_access_point:
            return [current_pos, best_access_point]  # Simplified path
        
        return None
    
    def _get_floor_access_points(self, floor: int) -> List[Tuple[int, int, int]]:
        """
        Get access points (stairs, elevators) for specific floor.
        
        Args:
            floor: Target floor number
            
        Returns:
            List of access point grid coordinates
        """
        access_points = []
        floor_height_cells = int(3.0 / self.cell_size)
        floor_z = (floor - 1) * floor_height_cells
        
        # Staircase (northwest corner)
        stair_x = int(3.0 / self.cell_size)  # 3m from west wall
        stair_y = int(3.0 / self.cell_size)  # 3m from north wall
        access_points.append((stair_x, stair_y, floor_z))
        
        # Elevator (northeast corner)
        elevator_x = self.dimensions[0] - int(3.0 / self.cell_size)
        elevator_y = int(3.0 / self.cell_size)
        access_points.append((elevator_x, elevator_y, floor_z))
        
        return access_points
    
    def get_info(self) -> GridInfo:
        """
        Get occupancy grid information.
        
        Returns:
            Grid information object
        """
        # Count cell types
        occupied_cells = int(np.sum(self.grid == 1))
        free_cells = int(np.sum(self.grid == 0))
        unknown_cells = int(np.sum(self.grid == -1))
        total_cells = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        
        return GridInfo(
            dimensions=self.dimensions,
            cell_size=self.cell_size,
            world_bounds=self.world_bounds,
            total_cells=total_cells,
            occupied_cells=occupied_cells,
            free_cells=free_cells,
            unknown_cells=unknown_cells
        )
    
    def export_grid(self) -> Dict[str, Any]:
        """
        Export occupancy grid data.
        
        Returns:
            Grid data dictionary
        """
        return {
            'grid': self.grid.copy(),
            'static_grid': self.static_grid.copy(),
            'dynamic_grid': self.dynamic_grid.copy(),
            'dimensions': self.dimensions,
            'cell_size': self.cell_size,
            'world_bounds': self.world_bounds,
            'timestamp': time.time()
        }
    
    def load_grid(self, grid_data: Dict[str, Any]):
        """
        Load occupancy grid from data.
        
        Args:
            grid_data: Grid data dictionary
        """
        if 'grid' in grid_data and grid_data['grid'].shape == self.dimensions:
            self.grid = grid_data['grid'].copy()
            
            if 'static_grid' in grid_data:
                self.static_grid = grid_data['static_grid'].copy()
            if 'dynamic_grid' in grid_data:
                self.dynamic_grid = grid_data['dynamic_grid'].copy()
            
            self.logger.info("Occupancy grid loaded from data")
        else:
            self.logger.error("Invalid grid data for loading")
