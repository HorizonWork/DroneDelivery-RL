import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from scipy.ndimage import binary_dilation

dataclass
class GridInfo:

    dimensions: Tuple[int, int, int]
    cell_size: float
    world_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    total_cells: int
    occupied_cells: int
    free_cells: int
    unknown_cells: int

class OccupancyGrid3D:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.cell_size = config.get('cell_size', 0.5)
        self.building_dims = config.get('building_dims', {
            'length': 20.0, 'width': 40.0, 'height': 15.0
        })

        self.dimensions = (
            int(np.ceil(self.building_dims['length'] / self.cell_size)),
            int(np.ceil(self.building_dims['width'] / self.cell_size)),
            int(np.ceil(self.building_dims['height'] / self.cell_size))
        )

        total_cells = self.dimensions[0]  self.dimensions[1]  self.dimensions[2]
        self.logger.info(f"Grid dimensions: {self.dimensions} = {total_cells} total cells")

        self.grid = np.full(self.dimensions, -1, dtype=np.int8)

        self.static_grid = np.zeros(self.dimensions, dtype=np.int8)
        self.dynamic_grid = np.zeros(self.dimensions, dtype=np.int8)

        self.inflation_radius = config.get('inflation_radius', 0.5)
        self.inflation_cells = int(np.ceil(self.inflation_radius / self.cell_size))

        self.inflation_kernel = self._create_inflation_kernel()

        self.world_bounds = (
            (0.0, self.building_dims['length']),
            (0.0, self.building_dims['width']),
            (0.0, self.building_dims['height'])
        )

        self._initialize_building_structure()

        self.logger.info(f"3D Occupancy Grid initialized: {self.dimensions}")
        self.logger.info(f"Cell size: {self.cell_size}m, Inflation: {self.inflation_radius}m")

    def _create_inflation_kernel(self) - np.ndarray:

        kernel_size = 2  self.inflation_cells + 1
        kernel = np.zeros((kernel_size, kernel_size, kernel_size), dtype=bool)

        center = self.inflation_cells

        for x in range(kernel_size):
            for y in range(kernel_size):
                for z in range(kernel_size):
                    dx = (x - center)  self.cell_size
                    dy = (y - center)  self.cell_size
                    dz = (z - center)  self.cell_size

                    distance = np.sqrt(dx2 + dy2 + dz2)

                    if distance = self.inflation_radius:
                        kernel[x, y, z] = True

        return kernel

    def _initialize_building_structure(self):

        self.static_grid[:, :, 0] = 1
        self.static_grid[:, :, -1] = 1

        self.static_grid[0, :, :] = 1
        self.static_grid[-1, :, :] = 1
        self.static_grid[:, 0, :] = 1
        self.static_grid[:, -1, :] = 1

        floor_height_cells = int(3.0 / self.cell_size)

        for floor in range(1, 5):
            floor_z = floor  floor_height_cells
            if floor_z  self.dimensions[2]:
                self.static_grid[:, :, floor_z] = 1

                self.static_grid[2:6, 2:6, floor_z] = 0

                self.static_grid[-6:-2, 2:6, floor_z] = 0

        self._update_combined_grid()

        self.logger.info("Building structure initialized in occupancy grid")

    def world_to_grid(self, world_pos: Tuple[float, float, float]) - Tuple[int, int, int]:

        x = int(np.clip(world_pos[0] / self.cell_size, 0, self.dimensions[0] - 1))
        y = int(np.clip(world_pos[1] / self.cell_size, 0, self.dimensions[1] - 1))
        z = int(np.clip(world_pos[2] / self.cell_size, 0, self.dimensions[2] - 1))

        return (x, y, z)

    def grid_to_world(self, grid_pos: Tuple[int, int, int]) - Tuple[float, float, float]:

        x = (grid_pos[0] + 0.5)  self.cell_size
        y = (grid_pos[1] + 0.5)  self.cell_size
        z = (grid_pos[2] + 0.5)  self.cell_size

        return (x, y, z)

    def is_free(self, grid_pos: Tuple[int, int, int]) - bool:

        x, y, z = grid_pos

        if not (0 = x  self.dimensions[0] and
                0 = y  self.dimensions[1] and
                0 = z  self.dimensions[2]):
            return False

        return self.grid[x, y, z] == 0

    def is_occupied(self, grid_pos: Tuple[int, int, int]) - bool:

        x, y, z = grid_pos

        if not (0 = x  self.dimensions[0] and
                0 = y  self.dimensions[1] and
                0 = z  self.dimensions[2]):
            return True

        return self.grid[x, y, z] == 1

    def mark_obstacle(self, world_pos: Tuple[float, float, float],
                     dynamic: bool = False, radius: float = 0.2):

        grid_pos = self.world_to_grid(world_pos)

        target_grid = self.dynamic_grid if dynamic else self.static_grid

        inflation_cells = max(1, int(np.ceil(radius / self.cell_size)))

        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dz in range(-inflation_cells, inflation_cells + 1):
                    distance = np.sqrt(dx2 + dy2 + dz2)  self.cell_size
                    if distance  radius:
                        continue

                    x = grid_pos[0] + dx
                    y = grid_pos[1] + dy
                    z = grid_pos[2] + dz

                    if (0 = x  self.dimensions[0] and
                        0 = y  self.dimensions[1] and
                        0 = z  self.dimensions[2]):

                        target_grid[x, y, z] = 1

        self._update_combined_grid()

    def clear_dynamic_obstacles(self):

        self.dynamic_grid.fill(0)
        self._update_combined_grid()

    def _update_combined_grid(self):

        combined = np.maximum(self.static_grid, self.dynamic_grid)

        if self.inflation_cells  0:
            combined = binary_dilation(combined, structure=self.inflation_kernel).astype(np.int8)

        mask = (self.static_grid == 1)  (self.dynamic_grid == 1)  (combined == 1)
        self.grid[mask] = 1

        free_mask = (self.static_grid == 0)  (self.dynamic_grid == 0)  (combined == 0)
        self.grid[free_mask] = 0

    def get_min_clearance(self, grid_pos: Tuple[int, int, int], search_radius: int = 5) - float:

        x, y, z = grid_pos
        min_distance = float('inf')

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):

                    nx, ny, nz = x + dx, y + dy, z + dz

                    if (0 = nx  self.dimensions[0] and
                        0 = ny  self.dimensions[1] and
                        0 = nz  self.dimensions[2]):

                        if self.grid[nx, ny, nz] == 1:
                            distance = np.sqrt(dx2 + dy2 + dz2)  self.cell_size
                            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else search_radius  self.cell_size

    def update_grid(self, new_grid: np.ndarray):

        if new_grid.shape == self.dimensions:
            self.grid = new_grid.copy()
            self.logger.debug("Occupancy grid updated")
        else:
            self.logger.error(f"Grid shape mismatch: {new_grid.shape} vs {self.dimensions}")

    def get_occupancy_slice(self, floor: int) - np.ndarray:

        floor_height_cells = int(3.0 / self.cell_size)

        if 1 = floor = 5:
            floor_z_start = (floor - 1)  floor_height_cells
            floor_z_end = min(floor  floor_height_cells, self.dimensions[2])

            floor_slice = np.max(self.grid[:, :, floor_z_start:floor_z_end], axis=2)
            return floor_slice

        return np.zeros((self.dimensions[0], self.dimensions[1]))

    def get_vertical_profile(self, x: int, y: int) - np.ndarray:

        if (0 = x  self.dimensions[0] and 0 = y  self.dimensions[1]):
            return self.grid[x, y, :]

        return np.ones(self.dimensions[2])

    def is_floor_accessible(self, floor: int, position: Tuple[float, float]) - bool:

        grid_x = int(position[0] / self.cell_size)
        grid_y = int(position[1] / self.cell_size)

        floor_height_cells = int(3.0 / self.cell_size)
        floor_z = (floor - 1)  floor_height_cells + floor_height_cells

        grid_pos = (grid_x, grid_y, floor_z)
        return self.is_free(grid_pos)

    def find_path_to_floor(self, current_pos: Tuple[int, int, int],
                          target_floor: int) - Optional[List[Tuple[int, int, int]]]:

        current_floor = current_pos[2]

        if current_floor == target_floor - 1:
            return [current_pos]

        access_points = self._get_floor_access_points(target_floor)

        if not access_points:
            return None

        min_distance = float('inf')
        best_access_point = None

        for access_point in access_points:
            distance = np.linalg.norm(np.array(current_pos) - np.array(access_point))
            if distance  min_distance:
                min_distance = distance
                best_access_point = access_point

        if best_access_point:
            return [current_pos, best_access_point]

        return None

    def _get_floor_access_points(self, floor: int) - List[Tuple[int, int, int]]:

        access_points = []
        floor_height_cells = int(3.0 / self.cell_size)
        floor_z = (floor - 1)  floor_height_cells

        stair_x = int(3.0 / self.cell_size)
        stair_y = int(3.0 / self.cell_size)
        access_points.append((stair_x, stair_y, floor_z))

        elevator_x = self.dimensions[0] - int(3.0 / self.cell_size)
        elevator_y = int(3.0 / self.cell_size)
        access_points.append((elevator_x, elevator_y, floor_z))

        return access_points

    def get_info(self) - GridInfo:

        occupied_cells = int(np.sum(self.grid == 1))
        free_cells = int(np.sum(self.grid == 0))
        unknown_cells = int(np.sum(self.grid == -1))
        total_cells = self.dimensions[0]  self.dimensions[1]  self.dimensions[2]

        return GridInfo(
            dimensions=self.dimensions,
            cell_size=self.cell_size,
            world_bounds=self.world_bounds,
            total_cells=total_cells,
            occupied_cells=occupied_cells,
            free_cells=free_cells,
            unknown_cells=unknown_cells
        )

    def export_grid(self) - Dict[str, Any]:

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

        if 'grid' in grid_data and grid_data['grid'].shape == self.dimensions:
            self.grid = grid_data['grid'].copy()

            if 'static_grid' in grid_data:
                self.static_grid = grid_data['static_grid'].copy()
            if 'dynamic_grid' in grid_data:
                self.dynamic_grid = grid_data['dynamic_grid'].copy()

            self.logger.info("Occupancy grid loaded from data")
        else:
            self.logger.error("Invalid grid data for loading")
