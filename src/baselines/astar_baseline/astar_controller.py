import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import heapq
import json
from dataclasses import dataclass
from scipy.spatial.distance import euclidean

dataclass
class GridCell:

    x: int
    y: int
    z: int
    g_cost: float = float("inf")
    h_cost: float = 0.0
    f_cost: float = float("inf")
    parent: Optional["GridCell"] = None

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __lt__(self, other):
        return self.f_cost  other.f_cost

class AStarController:

    def __init__(self, config: Dict[str, Any], map_file: Optional[str] = None):

        self.floor_penalty = config.get("floor_penalty", 5.0)
        self.heuristic_weight = config.get("heuristic_weight", 1.0)

        if map_file:
            self._load_map_from_file(map_file, config)
        else:
            self._initialize_from_config(config)

        self.neighbors = self._generate_6_neighbors()

        self.current_path: List[Tuple[float, float, float]] = []
        self.path_index = 0

    def _initialize_from_config(self, config: Dict[str, Any]):

        print("  WARNING: Initializing from config (no map file provided)")
        print("  This is NOT recommended for research evaluation!")
        print("  Please generate map first: python src/environment/airsim_navigation.py")

        self.cell_size = config.get("cell_size", 0.5)

        test_bounds = config.get("test_bounds", [[0, 20], [0, 40], [0, 15]])
        self.world_bounds = np.array(test_bounds)

        self.floor_length = self.world_bounds[0, 1] - self.world_bounds[0, 0]
        self.floor_width = self.world_bounds[1, 1] - self.world_bounds[1, 0]
        total_height = self.world_bounds[2, 1] - self.world_bounds[2, 0]
        self.floor_height = 3.0

        self.grid_x = int(self.floor_length / self.cell_size)
        self.grid_y = int(self.floor_width / self.cell_size)
        self.grid_z = int(total_height / self.floor_height)
        self.floors = self.grid_z

        self.occupancy_grid = np.zeros(
            (self.grid_x, self.grid_y, self.grid_z), dtype=bool
        )

        print(f"  Initialized EMPTY test grid: {self.grid_x}{self.grid_y}{self.grid_z}")
        print(f"  Bounds: {self.world_bounds.tolist()}")

    def _load_map_from_file(self, map_file: str, config: Dict[str, Any]):

        print(f" Loading map from: {map_file}")

        with open(map_file, 'r') as f:
            metadata = json.load(f)

        self.cell_size = metadata['resolution']
        self.grid_x = metadata['dimensions']['x']
        self.grid_y = metadata['dimensions']['y']
        self.grid_z = metadata['dimensions']['z']

        bounds = metadata['bounds']
        self.world_bounds = np.array([
            [bounds['x_min'], bounds['x_max']],
            [bounds['y_min'], bounds['y_max']],
            [bounds['z_min'], bounds['z_max']]
        ])

        self.floor_length = self.world_bounds[0, 1] - self.world_bounds[0, 0]
        self.floor_width = self.world_bounds[1, 1] - self.world_bounds[1, 0]
        total_height = self.world_bounds[2, 1] - self.world_bounds[2, 0]
        self.floor_height = total_height / self.grid_z if self.grid_z  0 else 3.0
        self.floors = self.grid_z

        grid_file = metadata['files']['grid']
        loaded_grid = np.load(grid_file)

        self.occupancy_grid = (loaded_grid == 1)

        print(f" Loaded map: {self.grid_x}{self.grid_y}{self.grid_z} cells")
        print(f"   Bounds: X[{self.world_bounds[0,0]:.1f}, {self.world_bounds[0,1]:.1f}] "
              f"Y[{self.world_bounds[1,0]:.1f}, {self.world_bounds[1,1]:.1f}] "
              f"Z[{self.world_bounds[2,0]:.1f}, {self.world_bounds[2,1]:.1f}]")
        print(f"   Resolution: {self.cell_size} m/cell")
        print(f"   Occupied cells: {np.sum(self.occupancy_grid):,}")

    def _generate_6_neighbors(self) - List[Tuple[int, int, int]]:

        return [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

    def _generate_26_neighbors(self) - List[Tuple[int, int, int]]:

        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors.append((dx, dy, dz))
        return neighbors

    def update_occupancy_grid(self, obstacles: List[Tuple[float, float, float]]):

        self.occupancy_grid.fill(False)

        for obs_x, obs_y, obs_z in obstacles:
            grid_x = int(obs_x / self.cell_size)
            grid_y = int(obs_y / self.cell_size)
            grid_z = int(obs_z / self.floor_height)

            if (
                0 = grid_x  self.grid_x
                and 0 = grid_y  self.grid_y
                and 0 = grid_z  self.grid_z
            ):
                self.occupancy_grid[grid_x, grid_y, grid_z] = True

    def world_to_grid(
        self, world_pos: Tuple[float, float, float]
    ) - Tuple[int, int, int]:

        x_norm = (world_pos[0] - self.world_bounds[0, 0]) / self.cell_size
        y_norm = (world_pos[1] - self.world_bounds[1, 0]) / self.cell_size
        z_norm = (world_pos[2] - self.world_bounds[2, 0]) / self.cell_size

        grid_x = int(x_norm)
        grid_y = int(y_norm)
        grid_z = int(z_norm)

        grid_x = max(0, min(grid_x, self.grid_x - 1))
        grid_y = max(0, min(grid_y, self.grid_y - 1))
        grid_z = max(0, min(grid_z, self.grid_z - 1))

        return (grid_x, grid_y, grid_z)

    def grid_to_world(
        self, grid_pos: Tuple[int, int, int]
    ) - Tuple[float, float, float]:

        gx, gy, gz = grid_pos
        world_x = self.world_bounds[0, 0] + (gx + 0.5)  self.cell_size
        world_y = self.world_bounds[1, 0] + (gy + 0.5)  self.cell_size
        world_z = self.world_bounds[2, 0] + (gz + 0.5)  self.cell_size
        return (world_x, world_y, world_z)

    def heuristic(self, cell: GridCell, goal: GridCell) - float:

        dx = abs(cell.x - goal.x)  self.cell_size
        dy = abs(cell.y - goal.y)  self.cell_size
        dz = abs(cell.z - goal.z)  self.cell_size
        base_distance = np.sqrt(dx2 + dy2 + dz2)
        return self.heuristic_weight  base_distance

    def get_movement_cost(self, from_cell: GridCell, to_cell: GridCell) - float:

        dx = abs(to_cell.x - from_cell.x)  self.cell_size
        dy = abs(to_cell.y - from_cell.y)  self.cell_size
        dz = abs(to_cell.z - from_cell.z)  self.cell_size
        base_cost = np.sqrt(dx2 + dy2 + dz2)

        if to_cell.z != from_cell.z:
            base_cost += self.floor_penalty

        return base_cost

    def is_valid_cell(self, x: int, y: int, z: int) - bool:

        if not (0 = x  self.grid_x and 0 = y  self.grid_y and 0 = z  self.grid_z):
            return False
        return not self.occupancy_grid[x, y, z]

    def plan_path(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
    ) - List[Tuple[float, float, float]]:

        import time
        plan_start_time = time.time()

        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)

        start_cell = GridCell(start_grid[0], start_grid[1], start_grid[2])
        goal_cell = GridCell(goal_grid[0], goal_grid[1], goal_grid[2])

        start_cell.g_cost = 0.0
        start_cell.h_cost = self.heuristic(start_cell, goal_cell)
        start_cell.f_cost = start_cell.g_cost + start_cell.h_cost

        open_list = [start_cell]
        closed_set = set()
        nodes_expanded = 0
        last_progress_time = plan_start_time

        while open_list:
            current_time = time.time()
            elapsed = current_time - plan_start_time
            if current_time - last_progress_time = 5.0:
                print(f"    A searching... {nodes_expanded:,} nodes, {elapsed:.1f}s elapsed")
                last_progress_time = current_time

            current_cell = heapq.heappop(open_list)
            nodes_expanded += 1

            if current_cell == goal_cell:
                print(f"    Found path! Expanded {nodes_expanded:,} nodes in {elapsed:.1f}s")
                return self._reconstruct_path(current_cell)

            closed_set.add(current_cell)

            for dx, dy, dz in self.neighbors:
                neighbor_x = current_cell.x + dx
                neighbor_y = current_cell.y + dy
                neighbor_z = current_cell.z + dz

                if not self.is_valid_cell(neighbor_x, neighbor_y, neighbor_z):
                    continue

                neighbor = GridCell(neighbor_x, neighbor_y, neighbor_z)

                if neighbor in closed_set:
                    continue

                tentative_g = current_cell.g_cost + self.get_movement_cost(
                    current_cell, neighbor
                )

                neighbor_in_open = None
                for cell in open_list:
                    if cell == neighbor:
                        neighbor_in_open = cell
                        break

                if neighbor_in_open is None:
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = self.heuristic(neighbor, goal_cell)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    neighbor.parent = current_cell
                    heapq.heappush(open_list, neighbor)
                elif tentative_g  neighbor_in_open.g_cost:
                    neighbor_in_open.g_cost = tentative_g
                    neighbor_in_open.f_cost = (
                        neighbor_in_open.g_cost + neighbor_in_open.h_cost
                    )
                    neighbor_in_open.parent = current_cell

        return []

    def _reconstruct_path(
        self, goal_cell: GridCell
    ) - List[Tuple[float, float, float]]:

        path = []
        current = goal_cell

        while current is not None:
            world_pos = self.grid_to_world((current.x, current.y, current.z))
            path.append(world_pos)
            current = current.parent

        path.reverse()
        return path

    def set_path(self, path: List[Tuple[float, float, float]]):

        self.current_path = path
        self.path_index = 0

    def get_next_waypoint(
        self, current_pos: Tuple[float, float, float]
    ) - Optional[Tuple[float, float, float]]:

        if not self.current_path or self.path_index = len(self.current_path):
            return None

        current_waypoint = self.current_path[self.path_index]
        distance = euclidean(current_pos, current_waypoint)

        if distance  0.5:
            self.path_index += 1
            if self.path_index = len(self.current_path):
                return None

        return self.current_path[self.path_index]

    def is_path_valid(self, obstacles: List[Tuple[float, float, float]]) - bool:

        if not self.current_path:
            return True

        self.update_occupancy_grid(obstacles)

        for i in range(len(self.current_path)):
            world_pos = self.current_path[i]
            grid_pos = self.world_to_grid(world_pos)

            if not self.is_valid_cell(grid_pos[0], grid_pos[1], grid_pos[2]):
                return False

        return True
