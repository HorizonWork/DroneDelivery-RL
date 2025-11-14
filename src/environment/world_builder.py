import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

def _to_tuple3(
    values: Optional[Any], fallback: Tuple[float, float, float]
) - Tuple[float, float, float]:

    if values is None:
        values = fallback
    data = list(values)
    if len(data)  3:
        data.extend([0.0]  (3 - len(data)))
    return tuple(float(data[i]) for i in range(3))

class FloorType(Enum):

    RESIDENTIAL = "residential"
    OFFICE = "office"
    COMMERCIAL = "commercial"
    MIXED_USE = "mixed_use"

dataclass
class FloorSpec:

    floor_number: int
    floor_type: FloorType
    dimensions: Tuple[float, float, float]
    landing_targets: List[str]
    obstacle_density: float
    has_dynamic_obstacles: bool
    accessibility: float

dataclass
class BuildingSpec:

    floors: List[FloorSpec]
    total_height: float
    spawn_location: Tuple[float, float, float]
    vertical_connectors: List[str]
    emergency_exits: List[Tuple[float, float, float]]

class WorldBuilder:

    def __init__(self, config: Dict[str, Any] = None):

        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self.num_floors = self.config.get("num_floors", 5)
        self.floor_dimensions = self.config.get(
            "floor_dimensions",
            {"length": 20.0, "width": 40.0, "height": 3.0},
        )

        dynamic_cfg = self.config.get("dynamic_obstacles", {})

        self.enable_dynamic_obstacles = dynamic_cfg.get("enabled", True)
        self.num_dynamic_obstacles = dynamic_cfg.get("human_agents", 12)

        self.spawn_radius_m = float(dynamic_cfg.get("spawn_radius_m", 50.0))

        self.min_spawn_distance_from_drone_m = float(
            dynamic_cfg.get("min_spawn_distance_from_drone_m", 15.0)
        )

        if self.min_spawn_distance_from_drone_m = self.spawn_radius_m:
            self.logger.warning(
                f"min_spawn_distance ({self.min_spawn_distance_from_drone_m}) was = spawn_radius ({self.spawn_radius_m}). "
                f"Forcing min_distance to be radius/2 to ensure a valid spawn area."
            )
            self.min_spawn_distance_from_drone_m = self.spawn_radius_m / 2.0

        self.current_drone_spawn_location: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        self.building_spec = self._create_building_spec()
        self.static_obstacles: List[Tuple[float, float, float]] = []
        self.dynamic_obstacles: List[Dict[str, Any]] = []
        self.last_update_time = 0.0
        self.update_frequency = self.config.get("world_update_frequency", 10.0)
        self.airsim_bridge = None

        self.logger.info("World Builder initialized (Robust Final Version)")
        self.logger.info(
            f"Dynamic obstacles: {self.enable_dynamic_obstacles} ({self.num_dynamic_obstacles} agents)"
        )
        self.logger.info(
            f"SAFE ZONE radius around drone: {self.min_spawn_distance_from_drone_m}m"
        )

    def set_airsim_bridge(self, bridge):
        self.airsim_bridge = bridge
        self.logger.info("AirSim bridge connected to world builder")

    def update_spawn_reference(self, spawn_location: Tuple[float, float, float]):

        self.current_drone_spawn_location = spawn_location
        self.logger.debug(
            f"WorldBuilder origin for this episode is now drone spawn: {self.current_drone_spawn_location}"
        )

    def _create_building_spec(self) - BuildingSpec:
        floors = []
        for floor_num in range(1, self.num_floors + 1):
            floors.append(
                FloorSpec(
                    floor_number=floor_num,
                    floor_type=FloorType.OFFICE,
                    dimensions=(
                        self.floor_dimensions["length"],
                        self.floor_dimensions["width"],
                        self.floor_dimensions["height"],
                    ),
                    landing_targets=[
                        f"Landing_{floor_num}{i:02d}" for i in range(1, 7)
                    ],
                    obstacle_density=0.15,
                    has_dynamic_obstacles=self.enable_dynamic_obstacles,
                    accessibility=1.0,
                )
            )
        total_height = self.num_floors  self.floor_dimensions["height"]
        return BuildingSpec(
            floors=floors,
            total_height=total_height,
            spawn_location=self.current_drone_spawn_location,
            vertical_connectors=["stair_core_nw", "elevator_core_ne"],
            emergency_exits=[],
        )

    def build_world(self) - bool:

        self.logger.info("Building world...")
        try:
            self._generate_static_obstacles()
            if self.enable_dynamic_obstacles:
                self._initialize_dynamic_obstacles()
            self.logger.info("World building tasks completed.")
            return True
        except Exception as e:
            self.logger.error(f"World building failed: {e}")
            return False

    def _generate_static_obstacles(self):

        self.logger.warning("Static obstacle generation is DISABLED for stability.")
        self.static_obstacles.clear()

    def _initialize_dynamic_obstacles(self):

        self.dynamic_obstacles.clear()
        if not self.enable_dynamic_obstacles:
            return

        drone_x, drone_y, drone_z = self.current_drone_spawn_location
        self.logger.info(
            f"Generating {self.num_dynamic_obstacles} dynamic obstacles around drone at ({drone_x:.1f}, {drone_y:.1f})"
        )

        for i in range(self.num_dynamic_obstacles):
            angle = np.random.uniform(0, 2  np.pi)

            radius = np.random.uniform(
                self.min_spawn_distance_from_drone_m, self.spawn_radius_m
            )

            obs_x = drone_x + radius  np.cos(angle)
            obs_y = drone_y + radius  np.sin(angle)
            obs_z = drone_z

            agent = {
                "id": f"Human_{i + 1}",
                "position": [obs_x, obs_y, obs_z],
                "velocity": [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0],
            }
            self.dynamic_obstacles.append(agent)
            self.logger.debug(
                f"  - Spawned obstacle {i + 1} at ({obs_x:.1f}, {obs_y:.1f})"
            )

        self.logger.info(
            f"Successfully initialized {len(self.dynamic_obstacles)} dynamic obstacles."
        )

    def update_world_state(self):

        pass

    def get_obstacles(self) - Tuple[List, List[Tuple[float, float, float]]]:

        dynamic_positions = [
            tuple(agent["position"]) for agent in self.dynamic_obstacles
        ]
        return self.static_obstacles, dynamic_positions

    def reset_world(self):

        self.logger.debug("Resetting world state...")
        self._initialize_dynamic_obstacles()
        self.last_update_time = time.time()
        self.logger.debug("World state has been reset.")

    def _generate_floor_obstacles(
        self, floor_spec: FloorSpec
    ) - List[Tuple[float, float, float]]:
        self.logger.debug(
            f"'{self._generate_floor_obstacles.__name__}' is not used in the fixed version."
        )
        return []

    def _generate_wall_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:
        self.logger.debug(
            f"'{self._generate_wall_obstacles.__name__}' is not used in the fixed version."
        )
        return []

    def _generate_interior_obstacles(
        self, floor_spec: FloorSpec, floor_z: float, num_obstacles: int
    ) - List[Tuple[float, float, float]]:
        self.logger.debug(
            f"'{self._generate_interior_obstacles.__name__}' is not used in the fixed version."
        )
        return []

    def _generate_office_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:
        self.logger.debug(
            f"'{self._generate_office_obstacles.__name__}' is not used in the fixed version."
        )
        return []

    def _generate_residential_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:
        self.logger.debug(
            f"'{_generate_residential_obstacles.__name__}' is not used in the fixed version."
        )
        return []

    def get_world_bounds(self) - Dict[str, float]:
        return {
            "x_min": -self.spawn_radius_m,
            "x_max": self.spawn_radius_m,
            "y_min": -self.spawn_radius_m,
            "y_max": self.spawn_radius_m,
            "z_min": 0.0,
            "z_max": self.building_spec.total_height,
        }
