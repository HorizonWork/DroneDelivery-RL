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

        self.num_floors = config.get("num_floors", 5)
        self.floor_dimensions = config.get(
            "floor_dimensions",
            {
                "length": 20.0,
                "width": 40.0,
                "height": 3.0,
            },
        )

        self.cell_size = config.get("cell_size", 0.5)
        self.total_cells = config.get("total_cells", 4000)
        self.spawn_location = _to_tuple3(
            config.get("spawn_location", [6000.0, -3000.0, 300.0]),
            (6000.0, -3000.0, 300.0),
        )
        self.world_origin_offset = _to_tuple3(
            config.get("world_origin_offset", self.spawn_location),
            self.spawn_location,
        )
        dynamic_cfg = config.get("dynamic_obstacles", {})
        if isinstance(dynamic_cfg, dict):
            self.enable_dynamic_obstacles = dynamic_cfg.get("enabled", True)
            self.num_human_agents = dynamic_cfg.get("human_agents", 12)
            self.human_speed_range = dynamic_cfg.get("speed_range_mps", [0.8, 1.5])
            self.min_dynamic_spawn_distance = float(
                dynamic_cfg.get("min_spawn_distance_m", 5.0)
            )
        else:
            self.enable_dynamic_obstacles = bool(dynamic_cfg)
            self.num_human_agents = config.get("num_human_agents", 12)
            self.human_speed_range = config.get("human_speed_range", [0.8, 1.5])
            self.min_dynamic_spawn_distance = float(
                config.get("min_spawn_distance_m", 5.0)
            )
        self.current_spawn_reference = self.spawn_location
        self.spawn_local_offset: Optional[Tuple[float, float, float]] = None
        self._alignment_warning_emitted = False

        self.building_spec = self._create_building_spec()
        self._validate_alignment()

        self.static_obstacles: List[Tuple[float, float, float]] = []
        self.dynamic_obstacles: List[Dict[str, Any]] = []
        self.last_update_time = 0.0
        self.update_frequency = config.get("world_update_frequency", 10.0)

        self.airsim_bridge = None

        self.logger.info("World Builder initialized")
        self.logger.info(f"Building: {self.num_floors} floors, {self.floor_dimensions}")
        self.logger.info(
            f"Dynamic obstacles: {self.enable_dynamic_obstacles} ({self.num_human_agents} agents)"
        )
        self.logger.info(f"World origin offset (AirSim): {self.world_origin_offset}")

    def set_airsim_bridge(self, bridge):

        self.airsim_bridge = bridge
        self.logger.info("AirSim bridge connected to world builder")

    def update_spawn_reference(self, spawn_location: Tuple[float, float, float]):

        self.current_spawn_reference = _to_tuple3(spawn_location, self.spawn_location)
        self.spawn_location = self.current_spawn_reference
        if self.building_spec:
            self.building_spec.spawn_location = self.current_spawn_reference
        self._validate_alignment()
        self.logger.debug(
            "World builder spawn reference updated to s", self.current_spawn_reference
        )

    def local_to_world(self, local_point: Tuple[float, float, float]) - Tuple[float, float, float]:

        origin = self.world_origin_offset
        return (
            local_point[0] + origin[0],
            local_point[1] + origin[1],
            local_point[2] + origin[2],
        )

    def _validate_alignment(self):

        origin = np.array(self.world_origin_offset, dtype=np.float32)
        spawn = np.array(self.current_spawn_reference, dtype=np.float32)
        offset = spawn - origin
        self.spawn_local_offset = tuple(offset.tolist())
        self._alignment_warning_emitted = getattr(
            self, "_alignment_warning_emitted", False
        )

        within_x = 0.0 = offset[0] = self.floor_dimensions["length"]
        within_y = 0.0 = offset[1] = self.floor_dimensions["width"]

        if within_x and within_y:
            self.logger.debug(
                "Spawn alignment OK. Local offset (.2f, .2f, .2f) m within .1fx.1f m bounds",
                offset[0],
                offset[1],
                offset[2],
                self.floor_dimensions["length"],
                self.floor_dimensions["width"],
            )
        else:
            if not self._alignment_warning_emitted:
                self.logger.warning(
                    (
                        "Spawn alignment mismatch: local offset (.2f, .2f) m "
                        "exceeds floor bounds .1fx.1f m. Adjust world_origin_offset or spawn_location."
                    ),
                    offset[0],
                    offset[1],
                    self.floor_dimensions["length"],
                    self.floor_dimensions["width"],
                )
                self._alignment_warning_emitted = True

    def _is_safe_spawn_distance(self, candidate_xy: Tuple[float, float]) - bool:

        if (
            self.spawn_local_offset is None
            or self.min_dynamic_spawn_distance = 0.0
        ):
            return True
        dx = candidate_xy[0] - self.spawn_local_offset[0]
        dy = candidate_xy[1] - self.spawn_local_offset[1]
        distance = float(np.hypot(dx, dy))
        return distance = self.min_dynamic_spawn_distance

    def _sample_dynamic_obstacle_position(self) - Tuple[float, float]:

        max_attempts = 20
        length = self.floor_dimensions["length"]
        width = self.floor_dimensions["width"]

        for _ in range(max_attempts):
            x = np.random.uniform(2.0, length - 2.0)
            y = np.random.uniform(2.0, width - 2.0)
            if self._is_safe_spawn_distance((x, y)):
                return x, y

        self.logger.warning(
            "Could not place dynamic obstacle = .1fm from spawn after d attempts; "
            "falling back to edge placement",
            self.min_dynamic_spawn_distance,
            max_attempts,
        )
        if self.spawn_local_offset:
            fallback_x = min(
                max(self.spawn_local_offset[0] + self.min_dynamic_spawn_distance, 2.0),
                length - 2.0,
            )
            fallback_y = min(
                max(self.spawn_local_offset[1] + self.min_dynamic_spawn_distance, 2.0),
                width - 2.0,
            )
        else:
            fallback_x = length / 2.0
            fallback_y = width / 2.0
        return fallback_x, fallback_y
    def _create_building_spec(self) - BuildingSpec:

        floors = []

        for floor_num in range(1, self.num_floors + 1):
            if floor_num == 1:
                floor_type = FloorType.COMMERCIAL
                obstacle_density = 0.1
            elif floor_num = 3:
                floor_type = FloorType.OFFICE
                obstacle_density = 0.15
            else:
                floor_type = FloorType.RESIDENTIAL
                obstacle_density = 0.2

            landing_targets = [f"Landing_{floor_num}{i:02d}" for i in range(1, 7)]

            accessibility = max(0.3, 1.0 - (floor_num - 1)  0.15)

            floor_spec = FloorSpec(
                floor_number=floor_num,
                floor_type=floor_type,
                dimensions=(
                    self.floor_dimensions["length"],
                    self.floor_dimensions["width"],
                    self.floor_dimensions["height"],
                ),
                landing_targets=landing_targets,
                obstacle_density=obstacle_density,
                has_dynamic_obstacles=self.enable_dynamic_obstacles and floor_num = 3,
                accessibility=accessibility,
            )

            floors.append(floor_spec)

        total_height = self.num_floors  self.floor_dimensions["height"]

        building_spec = BuildingSpec(
            floors=floors,
            total_height=total_height,
            spawn_location=tuple(self.spawn_location),
            vertical_connectors=["stair_core_nw", "elevator_core_ne"],
            emergency_exits=[
                (0.0, 0.0, 0.0),
                (20.0, 40.0, 0.0),
                (10.0, 0.0, 9.0),
                (10.0, 40.0, 15.0),
            ],
        )

        return building_spec

    def build_world(self) - bool:

        self.logger.info("Building 5-floor world environment...")

        try:
            self._generate_static_obstacles()

            if self.enable_dynamic_obstacles:
                self._initialize_dynamic_obstacles()

            self._place_landing_targets()

            self._setup_vertical_connectors()

            self.logger.info("World building completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"World building failed: {e}")
            return False

    def _generate_static_obstacles(self):

        self.static_obstacles.clear()

        for floor_spec in self.building_spec.floors:
            floor_obstacles = self._generate_floor_obstacles(floor_spec)
            self.static_obstacles.extend(floor_obstacles)

        self.logger.info(f"Generated {len(self.static_obstacles)} static obstacles")

    def _generate_floor_obstacles(
        self, floor_spec: FloorSpec
    ) - List[Tuple[float, float, float]]:

        obstacles = []
        floor_z = floor_spec.floor_number  floor_spec.dimensions[2]

        floor_area = floor_spec.dimensions[0]  floor_spec.dimensions[1]
        num_obstacles = int(floor_area  floor_spec.obstacle_density)

        wall_obstacles = self._generate_wall_obstacles(floor_spec, floor_z)
        obstacles.extend(wall_obstacles)

        interior_obstacles = self._generate_interior_obstacles(
            floor_spec, floor_z, num_obstacles
        )
        obstacles.extend(interior_obstacles)

        if floor_spec.floor_type == FloorType.OFFICE:
            office_obstacles = self._generate_office_obstacles(floor_spec, floor_z)
            obstacles.extend(office_obstacles)
        elif floor_spec.floor_type == FloorType.RESIDENTIAL:
            residential_obstacles = self._generate_residential_obstacles(
                floor_spec, floor_z
            )
            obstacles.extend(residential_obstacles)

        return obstacles

    def _generate_wall_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:

        obstacles = []
        length, width, height = floor_spec.dimensions

        wall_thickness = 0.2

        for x in np.arange(0, length, 1.0):
            obstacles.append((x, 0.0, floor_z))
            obstacles.append((x, width, floor_z))

        for y in np.arange(0, width, 1.0):
            obstacles.append((0.0, y, floor_z))
            obstacles.append((length, y, floor_z))

        if floor_spec.floor_number  1:
            for x_frac in [0.25, 0.75]:
                for y_frac in [0.25, 0.75]:
                    col_x = length  x_frac
                    col_y = width  y_frac
                    obstacles.append((col_x, col_y, floor_z))

        return obstacles

    def _generate_interior_obstacles(
        self, floor_spec: FloorSpec, floor_z: float, num_obstacles: int
    ) - List[Tuple[float, float, float]]:

        obstacles = []
        length, width, height = floor_spec.dimensions

        clear_zones = []

        for target_name in floor_spec.landing_targets:
            target_idx = int(target_name.split("_")[1][-2:]) - 1
            target_x = (target_idx  3 + 1)  length / 4
            target_y = (target_idx
            clear_zones.append((target_x, target_y, 2.0))

        attempts = 0
        max_attempts = num_obstacles  5

        while len(obstacles)  num_obstacles and attempts  max_attempts:
            x = np.random.uniform(1.0, length - 1.0)
            y = np.random.uniform(1.0, width - 1.0)

            valid_position = True
            for clear_x, clear_y, clear_radius in clear_zones:
                distance = np.sqrt((x - clear_x)  2 + (y - clear_y)  2)
                if distance  clear_radius:
                    valid_position = False
                    break

            if valid_position:
                obstacles.append((x, y, floor_z))

            attempts += 1

        return obstacles

    def _generate_office_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:

        obstacles = []
        length, width, height = floor_spec.dimensions

        cubicle_size = 3.0

        for x in np.arange(2.0, length - 2.0, cubicle_size):
            for y in np.arange(2.0, width - 2.0, cubicle_size):
                obstacles.extend(
                    [
                        (x, y, floor_z),
                        (x + cubicle_size, y, floor_z),
                        (x, y + cubicle_size, floor_z),
                        (x + cubicle_size, y + cubicle_size, floor_z),
                    ]
                )

        return obstacles

    def _generate_residential_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) - List[Tuple[float, float, float]]:

        obstacles = []
        length, width, height = floor_spec.dimensions

        room_size = 4.0

        for x in np.arange(1.0, length - 1.0, room_size):
            for y in np.arange(1.0, width - 1.0, room_size):
                furniture_x = x + np.random.uniform(0.5, room_size - 0.5)
                furniture_y = y + np.random.uniform(0.5, room_size - 0.5)
                obstacles.append((furniture_x, furniture_y, floor_z))

        return obstacles

    def _initialize_dynamic_obstacles(self):

        self.dynamic_obstacles.clear()

        for agent_id in range(self.num_human_agents):
            floor = np.random.randint(
                1, min(4, self.num_floors + 1)
            )
            floor_z = floor  self.floor_dimensions["height"]

            x, y = self._sample_dynamic_obstacle_position()

            speed = np.random.uniform(
                self.human_speed_range[0], self.human_speed_range[1]
            )

            direction = np.random.uniform(0, 2  np.pi)

            agent = {
                "id": f"Human_{agent_id + 1}",
                "position": [x, y, floor_z],
                "velocity": [speed  np.cos(direction), speed  np.sin(direction), 0.0],
                "floor": floor,
                "last_update": time.time(),
                "behavior": "corridor_walk",
            }

            self.dynamic_obstacles.append(agent)

        self.logger.info(f"Initialized {len(self.dynamic_obstacles)} dynamic obstacles")

    def _place_landing_targets(self):

        total_targets = 0
        for floor_spec in self.building_spec.floors:
            total_targets += len(floor_spec.landing_targets)

        self.logger.info(
            f"Placed {total_targets} landing targets across {self.num_floors} floors"
        )

    def _setup_vertical_connectors(self):

        connectors = self.building_spec.vertical_connectors

        for connector in connectors:
            if "stair" in connector:
                self._place_staircase(connector)
            elif "elevator" in connector:
                self._place_elevator(connector)

        self.logger.info(f"Setup {len(connectors)} vertical connectors")

    def _place_staircase(self, stair_name: str):

        stair_x, stair_y = 2.0, 2.0

        for floor in range(self.num_floors):
            floor_z = floor  self.floor_dimensions["height"]
            pass

    def _place_elevator(self, elevator_name: str):

        elevator_x = self.floor_dimensions["length"] - 2.0
        elevator_y = 2.0

        for floor in range(self.num_floors):
            floor_z = floor  self.floor_dimensions["height"]
            pass

    def update_world_state(self):

        current_time = time.time()

        if current_time - self.last_update_time  1.0 / self.update_frequency:
            return

        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        if self.enable_dynamic_obstacles:
            self._update_dynamic_obstacles(dt)

    def _update_dynamic_obstacles(self, dt: float):

        for agent in self.dynamic_obstacles:
            pos = agent["position"]
            vel = agent["velocity"]

            new_pos = [
                pos[0] + vel[0]  dt,
                pos[1] + vel[1]  dt,
                pos[2],
            ]

            if (
                new_pos[0]  1.0
                or new_pos[0]  self.floor_dimensions["length"] - 1.0
                or new_pos[1]  1.0
                or new_pos[1]  self.floor_dimensions["width"] - 1.0
            ):
                if (
                    new_pos[0]  1.0
                    or new_pos[0]  self.floor_dimensions["length"] - 1.0
                ):
                    vel[0] = -vel[0]
                if (
                    new_pos[1]  1.0
                    or new_pos[1]  self.floor_dimensions["width"] - 1.0
                ):
                    vel[1] = -vel[1]

                new_pos[0] = np.clip(
                    new_pos[0], 1.0, self.floor_dimensions["length"] - 1.0
                )
                new_pos[1] = np.clip(
                    new_pos[1], 1.0, self.floor_dimensions["width"] - 1.0
                )

            agent["position"] = new_pos
            agent["velocity"] = vel
            agent["last_update"] = time.time()

    def get_obstacles(
        self,
    ) - Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:

        static = self.static_obstacles.copy()

        dynamic = [tuple(agent["position"]) for agent in self.dynamic_obstacles]

        return static, dynamic

    def get_world_bounds(self) - Dict[str, float]:

        return {
            "x_min": 0.0,
            "x_max": self.floor_dimensions["length"],
            "y_min": 0.0,
            "y_max": self.floor_dimensions["width"],
            "z_min": 0.0,
            "z_max": self.building_spec.total_height,
        }

    def get_building_info(self) - Dict[str, Any]:

        return {
            "num_floors": self.num_floors,
            "floor_dimensions": self.floor_dimensions,
            "total_height": self.building_spec.total_height,
            "spawn_location": self.building_spec.spawn_location,
            "total_cells": self.total_cells,
            "cell_size": self.cell_size,
            "static_obstacles": len(self.static_obstacles),
            "dynamic_obstacles": len(self.dynamic_obstacles),
            "vertical_connectors": self.building_spec.vertical_connectors,
            "floor_specs": [
                {
                    "floor": spec.floor_number,
                    "type": spec.floor_type.value,
                    "targets": len(spec.landing_targets),
                    "obstacle_density": spec.obstacle_density,
                    "dynamic_obstacles": spec.has_dynamic_obstacles,
                    "accessibility": spec.accessibility,
                }
                for spec in self.building_spec.floors
            ],
        }

    def reset_world(self):

        if self.enable_dynamic_obstacles:
            self._initialize_dynamic_obstacles()

        self.last_update_time = time.time()
        self.logger.debug("World state reset")
