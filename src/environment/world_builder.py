"""
World Builder
Constructs and manages 5-floor building environment in AirSim.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class FloorType(Enum):
    """Floor types for different layouts."""

    RESIDENTIAL = "residential"
    OFFICE = "office"
    COMMERCIAL = "commercial"
    MIXED_USE = "mixed_use"


@dataclass
class FloorSpec:
    """Specification for a single floor."""

    floor_number: int
    floor_type: FloorType
    dimensions: Tuple[float, float, float]  # length, width, height
    landing_targets: List[str]
    obstacle_density: float
    has_dynamic_obstacles: bool
    accessibility: float  # 0.0 to 1.0


@dataclass
class BuildingSpec:
    """Complete building specification."""

    floors: List[FloorSpec]
    total_height: float
    spawn_location: Tuple[float, float, float]
    vertical_connectors: List[str]  # stairs, elevators
    emergency_exits: List[Tuple[float, float, float]]


class WorldBuilder:
    """
    Builds and manages 5-floor building environment.
    Handles static and dynamic obstacles, target placement, and world state.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize world builder."""
        self.config = config or {}  # ← FIX: Handle None
        self.logger = logging.getLogger(__name__)

        # Building specifications (from report)
        self.num_floors = config.get("num_floors", 5)
        self.floor_dimensions = config.get(
            "floor_dimensions",
            {
                "length": 20.0,  # meters
                "width": 40.0,  # meters
                "height": 3.0,  # meters
            },
        )

        # World parameters
        self.cell_size = config.get("cell_size", 0.5)  # meters
        self.total_cells = config.get("total_cells", 4000)  # As per report
        self.spawn_location = config.get("spawn_location", [6000.0, -3000.0, 300.0])

        # Dynamic obstacles (human agents)
        self.enable_dynamic_obstacles = config.get("dynamic_obstacles", True)
        self.num_human_agents = config.get("num_human_agents", 12)
        self.human_speed_range = config.get("human_speed_range", [0.8, 1.5])  # m/s

        # Build complete building specification
        self.building_spec = self._create_building_spec()

        # World state tracking
        self.static_obstacles: List[Tuple[float, float, float]] = []
        self.dynamic_obstacles: List[Dict[str, Any]] = []
        self.last_update_time = 0.0
        self.update_frequency = config.get("world_update_frequency", 10.0)  # Hz

        # AirSim integration
        self.airsim_bridge = None

        self.logger.info("World Builder initialized")
        self.logger.info(f"Building: {self.num_floors} floors, {self.floor_dimensions}")
        self.logger.info(
            f"Dynamic obstacles: {self.enable_dynamic_obstacles} ({self.num_human_agents} agents)"
        )

    def set_airsim_bridge(self, bridge):
        """Set AirSim bridge reference."""
        self.airsim_bridge = bridge
        self.logger.info("AirSim bridge connected to world builder")

    def _create_building_spec(self) -> BuildingSpec:
        """Create complete building specification."""
        floors = []

        for floor_num in range(1, self.num_floors + 1):
            # Determine floor type based on level
            if floor_num == 1:
                floor_type = FloorType.COMMERCIAL
                obstacle_density = 0.1
            elif floor_num <= 3:
                floor_type = FloorType.OFFICE
                obstacle_density = 0.15
            else:
                floor_type = FloorType.RESIDENTIAL
                obstacle_density = 0.2

            # Generate landing targets for this floor
            landing_targets = [f"Landing_{floor_num}{i:02d}" for i in range(1, 7)]

            # Floor accessibility (higher floors are harder)
            accessibility = max(0.3, 1.0 - (floor_num - 1) * 0.15)

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
                has_dynamic_obstacles=self.enable_dynamic_obstacles and floor_num <= 3,
                accessibility=accessibility,
            )

            floors.append(floor_spec)

        # Building-level specifications
        total_height = self.num_floors * self.floor_dimensions["height"]

        building_spec = BuildingSpec(
            floors=floors,
            total_height=total_height,
            spawn_location=tuple(self.spawn_location),
            vertical_connectors=["stair_core_nw", "elevator_core_ne"],
            emergency_exits=[
                (0.0, 0.0, 0.0),  # Ground level exit
                (20.0, 40.0, 0.0),  # Opposite corner
                (10.0, 0.0, 9.0),  # Mid-level emergency
                (10.0, 40.0, 15.0),  # Top level emergency
            ],
        )

        return building_spec

    def build_world(self) -> bool:
        """
        Build complete world in AirSim environment.

        Returns:
            Success status
        """
        self.logger.info("Building 5-floor world environment...")

        try:
            # Generate static obstacles
            self._generate_static_obstacles()

            # Initialize dynamic obstacles (human agents)
            if self.enable_dynamic_obstacles:
                self._initialize_dynamic_obstacles()

            # Place landing targets
            self._place_landing_targets()

            # Set up vertical connectors
            self._setup_vertical_connectors()

            self.logger.info("World building completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"World building failed: {e}")
            return False

    def _generate_static_obstacles(self):
        """Generate static obstacles for all floors."""
        self.static_obstacles.clear()

        for floor_spec in self.building_spec.floors:
            floor_obstacles = self._generate_floor_obstacles(floor_spec)
            self.static_obstacles.extend(floor_obstacles)

        self.logger.info(f"Generated {len(self.static_obstacles)} static obstacles")

    def _generate_floor_obstacles(
        self, floor_spec: FloorSpec
    ) -> List[Tuple[float, float, float]]:
        """
        Generate obstacles for a specific floor.

        Args:
            floor_spec: Floor specification

        Returns:
            List of obstacle positions
        """
        obstacles = []
        floor_z = floor_spec.floor_number * floor_spec.dimensions[2]

        # Number of obstacles based on density
        floor_area = floor_spec.dimensions[0] * floor_spec.dimensions[1]
        num_obstacles = int(floor_area * floor_spec.obstacle_density)

        # Wall obstacles (structural elements)
        wall_obstacles = self._generate_wall_obstacles(floor_spec, floor_z)
        obstacles.extend(wall_obstacles)

        # Interior obstacles (furniture, equipment)
        interior_obstacles = self._generate_interior_obstacles(
            floor_spec, floor_z, num_obstacles
        )
        obstacles.extend(interior_obstacles)

        # Floor-specific obstacles
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
    ) -> List[Tuple[float, float, float]]:
        """Generate wall and structural obstacles."""
        obstacles = []
        length, width, height = floor_spec.dimensions

        # Perimeter walls (sample points along walls)
        wall_thickness = 0.2

        # North and south walls
        for x in np.arange(0, length, 1.0):
            obstacles.append((x, 0.0, floor_z))  # North wall
            obstacles.append((x, width, floor_z))  # South wall

        # East and west walls
        for y in np.arange(0, width, 1.0):
            obstacles.append((0.0, y, floor_z))  # West wall
            obstacles.append((length, y, floor_z))  # East wall

        # Interior structural elements (columns, load-bearing walls)
        if floor_spec.floor_number > 1:  # No columns on ground floor
            # Support columns at quarter points
            for x_frac in [0.25, 0.75]:
                for y_frac in [0.25, 0.75]:
                    col_x = length * x_frac
                    col_y = width * y_frac
                    obstacles.append((col_x, col_y, floor_z))

        return obstacles

    def _generate_interior_obstacles(
        self, floor_spec: FloorSpec, floor_z: float, num_obstacles: int
    ) -> List[Tuple[float, float, float]]:
        """Generate random interior obstacles."""
        obstacles = []
        length, width, height = floor_spec.dimensions

        # Keep clear zones around landing targets and spawn
        clear_zones = []

        # Landing target clear zones
        for target_name in floor_spec.landing_targets:
            # Estimate target position (would be refined with actual target manager)
            target_idx = int(target_name.split("_")[1][-2:]) - 1
            target_x = (target_idx % 3 + 1) * length / 4
            target_y = (target_idx // 3 + 1) * width / 3
            clear_zones.append((target_x, target_y, 2.0))  # 2m clear radius

        # Generate random obstacles avoiding clear zones
        attempts = 0
        max_attempts = num_obstacles * 5

        while len(obstacles) < num_obstacles and attempts < max_attempts:
            # Random position
            x = np.random.uniform(1.0, length - 1.0)
            y = np.random.uniform(1.0, width - 1.0)

            # Check clear zones
            valid_position = True
            for clear_x, clear_y, clear_radius in clear_zones:
                distance = np.sqrt((x - clear_x) ** 2 + (y - clear_y) ** 2)
                if distance < clear_radius:
                    valid_position = False
                    break

            if valid_position:
                obstacles.append((x, y, floor_z))

            attempts += 1

        return obstacles

    def _generate_office_obstacles(
        self, floor_spec: FloorSpec, floor_z: float
    ) -> List[Tuple[float, float, float]]:
        """Generate office-specific obstacles (desks, cubicles)."""
        obstacles = []
        length, width, height = floor_spec.dimensions

        # Office cubicle layout
        cubicle_size = 3.0  # 3x3 meter cubicles

        for x in np.arange(2.0, length - 2.0, cubicle_size):
            for y in np.arange(2.0, width - 2.0, cubicle_size):
                # Cubicle corners
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
    ) -> List[Tuple[float, float, float]]:
        """Generate residential obstacles (furniture, appliances)."""
        obstacles = []
        length, width, height = floor_spec.dimensions

        # Apartment-style layout
        room_size = 4.0  # 4x4 meter rooms

        for x in np.arange(1.0, length - 1.0, room_size):
            for y in np.arange(1.0, width - 1.0, room_size):
                # Furniture placement (random within room)
                furniture_x = x + np.random.uniform(0.5, room_size - 0.5)
                furniture_y = y + np.random.uniform(0.5, room_size - 0.5)
                obstacles.append((furniture_x, furniture_y, floor_z))

        return obstacles

    def _initialize_dynamic_obstacles(self):
        """Initialize dynamic obstacles (human agents)."""
        self.dynamic_obstacles.clear()

        for agent_id in range(self.num_human_agents):
            # Random starting position
            floor = np.random.randint(
                1, min(4, self.num_floors + 1)
            )  # Humans on floors 1-3
            floor_z = floor * self.floor_dimensions["height"]

            # Random position within floor bounds
            x = np.random.uniform(2.0, self.floor_dimensions["length"] - 2.0)
            y = np.random.uniform(2.0, self.floor_dimensions["width"] - 2.0)

            # Random speed within range
            speed = np.random.uniform(
                self.human_speed_range[0], self.human_speed_range[1]
            )

            # Random initial direction
            direction = np.random.uniform(0, 2 * np.pi)

            agent = {
                "id": f"Human_{agent_id + 1}",
                "position": [x, y, floor_z],
                "velocity": [speed * np.cos(direction), speed * np.sin(direction), 0.0],
                "floor": floor,
                "last_update": time.time(),
                "behavior": "corridor_walk",  # Simple behavior model
            }

            self.dynamic_obstacles.append(agent)

        self.logger.info(f"Initialized {len(self.dynamic_obstacles)} dynamic obstacles")

    def _place_landing_targets(self):
        """Place landing targets in AirSim environment."""
        # This would integrate with target_manager to place actual objects
        # For now, just log the placement

        total_targets = 0
        for floor_spec in self.building_spec.floors:
            total_targets += len(floor_spec.landing_targets)

        self.logger.info(
            f"Placed {total_targets} landing targets across {self.num_floors} floors"
        )

    def _setup_vertical_connectors(self):
        """Setup stairs and elevators for multi-floor navigation."""
        connectors = self.building_spec.vertical_connectors

        for connector in connectors:
            # Place connector objects/waypoints
            if "stair" in connector:
                self._place_staircase(connector)
            elif "elevator" in connector:
                self._place_elevator(connector)

        self.logger.info(f"Setup {len(connectors)} vertical connectors")

    def _place_staircase(self, stair_name: str):
        """Place staircase connector."""
        # Staircase in northwest corner
        stair_x, stair_y = 2.0, 2.0

        for floor in range(self.num_floors):
            floor_z = floor * self.floor_dimensions["height"]
            # Stair waypoints would be placed here
            pass

    def _place_elevator(self, elevator_name: str):
        """Place elevator connector."""
        # Elevator in northeast corner
        elevator_x = self.floor_dimensions["length"] - 2.0
        elevator_y = 2.0

        for floor in range(self.num_floors):
            floor_z = floor * self.floor_dimensions["height"]
            # Elevator waypoints would be placed here
            pass

    def update_world_state(self):
        """Update dynamic world state."""
        current_time = time.time()

        # Check if update is needed
        if current_time - self.last_update_time < 1.0 / self.update_frequency:
            return

        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update dynamic obstacles
        if self.enable_dynamic_obstacles:
            self._update_dynamic_obstacles(dt)

    def _update_dynamic_obstacles(self, dt: float):
        """Update positions of dynamic obstacles."""
        for agent in self.dynamic_obstacles:
            # Simple corridor walking behavior
            pos = agent["position"]
            vel = agent["velocity"]

            # Update position
            new_pos = [
                pos[0] + vel[0] * dt,
                pos[1] + vel[1] * dt,
                pos[2],  # Z stays on same floor
            ]

            # Boundary checking and collision avoidance
            if (
                new_pos[0] < 1.0
                or new_pos[0] > self.floor_dimensions["length"] - 1.0
                or new_pos[1] < 1.0
                or new_pos[1] > self.floor_dimensions["width"] - 1.0
            ):
                # Bounce off walls
                if (
                    new_pos[0] < 1.0
                    or new_pos[0] > self.floor_dimensions["length"] - 1.0
                ):
                    vel[0] = -vel[0]
                if (
                    new_pos[1] < 1.0
                    or new_pos[1] > self.floor_dimensions["width"] - 1.0
                ):
                    vel[1] = -vel[1]

                # Clamp position to bounds
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
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """
        Get current obstacle positions.

        Returns:
            (static_obstacles, dynamic_obstacles) tuple
        """
        # Static obstacles
        static = self.static_obstacles.copy()

        # Dynamic obstacles (extract positions)
        dynamic = [tuple(agent["position"]) for agent in self.dynamic_obstacles]

        return static, dynamic

    def get_world_bounds(self) -> Dict[str, float]:
        """
        Get world coordinate bounds.

        Returns:
            Dictionary with min/max bounds
        """
        return {
            "x_min": 0.0,
            "x_max": self.floor_dimensions["length"],
            "y_min": 0.0,
            "y_max": self.floor_dimensions["width"],
            "z_min": 0.0,
            "z_max": self.building_spec.total_height,
        }

    def get_building_info(self) -> Dict[str, Any]:
        """
        Get comprehensive building information.

        Returns:
            Building specification dictionary
        """
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
        """Reset world state for new episode."""
        # Regenerate dynamic obstacles
        if self.enable_dynamic_obstacles:
            self._initialize_dynamic_obstacles()

        self.last_update_time = time.time()
        self.logger.debug("World state reset")
