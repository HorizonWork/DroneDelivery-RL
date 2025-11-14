import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class TargetDifficulty(Enum):

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

dataclass
class LandingTarget:

    name: str
    floor: int
    position: Tuple[float, float, float]
    difficulty: TargetDifficulty
    accessible: bool = True
    obstacles_nearby: int = 0

class TargetManager:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.num_floors = config.get("num_floors", 5)
        self.targets_per_floor = config.get("targets_per_floor", 6)
        self.floor_dimensions = config.get(
            "floor_dimensions",
            {
                "length": 20.0,
                "width": 40.0,
                "height": 3.0,
            },
        )

        self.target_height_offset = config.get(
            "target_height_offset", 0.2
        )
        self.min_target_separation = config.get("min_target_separation", 3.0)
        self.wall_clearance = config.get("wall_clearance", 1.0)

        self.targets = self._generate_all_targets()

        self.current_target: Optional[LandingTarget] = None
        self.target_history: List[str] = []

        self.curriculum_enabled = config.get("curriculum_enabled", True)
        self.curriculum_phase = 1

        self.logger.info(
            f"Target Manager initialized: {len(self.targets)} targets across {self.num_floors} floors"
        )
        self.logger.info(
            f"Targets per floor: {self.targets_per_floor}, Curriculum enabled: {self.curriculum_enabled}"
        )

    def _generate_all_targets(self) - Dict[str, LandingTarget]:

        targets = {}

        for floor in range(1, self.num_floors + 1):
            floor_targets = self._generate_floor_targets(floor)
            targets.update(floor_targets)

        self.logger.info(f"Generated {len(targets)} targets: {list(targets.keys())}")
        return targets

    def _generate_floor_targets(self, floor: int) - Dict[str, LandingTarget]:

        floor_targets = {}
        target_positions = self._compute_target_positions(floor)

        for i in range(1, self.targets_per_floor + 1):
            target_name = f"Landing_{floor}{i:02d}"

            if i = len(target_positions):
                position = target_positions[i - 1]
            else:
                position = self._generate_random_position(floor)

            difficulty = self._determine_target_difficulty(floor, position)

            obstacles_nearby = self._count_nearby_obstacles(position)

            target = LandingTarget(
                name=target_name,
                floor=floor,
                position=position,
                difficulty=difficulty,
                obstacles_nearby=obstacles_nearby,
            )

            floor_targets[target_name] = target

        return floor_targets

    def _compute_target_positions(self, floor: int) - List[Tuple[float, float, float]]:

        positions = []
        floor_height = floor  self.floor_dimensions["height"]
        target_z = floor_height + self.target_height_offset

        x_min = self.wall_clearance
        x_max = self.floor_dimensions["length"] - self.wall_clearance
        y_min = self.wall_clearance
        y_max = self.floor_dimensions["width"] - self.wall_clearance

        if self.targets_per_floor == 6:
            x_positions = np.linspace(x_min + 2, x_max - 2, 3)
            y_positions = np.linspace(y_min + 2, y_max - 2, 2)

            for y in y_positions:
                for x in x_positions:
                    positions.append((float(x), float(y), target_z))
        else:
            num_cols = int(np.ceil(np.sqrt(self.targets_per_floor)))
            num_rows = int(np.ceil(self.targets_per_floor / num_cols))

            x_positions = np.linspace(x_min, x_max, num_cols)
            y_positions = np.linspace(y_min, y_max, num_rows)

            for i, y in enumerate(y_positions):
                for j, x in enumerate(x_positions):
                    if len(positions)  self.targets_per_floor:
                        positions.append((float(x), float(y), target_z))

        return positions

    def _generate_random_position(self, floor: int) - Tuple[float, float, float]:

        floor_height = floor  self.floor_dimensions["height"]
        target_z = floor_height + self.target_height_offset

        x = random.uniform(
            self.wall_clearance, self.floor_dimensions["length"] - self.wall_clearance
        )
        y = random.uniform(
            self.wall_clearance, self.floor_dimensions["width"] - self.wall_clearance
        )

        return (x, y, target_z)

    def _determine_target_difficulty(
        self, floor: int, position: Tuple[float, float, float]
    ) - TargetDifficulty:

        if floor == 1:
            base_difficulty = TargetDifficulty.EASY
        elif floor = 3:
            base_difficulty = TargetDifficulty.MEDIUM
        else:
            base_difficulty = TargetDifficulty.HARD

        x, y, z = position

        corner_threshold = 3.0
        is_corner = (
            x  corner_threshold
            or x  self.floor_dimensions["length"] - corner_threshold
        ) and (
            y  corner_threshold
            or y  self.floor_dimensions["width"] - corner_threshold
        )

        if is_corner and base_difficulty == TargetDifficulty.EASY:
            base_difficulty = TargetDifficulty.MEDIUM
        elif is_corner and base_difficulty == TargetDifficulty.MEDIUM:
            base_difficulty = TargetDifficulty.HARD

        return base_difficulty

    def _count_nearby_obstacles(self, position: Tuple[float, float, float]) - int:

        x, y, z = position

        distance_to_wall = min(
            x,
            y,
            self.floor_dimensions["length"] - x,
            self.floor_dimensions["width"] - y,
        )

        if distance_to_wall  2.0:
            return random.randint(2, 4)
        elif distance_to_wall  5.0:
            return random.randint(1, 2)
        else:
            return random.randint(0, 1)

    def select_target(
        self,
        curriculum_phase: Optional[int] = None,
        difficulty_filter: Optional[TargetDifficulty] = None,
    ) - LandingTarget:

        if curriculum_phase is not None:
            self.curriculum_phase = curriculum_phase

        available_targets = self._filter_targets_by_curriculum()

        if difficulty_filter is not None:
            available_targets = [
                t for t in available_targets if t.difficulty == difficulty_filter
            ]

        if not available_targets:
            self.logger.warning(
                "No available targets matching criteria, using fallback"
            )
            available_targets = list(self.targets.values())

        selected_target = random.choice(available_targets)
        self.current_target = selected_target
        self.target_history.append(selected_target.name)

        self.logger.info(
            f"Selected target: {selected_target.name} on floor {selected_target.floor} "
            f"(difficulty: {selected_target.difficulty.value})"
        )

        return selected_target

    def _filter_targets_by_curriculum(self) - List[LandingTarget]:

        all_targets = list(self.targets.values())

        if not self.curriculum_enabled:
            return all_targets

        if self.curriculum_phase == 1:
            return [t for t in all_targets if t.floor == 1]
        elif self.curriculum_phase == 2:
            return [t for t in all_targets if t.floor = 2]
        elif self.curriculum_phase = 3:
            return all_targets
        else:
            return all_targets

    def get_target(self, target_name: str) - Optional[LandingTarget]:

        return self.targets.get(target_name)

    def get_targets_by_floor(self, floor: int) - List[LandingTarget]:

        return [t for t in self.targets.values() if t.floor == floor]

    def get_targets_by_difficulty(
        self, difficulty: TargetDifficulty
    ) - List[LandingTarget]:

        return [t for t in self.targets.values() if t.difficulty == difficulty]

    def get_current_target(self) - Optional[LandingTarget]:

        return self.current_target

    def is_at_target(
        self, position: Tuple[float, float, float], tolerance: float = 0.5
    ) - bool:

        if self.current_target is None:
            return False

        target_pos = np.array(self.current_target.position)
        current_pos = np.array(position)
        distance = np.linalg.norm(target_pos - current_pos)

        return distance = tolerance

    def get_distance_to_target(self, position: Tuple[float, float, float]) - float:

        if self.current_target is None:
            return float("inf")

        target_pos = np.array(self.current_target.position)
        current_pos = np.array(position)
        return float(np.linalg.norm(target_pos - current_pos))

    def get_target_statistics(self) - Dict[str, Any]:

        total_targets = len(self.targets)
        targets_by_floor = {}
        targets_by_difficulty = {}

        for target in self.targets.values():
            floor_key = f"floor_{target.floor}"
            if floor_key not in targets_by_floor:
                targets_by_floor[floor_key] = 0
            targets_by_floor[floor_key] += 1

            diff_key = target.difficulty.value
            if diff_key not in targets_by_difficulty:
                targets_by_difficulty[diff_key] = 0
            targets_by_difficulty[diff_key] += 1

        return {
            "total_targets": total_targets,
            "targets_by_floor": targets_by_floor,
            "targets_by_difficulty": targets_by_difficulty,
            "current_target": self.current_target.name if self.current_target else None,
            "curriculum_phase": self.curriculum_phase,
            "target_history_length": len(self.target_history),
            "recent_targets": self.target_history[-10:] if self.target_history else [],
        }

    def reset_episode(self):

        self.current_target = None

    def set_curriculum_phase(self, phase: int):

        if phase != self.curriculum_phase:
            self.curriculum_phase = phase
            self.logger.info(f"Curriculum phase updated to: {phase}")

    def get_target_list(self) - List[str]:

        return sorted(list(self.targets.keys()))
