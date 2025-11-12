"""
Target Manager
Manages Landing_101-506 target system for 5-floor building.
"""

import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TargetDifficulty(Enum):
    """Target difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class LandingTarget:
    """Landing target specification."""

    name: str
    floor: int
    position: Tuple[float, float, float]
    difficulty: TargetDifficulty
    accessible: bool = True
    obstacles_nearby: int = 0


class TargetManager:
    """
    Manages Landing_101-506 target system.
    Implements systematic target naming and placement for 5-floor building.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Building specifications (from report)
        self.num_floors = config.get("num_floors", 5)
        self.targets_per_floor = config.get("targets_per_floor", 6)
        self.floor_dimensions = config.get(
            "floor_dimensions",
            {
                "length": 20.0,  # meters
                "width": 40.0,  # meters
                "height": 3.0,  # meters
            },
        )

        # Target placement parameters
        self.target_height_offset = config.get(
            "target_height_offset", 0.2
        )  # 0.2m above floor
        self.min_target_separation = config.get("min_target_separation", 3.0)  # meters
        self.wall_clearance = config.get("wall_clearance", 1.0)  # meters from walls

        # Generate all targets (Landing_101-506)
        self.targets = self._generate_all_targets()

        # Current episode target
        self.current_target: Optional[LandingTarget] = None
        self.target_history: List[str] = []

        # Curriculum learning support
        self.curriculum_enabled = config.get("curriculum_enabled", True)
        self.curriculum_phase = 1  # Start with single floor

        self.logger.info(
            f"Target Manager initialized: {len(self.targets)} targets across {self.num_floors} floors"
        )
        self.logger.info(
            f"Targets per floor: {self.targets_per_floor}, Curriculum enabled: {self.curriculum_enabled}"
        )

    def _generate_all_targets(self) -> Dict[str, LandingTarget]:
        """
        Generate all Landing_101-506 targets with systematic naming.

        Returns:
            Dictionary mapping target names to LandingTarget objects
        """
        targets = {}

        for floor in range(1, self.num_floors + 1):
            floor_targets = self._generate_floor_targets(floor)
            targets.update(floor_targets)

        self.logger.info(f"Generated {len(targets)} targets: {list(targets.keys())}")
        return targets

    def _generate_floor_targets(self, floor: int) -> Dict[str, LandingTarget]:
        """
        Generate targets for a specific floor.

        Args:
            floor: Floor number (1-5)

        Returns:
            Dictionary of targets for this floor
        """
        floor_targets = {}
        target_positions = self._compute_target_positions(floor)

        for i in range(1, self.targets_per_floor + 1):
            # Systematic naming: Landing_{floor}{target:02d}
            target_name = f"Landing_{floor}{i:02d}"

            # Get position for this target
            if i <= len(target_positions):
                position = target_positions[i - 1]
            else:
                # Fallback: generate random position
                position = self._generate_random_position(floor)

            # Determine difficulty based on floor and position
            difficulty = self._determine_target_difficulty(floor, position)

            # Count nearby obstacles (simplified)
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

    def _compute_target_positions(self, floor: int) -> List[Tuple[float, float, float]]:
        """
        Compute optimal target positions for a floor.

        Args:
            floor: Floor number

        Returns:
            List of target positions
        """
        positions = []
        floor_height = floor * self.floor_dimensions["height"]
        target_z = floor_height + self.target_height_offset

        # Create grid of potential positions with wall clearance
        x_min = self.wall_clearance
        x_max = self.floor_dimensions["length"] - self.wall_clearance
        y_min = self.wall_clearance
        y_max = self.floor_dimensions["width"] - self.wall_clearance

        # For 6 targets per floor, create 2×3 grid
        if self.targets_per_floor == 6:
            x_positions = np.linspace(x_min + 2, x_max - 2, 3)  # 3 columns
            y_positions = np.linspace(y_min + 2, y_max - 2, 2)  # 2 rows

            for y in y_positions:
                for x in x_positions:
                    positions.append((float(x), float(y), target_z))
        else:
            # General case: distribute targets evenly
            num_cols = int(np.ceil(np.sqrt(self.targets_per_floor)))
            num_rows = int(np.ceil(self.targets_per_floor / num_cols))

            x_positions = np.linspace(x_min, x_max, num_cols)
            y_positions = np.linspace(y_min, y_max, num_rows)

            for i, y in enumerate(y_positions):
                for j, x in enumerate(x_positions):
                    if len(positions) < self.targets_per_floor:
                        positions.append((float(x), float(y), target_z))

        return positions

    def _generate_random_position(self, floor: int) -> Tuple[float, float, float]:
        """
        Generate random position on specified floor.

        Args:
            floor: Floor number

        Returns:
            Random position tuple
        """
        floor_height = floor * self.floor_dimensions["height"]
        target_z = floor_height + self.target_height_offset

        # Random position with wall clearance
        x = random.uniform(
            self.wall_clearance, self.floor_dimensions["length"] - self.wall_clearance
        )
        y = random.uniform(
            self.wall_clearance, self.floor_dimensions["width"] - self.wall_clearance
        )

        return (x, y, target_z)

    def _determine_target_difficulty(
        self, floor: int, position: Tuple[float, float, float]
    ) -> TargetDifficulty:
        """
        Determine target difficulty based on floor and position.

        Args:
            floor: Floor number
            position: Target position

        Returns:
            Target difficulty level
        """
        # Floor-based difficulty
        if floor == 1:
            base_difficulty = TargetDifficulty.EASY
        elif floor <= 3:
            base_difficulty = TargetDifficulty.MEDIUM
        else:
            base_difficulty = TargetDifficulty.HARD

        # Position-based adjustments
        x, y, z = position

        # Corners are more difficult
        corner_threshold = 3.0
        is_corner = (
            x < corner_threshold
            or x > self.floor_dimensions["length"] - corner_threshold
        ) and (
            y < corner_threshold
            or y > self.floor_dimensions["width"] - corner_threshold
        )

        if is_corner and base_difficulty == TargetDifficulty.EASY:
            base_difficulty = TargetDifficulty.MEDIUM
        elif is_corner and base_difficulty == TargetDifficulty.MEDIUM:
            base_difficulty = TargetDifficulty.HARD

        return base_difficulty

    def _count_nearby_obstacles(self, position: Tuple[float, float, float]) -> int:
        """
        Count obstacles near target position (simplified).

        Args:
            position: Target position

        Returns:
            Number of nearby obstacles
        """
        # Simplified obstacle counting
        # In real implementation, would query environment/occupancy grid
        x, y, z = position

        # Assume more obstacles near walls and corners
        distance_to_wall = min(
            x,
            y,
            self.floor_dimensions["length"] - x,
            self.floor_dimensions["width"] - y,
        )

        if distance_to_wall < 2.0:
            return random.randint(2, 4)
        elif distance_to_wall < 5.0:
            return random.randint(1, 2)
        else:
            return random.randint(0, 1)

    def select_target(
        self,
        curriculum_phase: Optional[int] = None,
        difficulty_filter: Optional[TargetDifficulty] = None,
    ) -> LandingTarget:
        """
        Select target based on curriculum phase and difficulty.

        Args:
            curriculum_phase: Current curriculum phase (1=single floor, 2=two floors, etc.)
            difficulty_filter: Filter by difficulty level

        Returns:
            Selected landing target
        """
        if curriculum_phase is not None:
            self.curriculum_phase = curriculum_phase

        # Filter targets based on curriculum phase
        available_targets = self._filter_targets_by_curriculum()

        # Apply difficulty filter
        if difficulty_filter is not None:
            available_targets = [
                t for t in available_targets if t.difficulty == difficulty_filter
            ]

        # Select random target from available options
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

    def _filter_targets_by_curriculum(self) -> List[LandingTarget]:
        """
        Filter targets based on curriculum phase.

        Returns:
            List of available targets for current curriculum phase
        """
        all_targets = list(self.targets.values())

        if not self.curriculum_enabled:
            return all_targets

        if self.curriculum_phase == 1:
            # Phase 1: Only floor 1 targets
            return [t for t in all_targets if t.floor == 1]
        elif self.curriculum_phase == 2:
            # Phase 2: Floors 1-2 targets
            return [t for t in all_targets if t.floor <= 2]
        elif self.curriculum_phase >= 3:
            # Phase 3+: All floors
            return all_targets
        else:
            return all_targets

    def get_target(self, target_name: str) -> Optional[LandingTarget]:
        """
        Get specific target by name.

        Args:
            target_name: Target name (e.g., "Landing_301")

        Returns:
            LandingTarget object or None if not found
        """
        return self.targets.get(target_name)

    def get_targets_by_floor(self, floor: int) -> List[LandingTarget]:
        """
        Get all targets on specific floor.

        Args:
            floor: Floor number

        Returns:
            List of targets on specified floor
        """
        return [t for t in self.targets.values() if t.floor == floor]

    def get_targets_by_difficulty(
        self, difficulty: TargetDifficulty
    ) -> List[LandingTarget]:
        """
        Get all targets with specific difficulty.

        Args:
            difficulty: Target difficulty level

        Returns:
            List of targets with specified difficulty
        """
        return [t for t in self.targets.values() if t.difficulty == difficulty]

    def get_current_target(self) -> Optional[LandingTarget]:
        """
        Get currently selected target.

        Returns:
            Current target or None
        """
        return self.current_target

    def is_at_target(
        self, position: Tuple[float, float, float], tolerance: float = 0.5
    ) -> bool:
        """
        Check if position is within target tolerance.

        Args:
            position: Current position
            tolerance: Distance tolerance

        Returns:
            True if within target tolerance
        """
        if self.current_target is None:
            return False

        target_pos = np.array(self.current_target.position)
        current_pos = np.array(position)
        distance = np.linalg.norm(target_pos - current_pos)

        return distance <= tolerance

    def get_distance_to_target(self, position: Tuple[float, float, float]) -> float:
        """
        Get distance to current target.

        Args:
            position: Current position

        Returns:
            Distance to target (or infinity if no target)
        """
        if self.current_target is None:
            return float("inf")

        target_pos = np.array(self.current_target.position)
        current_pos = np.array(position)
        return float(np.linalg.norm(target_pos - current_pos))

    def get_target_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about targets.

        Returns:
            Dictionary with target statistics
        """
        total_targets = len(self.targets)
        targets_by_floor = {}
        targets_by_difficulty = {}

        for target in self.targets.values():
            # Count by floor
            floor_key = f"floor_{target.floor}"
            if floor_key not in targets_by_floor:
                targets_by_floor[floor_key] = 0
            targets_by_floor[floor_key] += 1

            # Count by difficulty
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
        """Reset for new episode."""
        self.current_target = None
        # Keep target history for analysis

    def set_curriculum_phase(self, phase: int):
        """
        Set curriculum learning phase.

        Args:
            phase: Curriculum phase (1, 2, 3, ...)
        """
        if phase != self.curriculum_phase:
            self.curriculum_phase = phase
            self.logger.info(f"Curriculum phase updated to: {phase}")

    def get_target_list(self) -> List[str]:
        """
        Get list of all target names.

        Returns:
            List of target names (Landing_101-506)
        """
        return sorted(list(self.targets.keys()))
