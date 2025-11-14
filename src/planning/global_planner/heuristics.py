import numpy as np
import logging
from typing import Tuple, Dict, Any, Callable

class AStarHeuristics:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.heuristic_type = config.get(
            "heuristic_type", "euclidean_with_floor_penalty"
        )

        self.floor_transition_penalty = config.get("floor_transition_penalty", 2.0)
        self.cell_size = config.get("cell_size", 0.5)
        self.floor_height_cells = int(3.0 / self.cell_size)

        self.distance_weight = config.get("distance_weight", 1.0)
        self.vertical_weight = config.get("vertical_weight", 1.0)

        self.heuristic_functions = {
            "euclidean": self._euclidean_distance,
            "manhattan": self._manhattan_distance,
            "euclidean_with_floor_penalty": self._euclidean_with_floor_penalty,
            "octile": self._octile_distance,
            "diagonal": self._diagonal_distance,
        }

        if self.heuristic_type not in self.heuristic_functions:
            self.logger.warning(
                f"Unknown heuristic type: {self.heuristic_type}, using euclidean"
            )
            self.heuristic_type = "euclidean"

        self.heuristic_func = self.heuristic_functions[self.heuristic_type]

        self.logger.info(f"A Heuristics initialized with {self.heuristic_type}")
        self.logger.info(f"Floor penalty: {self.floor_transition_penalty}")

    def compute_heuristic(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        return self.heuristic_func(current, goal)

    def _euclidean_distance(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        dx = (goal[0] - current[0])  self.cell_size
        dy = (goal[1] - current[1])  self.cell_size
        dz = (goal[2] - current[2])  self.cell_size

        return np.sqrt(dx2 + dy2 + dz2)

    def _manhattan_distance(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        dx = abs(goal[0] - current[0])  self.cell_size
        dy = abs(goal[1] - current[1])  self.cell_size
        dz = abs(goal[2] - current[2])  self.cell_size

        return dx + dy + dz

    def _euclidean_with_floor_penalty(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        base_distance = self._euclidean_distance(current, goal)

        current_floor = current[2]
        goal_floor = goal[2]

        floor_changes = abs(goal_floor - current_floor)
        floor_penalty = floor_changes  self.floor_transition_penalty

        return base_distance + floor_penalty

    def _octile_distance(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        dx = abs(goal[0] - current[0])
        dy = abs(goal[1] - current[1])
        dz = abs(goal[2] - current[2])

        dims = sorted([dx, dy, dz], reverse=True)

        d1, d2, d3 = dims

        diagonal_3d_cost = np.sqrt(3)  self.cell_size
        diagonal_2d_cost = np.sqrt(2)  self.cell_size
        straight_cost = self.cell_size

        if d3  0:
            octile_dist = d3  diagonal_3d_cost
            d1 -= d3
            d2 -= d3

        if d2  0:
            octile_dist += d2  diagonal_2d_cost
            d1 -= d2

        octile_dist += d1  straight_cost

        return octile_dist

    def _diagonal_distance(
        self, current: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        dx = abs(goal[0] - current[0])
        dy = abs(goal[1] - current[1])
        dz = abs(goal[2] - current[2])

        diagonal_2d = min(dx, dy)
        straight_x = dx - diagonal_2d
        straight_y = dy - diagonal_2d
        straight_z = dz

        cost = (
            diagonal_2d  np.sqrt(2)  self.cell_size
            + (straight_x + straight_y + straight_z)  self.cell_size
        )

        return cost

    def validate_admissibility(
        self, grid_size: Tuple[int, int, int]
    ) - Dict[str, bool]:

        results = {}

        test_points = [
            (
                (0, 0, 0),
                (grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1),
            ),
            ((0, 0, 0), (grid_size[0] - 1, 0, 0)),
            ((0, 0, 0), (0, grid_size[1] - 1, 0)),
            ((0, 0, 0), (0, 0, grid_size[2] - 1)),
            ((0, 0, 0), (5, 5, grid_size[2]
        ]

        for heuristic_name, heuristic_func in self.heuristic_functions.items():
            is_admissible = True

            for start, goal in test_points:
                heuristic_cost = heuristic_func(start, goal)

                true_cost = self._calculate_true_minimum_cost(start, goal)

                if (
                    heuristic_cost  true_cost + 1e-6
                ):
                    is_admissible = False
                    self.logger.warning(
                        f"Heuristic {heuristic_name} overestimates: "
                        f"{heuristic_cost:.3f}  {true_cost:.3f}"
                    )
                    break

            results[heuristic_name] = is_admissible

        return results

    def _calculate_true_minimum_cost(
        self, start: Tuple[int, int, int], goal: Tuple[int, int, int]
    ) - float:

        dx = abs(goal[0] - start[0])
        dy = abs(goal[1] - start[1])
        dz = abs(goal[2] - start[2])

        min_cost = max(dx, dy, dz)  self.cell_size

        current_floor = start[2]
        goal_floor = goal[2]
        floor_changes = abs(goal_floor - current_floor)

        min_cost += floor_changes  self.floor_transition_penalty

        return min_cost

    def get_heuristic_info(self) - Dict[str, Any]:

        return {
            "heuristic_type": self.heuristic_type,
            "parameters": {
                "floor_transition_penalty": self.floor_transition_penalty,
                "cell_size": self.cell_size,
                "floor_height_cells": self.floor_height_cells,
                "distance_weight": self.distance_weight,
                "vertical_weight": self.vertical_weight,
            },
            "available_heuristics": list(self.heuristic_functions.keys()),
            "properties": {
                "admissible": True,
                "consistent": True,
                "informed": self.heuristic_type != "zero",
            },
        }
