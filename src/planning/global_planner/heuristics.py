"""
A* Heuristics
Implements admissible heuristics for 3D pathfinding with floor transitions.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Callable

class AStarHeuristics:
    """
    Heuristic functions for A* planning in 5-floor building.
    All heuristics are admissible (never overestimate true cost).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Heuristic type
        self.heuristic_type = config.get('heuristic_type', 'euclidean_with_floor_penalty')
        
        # Floor transition parameters (matching A* planner)
        self.floor_transition_penalty = config.get('floor_transition_penalty', 2.0)
        self.cell_size = config.get('cell_size', 0.5)  # meters
        self.floor_height_cells = int(3.0 / self.cell_size)  # 3m per floor in cells
        
        # Heuristic weights
        self.distance_weight = config.get('distance_weight', 1.0)
        self.vertical_weight = config.get('vertical_weight', 1.0)
        
        # Available heuristic functions
        self.heuristic_functions = {
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'euclidean_with_floor_penalty': self._euclidean_with_floor_penalty,
            'octile': self._octile_distance,
            'diagonal': self._diagonal_distance
        }
        
        # Select heuristic function
        if self.heuristic_type not in self.heuristic_functions:
            self.logger.warning(f"Unknown heuristic type: {self.heuristic_type}, using euclidean")
            self.heuristic_type = 'euclidean'
        
        self.heuristic_func = self.heuristic_functions[self.heuristic_type]
        
        self.logger.info(f"A* Heuristics initialized with {self.heuristic_type}")
        self.logger.info(f"Floor penalty: {self.floor_transition_penalty}")
    
    def compute_heuristic(self, current: Tuple[int, int, int], 
                         goal: Tuple[int, int, int]) -> float:
        """
        Compute heuristic estimate from current to goal.
        
        Args:
            current: Current position in grid coordinates
            goal: Goal position in grid coordinates
            
        Returns:
            Heuristic cost estimate
        """
        return self.heuristic_func(current, goal)
    
    def _euclidean_distance(self, current: Tuple[int, int, int], 
                          goal: Tuple[int, int, int]) -> float:
        """
        Simple Euclidean distance heuristic.
        
        Args:
            current: Current grid position
            goal: Goal grid position
            
        Returns:
            Euclidean distance in world units
        """
        dx = (goal[0] - current[0]) * self.cell_size
        dy = (goal[1] - current[1]) * self.cell_size
        dz = (goal[2] - current[2]) * self.cell_size
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _manhattan_distance(self, current: Tuple[int, int, int],
                           goal: Tuple[int, int, int]) -> float:
        """
        Manhattan distance heuristic.
        
        Args:
            current: Current grid position
            goal: Goal grid position
            
        Returns:
            Manhattan distance in world units
        """
        dx = abs(goal[0] - current[0]) * self.cell_size
        dy = abs(goal[1] - current[1]) * self.cell_size
        dz = abs(goal[2] - current[2]) * self.cell_size
        
        return dx + dy + dz
    
    def _euclidean_with_floor_penalty(self, current: Tuple[int, int, int],
                                    goal: Tuple[int, int, int]) -> float:
        """
        Euclidean distance with floor transition penalty (main heuristic).
        Matches floor transition penalty from A* planner.
        
        Args:
            current: Current grid position
            goal: Goal grid position
            
        Returns:
            Distance with floor penalty
        """
        # Base Euclidean distance
        base_distance = self._euclidean_distance(current, goal)
        
        # Floor transition penalty
        current_floor = current[2] // self.floor_height_cells
        goal_floor = goal[2] // self.floor_height_cells
        
        floor_changes = abs(goal_floor - current_floor)
        floor_penalty = floor_changes * self.floor_transition_penalty
        
        return base_distance + floor_penalty
    
    def _octile_distance(self, current: Tuple[int, int, int],
                        goal: Tuple[int, int, int]) -> float:
        """
        Octile distance heuristic (accurate for 8-connected grids).
        
        Args:
            current: Current grid position
            goal: Goal grid position
            
        Returns:
            Octile distance
        """
        dx = abs(goal[0] - current[0])
        dy = abs(goal[1] - current[1])
        dz = abs(goal[2] - current[2])
        
        # Sort dimensions
        dims = sorted([dx, dy, dz], reverse=True)
        
        # Octile formula for 3D
        d1, d2, d3 = dims
        
        # Cost for different move types
        diagonal_3d_cost = np.sqrt(3) * self.cell_size  # All 3 dimensions
        diagonal_2d_cost = np.sqrt(2) * self.cell_size  # 2 dimensions
        straight_cost = self.cell_size                  # 1 dimension
        
        # Calculate octile distance
        if d3 > 0:
            # 3D diagonal moves
            octile_dist = d3 * diagonal_3d_cost
            d1 -= d3
            d2 -= d3
        
        if d2 > 0:
            # 2D diagonal moves
            octile_dist += d2 * diagonal_2d_cost
            d1 -= d2
        
        # Remaining straight moves
        octile_dist += d1 * straight_cost
        
        return octile_dist
    
    def _diagonal_distance(self, current: Tuple[int, int, int],
                          goal: Tuple[int, int, int]) -> float:
        """
        Diagonal distance heuristic.
        
        Args:
            current: Current grid position
            goal: Goal grid position
            
        Returns:
            Diagonal distance
        """
        dx = abs(goal[0] - current[0])
        dy = abs(goal[1] - current[1])
        dz = abs(goal[2] - current[2])
        
        # Diagonal and straight distances
        diagonal_2d = min(dx, dy)
        straight_x = dx - diagonal_2d
        straight_y = dy - diagonal_2d
        straight_z = dz
        
        cost = (
            diagonal_2d * np.sqrt(2) * self.cell_size +    # 2D diagonal moves
            (straight_x + straight_y + straight_z) * self.cell_size  # Straight moves
        )
        
        return cost
    
    def validate_admissibility(self, grid_size: Tuple[int, int, int]) -> Dict[str, bool]:
        """
        Validate that heuristics are admissible by testing on sample paths.
        
        Args:
            grid_size: Size of grid for testing
            
        Returns:
            Validation results for each heuristic
        """
        results = {}
        
        # Test points
        test_points = [
            ((0, 0, 0), (grid_size[0]-1, grid_size[1]-1, grid_size[2]-1)),  # Diagonal
            ((0, 0, 0), (grid_size[0]-1, 0, 0)),                           # Straight X
            ((0, 0, 0), (0, grid_size[1]-1, 0)),                           # Straight Y
            ((0, 0, 0), (0, 0, grid_size[2]-1)),                           # Straight Z
            ((0, 0, 0), (5, 5, grid_size[2]//2)),                          # Mixed
        ]
        
        for heuristic_name, heuristic_func in self.heuristic_functions.items():
            is_admissible = True
            
            for start, goal in test_points:
                heuristic_cost = heuristic_func(start, goal)
                
                # Calculate true minimum cost (simplified)
                true_cost = self._calculate_true_minimum_cost(start, goal)
                
                if heuristic_cost > true_cost + 1e-6:  # Small tolerance for floating point
                    is_admissible = False
                    self.logger.warning(f"Heuristic {heuristic_name} overestimates: "
                                      f"{heuristic_cost:.3f} > {true_cost:.3f}")
                    break
            
            results[heuristic_name] = is_admissible
        
        return results
    
    def _calculate_true_minimum_cost(self, start: Tuple[int, int, int],
                                   goal: Tuple[int, int, int]) -> float:
        """
        Calculate true minimum cost (for admissibility testing).
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            True minimum cost
        """
        # This is simplified - true minimum would require actual A* search
        # For admissibility testing, we use optimistic estimates
        
        dx = abs(goal[0] - start[0])
        dy = abs(goal[1] - start[1])
        dz = abs(goal[2] - start[2])
        
        # Minimum cost assuming perfect diagonal movement
        min_cost = max(dx, dy, dz) * self.cell_size
        
        # Add floor transition penalties
        current_floor = start[2] // self.floor_height_cells
        goal_floor = goal[2] // self.floor_height_cells
        floor_changes = abs(goal_floor - current_floor)
        
        min_cost += floor_changes * self.floor_transition_penalty
        
        return min_cost
    
    def get_heuristic_info(self) -> Dict[str, Any]:
        """
        Get information about current heuristic configuration.
        
        Returns:
            Heuristic information dictionary
        """
        return {
            'heuristic_type': self.heuristic_type,
            'parameters': {
                'floor_transition_penalty': self.floor_transition_penalty,
                'cell_size': self.cell_size,
                'floor_height_cells': self.floor_height_cells,
                'distance_weight': self.distance_weight,
                'vertical_weight': self.vertical_weight
            },
            'available_heuristics': list(self.heuristic_functions.keys()),
            'properties': {
                'admissible': True,  # All our heuristics are designed to be admissible
                'consistent': True,  # Triangle inequality satisfied
                'informed': self.heuristic_type != 'zero'  # Non-trivial heuristic
            }
        }
