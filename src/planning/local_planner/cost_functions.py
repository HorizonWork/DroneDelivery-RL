"""
S-RRT Cost Functions
Implements Equation (3): C = ℓ + λc(1/dmin)² + λκ·κ²
Safety-oriented cost with clearance and curvature penalties.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class CostParameters:
    """Cost function parameters for Equation (3)."""
    lambda_c: float = 1.0      # Clearance penalty weight λc
    lambda_kappa: float = 2.0  # Curvature penalty weight λκ
    min_clearance: float = 0.5 # Minimum safe clearance (meters)
    max_curvature: float = 1.0 # Maximum acceptable curvature

class SRRTCostFunction:
    """
    S-RRT cost function implementation.
    Exact implementation of Equation (3) from report.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cost parameters (exact match with Equation 3)
        self.params = CostParameters(
            lambda_c=config.get('lambda_c', 1.0),
            lambda_kappa=config.get('lambda_kappa', 2.0),
            min_clearance=config.get('min_clearance', 0.5),
            max_curvature=config.get('max_curvature', 1.0)
        )
        
        # Obstacle radius for clearance calculation
        self.obstacle_radius = config.get('obstacle_radius', 0.3)        # meters
        
        # Curvature calculation parameters
        self.curvature_window = config.get('curvature_window', 3)        # points for curvature
        
        self.logger.info("S-RRT Cost Function initialized")
        self.logger.info(f"Equation (3): C = ℓ + {self.params.lambda_c}·(1/dmin)² + {self.params.lambda_kappa}·κ²")
        self.logger.info(f"Min clearance: {self.params.min_clearance}m")
    
    def calculate_edge_cost(self, parent_node, child_node, 
                          obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Calculate cost of edge between parent and child nodes.
        Implements Equation (3): C = ℓ + λc(1/dmin)² + λκ·κ²
        
        Args:
            parent_node: Parent S-RRT node
            child_node: Child S-RRT node  
            obstacles: Current obstacle positions
            
        Returns:
            Edge cost according to Equation (3)
        """
        # 1. Path length cost (ℓ)
        length_cost = np.linalg.norm(child_node.position - parent_node.position)
        
        # 2. Clearance cost (λc(1/dmin)²)
        clearance_cost = self._calculate_clearance_cost(parent_node, child_node, obstacles)
        
        # 3. Curvature cost (λκ·κ²)
        curvature_cost = self._calculate_curvature_cost(parent_node, child_node)
        
        # Total cost (Equation 3)
        total_cost = length_cost + clearance_cost + curvature_cost
        
        return float(total_cost)
    
    def _calculate_clearance_cost(self, parent_node, child_node,
                                obstacles: List[Tuple[float, float, float]]) -> float:
        """
        Calculate clearance penalty: λc(1/dmin)²
        
        Args:
            parent_node: Parent node
            child_node: Child node
            obstacles: Obstacle positions
            
        Returns:
            Clearance penalty cost
        """
        if not obstacles:
            return 0.0
        
        # Sample points along edge for clearance checking
        num_samples = max(3, int(np.linalg.norm(child_node.position - parent_node.position) / 0.2))
        min_clearance = float('inf')
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = parent_node.position + t * (child_node.position - parent_node.position)
            
            # Find minimum distance to any obstacle
            for obstacle_pos in obstacles:
                obstacle_array = np.array(obstacle_pos)
                distance = np.linalg.norm(sample_point - obstacle_array) - self.obstacle_radius
                min_clearance = min(min_clearance, max(0.01, distance))  # Avoid division by zero
        
        # Apply clearance penalty (1/dmin)²
        if min_clearance < self.params.min_clearance:
            clearance_penalty = self.params.lambda_c * (1.0 / min_clearance) ** 2
            
            # Cap extremely high penalties
            clearance_penalty = min(clearance_penalty, 1000.0)
        else:
            clearance_penalty = 0.0
        
        return clearance_penalty
    
    def _calculate_curvature_cost(self, parent_node, child_node) -> float:
        """
        Calculate curvature penalty: λκ·κ²
        
        Args:
            parent_node: Parent node
            child_node: Child node
            
        Returns:
            Curvature penalty cost
        """
        # Need grandparent for curvature calculation
        if parent_node.parent is None:
            return 0.0  # No curvature at root
        
        grandparent_node = parent_node.parent
        
        # Three consecutive points for curvature
        p1 = grandparent_node.position
        p2 = parent_node.position
        p3 = child_node.position
        
        # Calculate curvature using discrete approximation
        # κ ≈ ||(p3 - 2p2 + p1)|| / ||p3 - p1||²
        
        v1 = p2 - p1  # First segment
        v2 = p3 - p2  # Second segment
        
        # Second derivative approximation
        second_derivative = v2 - v1
        
        # Path length for normalization
        total_length = np.linalg.norm(v1) + np.linalg.norm(v2)
        
        if total_length < 1e-6:
            return 0.0  # Avoid division by zero
        
        # Curvature magnitude
        curvature = np.linalg.norm(second_derivative) / (total_length ** 2)
        
        # Apply curvature penalty
        curvature_penalty = self.params.lambda_kappa * (curvature ** 2)
        
        return curvature_penalty
    
    def evaluate_path_cost(self, path: List[np.ndarray],
                          obstacles: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Evaluate total cost of complete path.
        
        Args:
            path: Complete path as list of positions
            obstacles: Obstacle positions
            
        Returns:
            Cost breakdown dictionary
        """
        if len(path) < 2:
            return {'total_cost': 0.0, 'length_cost': 0.0, 'clearance_cost': 0.0, 'curvature_cost': 0.0}
        
        total_length = 0.0
        total_clearance_cost = 0.0
        total_curvature_cost = 0.0
        
        # Evaluate each edge
        for i in range(len(path) - 1):
            # Create temporary nodes for cost calculation
            parent_pos = path[i]
            child_pos = path[i + 1]
            
            # Length cost
            edge_length = np.linalg.norm(child_pos - parent_pos)
            total_length += edge_length
            
            # Clearance cost (simplified calculation)
            edge_clearance_cost = self._calculate_edge_clearance_cost(parent_pos, child_pos, obstacles)
            total_clearance_cost += edge_clearance_cost
            
            # Curvature cost (for edges with grandparent)
            if i > 0:
                grandparent_pos = path[i - 1]
                edge_curvature_cost = self._calculate_edge_curvature_cost(grandparent_pos, parent_pos, child_pos)
                total_curvature_cost += edge_curvature_cost
        
        total_cost = total_length + total_clearance_cost + total_curvature_cost
        
        return {
            'total_cost': float(total_cost),
            'length_cost': float(total_length),
            'clearance_cost': float(total_clearance_cost),
            'curvature_cost': float(total_curvature_cost),
            'length_fraction': total_length / max(total_cost, 1e-6),
            'clearance_fraction': total_clearance_cost / max(total_cost, 1e-6),
            'curvature_fraction': total_curvature_cost / max(total_cost, 1e-6)
        }
    
    def _calculate_edge_clearance_cost(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                     obstacles: List[Tuple[float, float, float]]) -> float:
        """Calculate clearance cost for a single edge."""
        if not obstacles:
            return 0.0
        
        # Sample points along edge
        num_samples = max(3, int(np.linalg.norm(end_pos - start_pos) / 0.2))
        min_clearance = float('inf')
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = start_pos + t * (end_pos - start_pos)
            
            for obstacle_pos in obstacles:
                obstacle_array = np.array(obstacle_pos)
                distance = np.linalg.norm(sample_point - obstacle_array) - self.obstacle_radius
                min_clearance = min(min_clearance, max(0.01, distance))
        
        # Apply clearance penalty
        if min_clearance < self.params.min_clearance:
            return self.params.lambda_c * (1.0 / min_clearance) ** 2
        
        return 0.0
    
    def _calculate_edge_curvature_cost(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate curvature cost for three consecutive points."""
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Second derivative
        second_derivative = v2 - v1
        total_length = np.linalg.norm(v1) + np.linalg.norm(v2)
        
        if total_length < 1e-6:
            return 0.0
        
        curvature = np.linalg.norm(second_derivative) / (total_length ** 2)
        return self.params.lambda_kappa * (curvature ** 2)
    
    def validate_cost_parameters(self) -> Dict[str, bool]:
        """
        Validate cost function parameters.
        
        Returns:
            Validation results
        """
        validation = {
            'lambda_c_positive': self.params.lambda_c > 0,
            'lambda_kappa_positive': self.params.lambda_kappa > 0,
            'min_clearance_reasonable': 0.1 <= self.params.min_clearance <= 2.0,
            'obstacle_radius_reasonable': 0.05 <= self.obstacle_radius <= 1.0,
            'parameters_balanced': self.params.lambda_c / self.params.lambda_kappa < 10.0
        }
        
        validation['overall_valid'] = all(validation.values())
        
        if not validation['overall_valid']:
            self.logger.warning("Cost function parameters may be suboptimal")
        
        return validation
