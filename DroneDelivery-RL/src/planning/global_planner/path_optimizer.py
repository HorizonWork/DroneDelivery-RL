"""
Path Optimizer
Optimizes A* paths for smoothness and energy efficiency.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import time

class PathOptimizer:
    """
    Path optimization for A* global plans.
    Smooths paths and optimizes for energy efficiency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.smoothing_weight = config.get('smoothing_weight', 1.0)
        self.obstacle_weight = config.get('obstacle_weight', 10.0)
        self.energy_weight = config.get('energy_weight', 0.5)
        
        # Smoothing methods
        self.method = config.get('method', 'spline')  # 'spline', 'gradient_descent', 'both'
        self.spline_degree = config.get('spline_degree', 3)
        self.spline_smoothness = config.get('spline_smoothness', 0.1)
        
        # Optimization constraints
        self.max_iterations = config.get('max_iterations', 100)
        self.convergence_tolerance = config.get('convergence_tolerance', 1e-6)
        self.max_optimization_time = config.get('max_optimization_time', 2.0)  # seconds
        
        # Safety constraints
        self.min_obstacle_distance = config.get('min_obstacle_distance', 0.5)  # meters
        self.max_velocity = config.get('max_velocity', 5.0)                    # m/s
        self.max_acceleration = config.get('max_acceleration', 2.0)             # m/s²
        
        self.logger.info("Path Optimizer initialized")
        self.logger.info(f"Method: {self.method}, Spline degree: {self.spline_degree}")
        self.logger.info(f"Weights - Smoothing: {self.smoothing_weight}, "
                        f"Obstacle: {self.obstacle_weight}, Energy: {self.energy_weight}")
    
    def optimize_path(self, path: List[Tuple[float, float, float]], 
                     occupancy_grid: Optional[Any] = None) -> List[Tuple[float, float, float]]:
        """
        Optimize path for smoothness and energy efficiency.
        
        Args:
            path: Original path from A* planner
            occupancy_grid: Occupancy grid for collision checking
            
        Returns:
            Optimized path
        """
        if len(path) < 3:
            return path  # Can't optimize very short paths
        
        optimization_start = time.time()
        
        try:
            if self.method == 'spline':
                optimized_path = self._optimize_with_spline(path)
            elif self.method == 'gradient_descent':
                optimized_path = self._optimize_with_gradient_descent(path, occupancy_grid)
            elif self.method == 'both':
                # First spline smoothing, then gradient descent refinement
                spline_path = self._optimize_with_spline(path)
                optimized_path = self._optimize_with_gradient_descent(spline_path, occupancy_grid)
            else:
                optimized_path = path
            
            # Validate optimized path
            if not self._validate_optimized_path(optimized_path, occupancy_grid):
                self.logger.warning("Optimized path validation failed, using original")
                optimized_path = path
            
            optimization_time = time.time() - optimization_start
            
            self.logger.info(f"Path optimized in {optimization_time:.3f}s")
            self.logger.info(f"Waypoints: {len(path)} → {len(optimized_path)}")
            
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"Path optimization failed: {e}")
            return path  # Return original if optimization fails
    
    def _optimize_with_spline(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Optimize path using spline interpolation.
        
        Args:
            path: Original path
            
        Returns:
            Spline-smoothed path
        """
        if len(path) < 4:  # Need at least 4 points for cubic spline
            return path
        
        try:
            # Convert path to numpy array
            path_array = np.array(path)
            
            # Parameterize path by cumulative distance
            distances = np.cumsum(np.concatenate(([0], np.linalg.norm(np.diff(path_array, axis=0), axis=1))))
            
            # Fit spline
            tck, u = splprep([path_array[:, 0], path_array[:, 1], path_array[:, 2]], 
                           u=distances, s=self.spline_smoothness, k=self.spline_degree)
            
            # Generate smoothed path
            num_points = max(len(path), int(distances[-1] / 0.5))  # Point every 0.5m
            u_new = np.linspace(0, distances[-1], num_points)
            
            smoothed_coords = splev(u_new, tck)
            smoothed_path = list(zip(smoothed_coords[0], smoothed_coords[1], smoothed_coords[2]))
            
            return smoothed_path
            
        except Exception as e:
            self.logger.error(f"Spline optimization error: {e}")
            return path
    
    def _optimize_with_gradient_descent(self, path: List[Tuple[float, float, float]],
                                      occupancy_grid: Optional[Any]) -> List[Tuple[float, float, float]]:
        """
        Optimize path using gradient descent.
        
        Args:
            path: Original path
            occupancy_grid: Occupancy grid for collision checking
            
        Returns:
            Gradient-optimized path
        """
        if len(path) < 3:
            return path
        
        try:
            # Convert path to optimization variable
            path_array = np.array(path[1:-1])  # Don't optimize start and end points
            x0 = path_array.flatten()
            
            # Optimization bounds (stay within reasonable area of original path)
            bounds = []
            for point in path_array:
                for coord in point:
                    bounds.append((coord - 2.0, coord + 2.0))  # ±2m from original
            
            # Objective function
            def objective(x):
                reshaped_path = x.reshape(-1, 3)
                full_path = [path[0]] + list(reshaped_path) + [path[-1]]
                return self._calculate_path_cost(full_path, occupancy_grid)
            
            # Optimize
            result = minimize(
                objective, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': self.max_iterations, 'ftol': self.convergence_tolerance}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                optimized_path = [path[0]] + list(optimized_points) + [path[-1]]
                return [(float(p[0]), float(p[1]), float(p[2])) for p in optimized_path]
            else:
                self.logger.warning("Gradient descent optimization failed")
                return path
                
        except Exception as e:
            self.logger.error(f"Gradient descent optimization error: {e}")
            return path
    
    def _calculate_path_cost(self, path: List[Tuple[float, float, float]], 
                           occupancy_grid: Optional[Any]) -> float:
        """
        Calculate total path cost for optimization.
        
        Args:
            path: Path to evaluate
            occupancy_grid: Occupancy grid
            
        Returns:
            Total path cost
        """
        if len(path) < 2:
            return 0.0
        
        path_array = np.array(path)
        total_cost = 0.0
        
        # 1. Smoothness cost (curvature)
        if len(path) > 2:
            # Second derivatives (curvature)
            second_derivatives = np.diff(path_array, n=2, axis=0)
            curvature_cost = np.sum(np.linalg.norm(second_derivatives, axis=1))
            total_cost += self.smoothing_weight * curvature_cost
        
        # 2. Distance cost
        distances = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
        distance_cost = np.sum(distances)
        total_cost += distance_cost
        
        # 3. Obstacle proximity cost
        if occupancy_grid is not None:
            proximity_cost = 0.0
            for point in path:
                grid_pos = occupancy_grid.world_to_grid(point)
                clearance = occupancy_grid.get_min_clearance(grid_pos)
                
                if clearance < self.min_obstacle_distance:
                    penalty = (self.min_obstacle_distance - clearance) ** 2
                    proximity_cost += penalty
            
            total_cost += self.obstacle_weight * proximity_cost
        
        # 4. Energy cost (velocity and acceleration profiles)
        if len(path) > 2:
            # Estimate velocities and accelerations
            dt = 0.1  # Assumed time step
            velocities = np.diff(path_array, axis=0) / dt
            accelerations = np.diff(velocities, axis=0) / dt
            
            # Energy proportional to velocity squared and acceleration squared
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            energy_cost = (np.sum(velocity_magnitudes ** 2) + 
                          np.sum(acceleration_magnitudes ** 2))
            
            total_cost += self.energy_weight * energy_cost
        
        return total_cost
    
    def _validate_optimized_path(self, path: List[Tuple[float, float, float]], 
                                occupancy_grid: Optional[Any]) -> bool:
        """
        Validate that optimized path is collision-free.
        
        Args:
            path: Optimized path
            occupancy_grid: Occupancy grid
            
        Returns:
            True if path is valid
        """
        if not path or occupancy_grid is None:
            return True  # Can't validate without grid
        
        # Check each waypoint
        for point in path:
            grid_pos = occupancy_grid.world_to_grid(point)
            if not occupancy_grid.is_free(grid_pos):
                return False
        
        # Check path segments for intermediate collisions
        for i in range(len(path) - 1):
            if not self._check_path_segment(path[i], path[i+1], occupancy_grid):
                return False
        
        return True
    
    def _check_path_segment(self, start: Tuple[float, float, float],
                          end: Tuple[float, float, float],
                          occupancy_grid: Any) -> bool:
        """
        Check if path segment is collision-free.
        
        Args:
            start: Segment start point
            end: Segment end point
            occupancy_grid: Occupancy grid
            
        Returns:
            True if segment is collision-free
        """
        # Sample points along segment
        num_samples = int(np.linalg.norm(np.array(end) - np.array(start)) / (occupancy_grid.cell_size * 0.5))
        num_samples = max(2, min(num_samples, 50))  # Limit sampling
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                start[2] + t * (end[2] - start[2])
            )
            
            grid_pos = occupancy_grid.world_to_grid(sample_point)
            if not occupancy_grid.is_free(grid_pos):
                return False
        
        return True
    
    def analyze_path_quality(self, path: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Analyze path quality metrics.
        
        Args:
            path: Path to analyze
            
        Returns:
            Quality metrics dictionary
        """
        if len(path) < 2:
            return {'error': 'Path too short for analysis'}
        
        path_array = np.array(path)
        
        # Distance metrics
        segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
        total_length = np.sum(segment_lengths)
        
        # Smoothness metrics
        if len(path) > 2:
            # Calculate turning angles
            vectors = np.diff(path_array, axis=0)
            unit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
            
            turning_angles = []
            for i in range(len(unit_vectors) - 1):
                dot_product = np.clip(np.dot(unit_vectors[i], unit_vectors[i+1]), -1, 1)
                angle = np.arccos(dot_product)
                turning_angles.append(angle)
            
            mean_turning_angle = np.mean(turning_angles) if turning_angles else 0.0
            max_turning_angle = np.max(turning_angles) if turning_angles else 0.0
            
            # Curvature (second derivative magnitude)
            if len(path) > 3:
                second_derivatives = np.diff(path_array, n=2, axis=0)
                curvatures = np.linalg.norm(second_derivatives, axis=1)
                mean_curvature = np.mean(curvatures)
                max_curvature = np.max(curvatures)
            else:
                mean_curvature = max_curvature = 0.0
        else:
            mean_turning_angle = max_turning_angle = 0.0
            mean_curvature = max_curvature = 0.0
        
        # Energy metrics (simplified)
        if len(path) > 1:
            # Estimate energy based on velocity profile
            dt = 0.1  # Assumed time step
            velocities = segment_lengths / dt
            energy_estimate = np.sum(velocities ** 2)  # Proportional to kinetic energy
        else:
            energy_estimate = 0.0
        
        return {
            'total_length': float(total_length),
            'num_waypoints': len(path),
            'mean_segment_length': float(np.mean(segment_lengths)),
            'std_segment_length': float(np.std(segment_lengths)),
            'mean_turning_angle': float(mean_turning_angle),
            'max_turning_angle': float(max_turning_angle),
            'mean_curvature': float(mean_curvature),
            'max_curvature': float(max_curvature),
            'energy_estimate': float(energy_estimate),
            'smoothness_score': float(1.0 / (1.0 + mean_curvature))  # Higher is smoother
        }
    
    def resample_path(self, path: List[Tuple[float, float, float]], 
                     target_spacing: float = 0.5) -> List[Tuple[float, float, float]]:
        """
        Resample path to uniform waypoint spacing.
        
        Args:
            path: Original path
            target_spacing: Desired spacing between waypoints
            
        Returns:
            Resampled path
        """
        if len(path) < 2:
            return path
        
        path_array = np.array(path)
        
        # Calculate cumulative distances
        segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
        cumulative_distances = np.concatenate(([0], np.cumsum(segment_lengths)))
        
        total_length = cumulative_distances[-1]
        
        if total_length <= target_spacing:
            return path  # Path too short to resample
        
        # Create new sample points
        num_samples = int(np.ceil(total_length / target_spacing)) + 1
        sample_distances = np.linspace(0, total_length, num_samples)
        
        # Interpolate positions at sample distances
        resampled_path = []
        
        for sample_dist in sample_distances:
            # Find segment containing this distance
            segment_idx = np.searchsorted(cumulative_distances, sample_dist) - 1
            segment_idx = max(0, min(segment_idx, len(path) - 2))
            
            # Interpolate within segment
            if segment_idx < len(path) - 1:
                segment_start_dist = cumulative_distances[segment_idx]
                segment_length = segment_lengths[segment_idx]
                
                if segment_length > 0:
                    t = (sample_dist - segment_start_dist) / segment_length
                    t = np.clip(t, 0, 1)
                    
                    start_point = path_array[segment_idx]
                    end_point = path_array[segment_idx + 1]
                    
                    interpolated_point = start_point + t * (end_point - start_point)
                    resampled_path.append(tuple(interpolated_point))
                else:
                    resampled_path.append(path[segment_idx])
            else:
                resampled_path.append(path[-1])
        
        return resampled_path
    
    def _optimize_with_gradient_descent(self, path: List[Tuple[float, float, float]],
                                      occupancy_grid: Optional[Any]) -> List[Tuple[float, float, float]]:
        """
        Optimize path using gradient descent.
        
        Args:
            path: Original path  
            occupancy_grid: Occupancy grid for constraints
            
        Returns:
            Optimized path
        """
        # Placeholder implementation - full gradient descent would be more complex
        
        # For now, just apply simple smoothing by averaging neighbors
        if len(path) <= 2:
            return path
        
        path_array = np.array(path)
        optimized_array = path_array.copy()
        
        # Apply multiple iterations of smoothing
        for iteration in range(5):
            for i in range(1, len(path_array) - 1):
                # Simple averaging with neighbors
                neighbor_average = (path_array[i-1] + path_array[i+1]) / 2.0
                
                # Blend with current position
                alpha = 0.3  # Smoothing strength
                optimized_array[i] = (1 - alpha) * path_array[i] + alpha * neighbor_average
            
            path_array = optimized_array.copy()
        
        return [tuple(point) for point in optimized_array]
    
    def _validate_optimized_path(self, path: List[Tuple[float, float, float]],
                               occupancy_grid: Optional[Any]) -> bool:
        """
        Validate optimized path doesn't violate constraints.
        
        Args:
            path: Optimized path
            occupancy_grid: Occupancy grid
            
        Returns:
            True if path is valid
        """
        if not path:
            return False
        
        # Check collision-free (if occupancy grid available)
        if occupancy_grid is not None:
            for point in path:
                grid_pos = occupancy_grid.world_to_grid(point)
                if not occupancy_grid.is_free(grid_pos):
                    return False
        
        # Check velocity constraints
        if len(path) > 1:
            dt = 0.1  # Assumed time step
            velocities = np.diff(np.array(path), axis=0) / dt
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            
            if np.any(velocity_magnitudes > self.max_velocity):
                return False
        
        # Check acceleration constraints
        if len(path) > 2:
            dt = 0.1
            velocities = np.diff(np.array(path), axis=0) / dt
            accelerations = np.diff(velocities, axis=0) / dt
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            if np.any(acceleration_magnitudes > self.max_acceleration):
                return False
        
        return True
    
    def compare_paths(self, original_path: List[Tuple[float, float, float]],
                     optimized_path: List[Tuple[float, float, float]]) -> Dict[str, Any]:
        """
        Compare original and optimized paths.
        
        Args:
            original_path: Original A* path
            optimized_path: Optimized path
            
        Returns:
            Comparison metrics
        """
        original_metrics = self.analyze_path_quality(original_path)
        optimized_metrics = self.analyze_path_quality(optimized_path)
        
        comparison = {
            'original': original_metrics,
            'optimized': optimized_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for key in ['total_length', 'mean_curvature', 'energy_estimate']:
            if key in original_metrics and key in optimized_metrics:
                original_val = original_metrics[key]
                optimized_val = optimized_metrics[key]
                
                if original_val > 0:
                    improvement_percent = (original_val - optimized_val) / original_val * 100
                    comparison['improvements'][key] = improvement_percent
        
        return comparison
