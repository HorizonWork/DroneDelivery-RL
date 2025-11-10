"""
Path Smoother
Advanced path smoothing utilities for better execution.
Integrates global A* and local S-RRT paths seamlessly.
"""

import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import splprep, splev, interp1d
from scipy.optimize import minimize_scalar
from dataclasses import dataclass

@dataclass
class SmoothingResult:
    """Path smoothing result."""
    smoothed_path: List[Tuple[float, float, float]]
    original_length: float
    smoothed_length: float
    max_curvature: float
    smoothing_time: float
    success: bool

class PathSmoother:
    """
    Advanced path smoothing for seamless global-local integration.
    Ensures smooth transitions between A* and S-RRT path segments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Smoothing parameters
        self.method = config.get('smoothing_method', 'spline')          # spline, bezier, gradient
        self.spline_degree = config.get('spline_degree', 3)             # cubic splines
        self.smoothness_factor = config.get('smoothness_factor', 0.1)   # 0 = exact fit, higher = smoother
        
        # Resampling parameters
        self.target_point_spacing = config.get('target_spacing', 0.5)   # meters between points
        self.min_points_per_meter = config.get('min_points_per_meter', 2)
        
        # Kinematic constraints
        self.max_velocity = config.get('max_velocity', 3.0)             # m/s
        self.max_acceleration = config.get('max_acceleration', 2.0)      # m/s²
        self.max_jerk = config.get('max_jerk', 5.0)                     # m/s³
        
        # Transition smoothing
        self.transition_blend_distance = config.get('transition_blend', 2.0)  # meters
        
        self.logger.info("Path Smoother initialized")
        self.logger.info(f"Method: {self.method}, Degree: {self.spline_degree}")
        self.logger.info(f"Constraints - Velocity: {self.max_velocity}m/s, "
                        f"Acceleration: {self.max_acceleration}m/s²")
    
    def smooth_path(self, path: List[Tuple[float, float, float]],
                   method: Optional[str] = None) -> SmoothingResult:
        """
        Smooth path using specified method.
        
        Args:
            path: Original path to smooth
            method: Smoothing method override
            
        Returns:
            Smoothing result
        """
        if len(path) < 3:
            return SmoothingResult(
                smoothed_path=path, original_length=0, smoothed_length=0,
                max_curvature=0, smoothing_time=0, success=False
            )
        
        smoothing_start = time.time()
        smoothing_method = method or self.method
        
        try:
            # Calculate original path length
            original_length = self._calculate_path_length(path)
            
            # Apply smoothing
            if smoothing_method == 'spline':
                smoothed_path = self._smooth_with_spline(path)
            elif smoothing_method == 'bezier':
                smoothed_path = self._smooth_with_bezier(path)
            elif smoothing_method == 'gradient':
                smoothed_path = self._smooth_with_gradient_descent(path)
            else:
                smoothed_path = path  # No smoothing
            
            # Calculate smoothed path metrics
            smoothed_length = self._calculate_path_length(smoothed_path)
            max_curvature = self._calculate_max_curvature(smoothed_path)
            
            # Validate kinematic constraints
            if not self._validate_kinematic_constraints(smoothed_path):
                self.logger.warning("Smoothed path violates kinematic constraints")
                smoothed_path = self._enforce_kinematic_constraints(smoothed_path)
                max_curvature = self._calculate_max_curvature(smoothed_path)
            
            smoothing_time = time.time() - smoothing_start
            
            return SmoothingResult(
                smoothed_path=smoothed_path,
                original_length=original_length,
                smoothed_length=smoothed_length,
                max_curvature=max_curvature,
                smoothing_time=smoothing_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Path smoothing failed: {e}")
            return SmoothingResult(
                smoothed_path=path, original_length=0, smoothed_length=0,
                max_curvature=0, smoothing_time=time.time() - smoothing_start,
                success=False
            )
    
    def _smooth_with_spline(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Smooth path using cubic spline interpolation.
        
        Args:
            path: Original path
            
        Returns:
            Spline-smoothed path
        """
        if len(path) < 4:  # Need at least 4 points for cubic spline
            return path
        
        path_array = np.array(path)
        
        # Parameterize by cumulative distance
        distances = np.cumsum(np.concatenate(([0], np.linalg.norm(np.diff(path_array, axis=0), axis=1))))
        
        # Fit spline
        tck, u = splprep([path_array[:, 0], path_array[:, 1], path_array[:, 2]], 
                        u=distances, s=self.smoothness_factor, k=self.spline_degree)
        
        # Generate smoothed path with target spacing
        total_distance = distances[-1]
        num_points = max(len(path), int(total_distance / self.target_point_spacing))
        u_new = np.linspace(0, total_distance, num_points)
        
        smoothed_coords = splev(u_new, tck)
        smoothed_path = list(zip(smoothed_coords[0], smoothed_coords[1], smoothed_coords[2]))
        
        return [(float(x), float(y), float(z)) for x, y, z in smoothed_path]
    
    def _smooth_with_bezier(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Smooth path using Bezier curve approximation.
        
        Args:
            path: Original path
            
        Returns:
            Bezier-smoothed path
        """
        # Simplified Bezier smoothing - fit Bezier curves to path segments
        if len(path) < 4:
            return path
        
        smoothed_path = []
        
        # Process path in overlapping segments
        segment_size = 4
        for i in range(0, len(path) - segment_size + 1, segment_size - 1):
            segment = path[i:i + segment_size]
            
            # Generate Bezier curve for segment
            bezier_points = self._generate_bezier_curve(segment, 10)  # 10 points per segment
            
            # Avoid duplicate points at segment boundaries
            if i > 0:
                bezier_points = bezier_points[1:]
            
            smoothed_path.extend(bezier_points)
        
        # Add remaining path points if any
        if len(path) % (segment_size - 1) != 1:
            smoothed_path.extend(path[len(smoothed_path):])
        
        return smoothed_path
    
    def _generate_bezier_curve(self, control_points: List[Tuple[float, float, float]], 
                              num_points: int) -> List[Tuple[float, float, float]]:
        """Generate Bezier curve from control points."""
        if len(control_points) != 4:
            return control_points
        
        P0, P1, P2, P3 = [np.array(p) for p in control_points]
        
        # Generate Bezier curve points
        bezier_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Cubic Bezier formula
            point = ((1-t)**3 * P0 + 
                    3*(1-t)**2*t * P1 + 
                    3*(1-t)*t**2 * P2 + 
                    t**3 * P3)
            
            bezier_points.append(tuple(point))
        
        return bezier_points
    
    def _smooth_with_gradient_descent(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Smooth path using gradient descent optimization.
        
        Args:
            path: Original path
            
        Returns:
            Gradient-optimized path
        """
        # Simple iterative smoothing
        if len(path) < 3:
            return path
        
        path_array = np.array(path)
        smoothed = path_array.copy()
        
        # Apply iterative averaging (simplified gradient descent)
        for iteration in range(10):
            for i in range(1, len(smoothed) - 1):
                # Weighted average of neighbors
                neighbor_avg = (smoothed[i-1] + smoothed[i+1]) / 2
                
                # Blend with current position
                alpha = 0.3  # Smoothing strength
                smoothed[i] = (1 - alpha) * smoothed[i] + alpha * neighbor_avg
        
        return [(float(p[0]), float(p[1]), float(p[2])) for p in smoothed]
    
    def blend_paths(self, global_path: List[Tuple[float, float, float]],
                   local_path: List[Tuple[float, float, float]],
                   blend_point: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Smoothly blend global A* path with local S-RRT path.
        
        Args:
            global_path: Global A* path
            local_path: Local S-RRT path
            blend_point: Position where paths should blend
            
        Returns:
            Blended path
        """
        if not global_path or not local_path:
            return global_path or local_path
        
        # Find blend points in both paths
        global_blend_idx = self._find_closest_waypoint(global_path, blend_point)
        local_blend_idx = self._find_closest_waypoint(local_path, blend_point)
        
        # Create transition zone
        transition_segment = self._create_smooth_transition(
            global_path[max(0, global_blend_idx-2):global_blend_idx+1],
            local_path[local_blend_idx:min(len(local_path), local_blend_idx+3)]
        )
        
        # Combine path segments
        blended_path = (
            global_path[:global_blend_idx] +
            transition_segment +
            local_path[local_blend_idx+1:]
        )
        
        return blended_path
    
    def _find_closest_waypoint(self, path: List[Tuple[float, float, float]],
                              target_point: Tuple[float, float, float]) -> int:
        """Find index of closest waypoint in path."""
        target_array = np.array(target_point)
        min_distance = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(path):
            distance = np.linalg.norm(np.array(waypoint) - target_array)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _create_smooth_transition(self, end_segment: List[Tuple[float, float, float]],
                                 start_segment: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Create smooth transition between path segments."""
        if not end_segment or not start_segment:
            return []
        
        # Use simple linear blending
        transition_points = []
        num_transition_points = 5
        
        end_point = np.array(end_segment[-1])
        start_point = np.array(start_segment[0])
        
        for i in range(num_transition_points):
            t = i / (num_transition_points - 1)
            transition_point = end_point + t * (start_point - end_point)
            transition_points.append(tuple(transition_point))
        
        return transition_points
    
    def _calculate_path_length(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            segment_length = np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
            total_length += segment_length
        
        return total_length
    
    def _calculate_max_curvature(self, path: List[Tuple[float, float, float]]) -> float:
        """Calculate maximum curvature along path."""
        if len(path) < 3:
            return 0.0
        
        path_array = np.array(path)
        max_curvature = 0.0
        
        for i in range(1, len(path_array) - 1):
            # Three consecutive points
            p1 = path_array[i-1]
            p2 = path_array[i]
            p3 = path_array[i+1]
            
            # Calculate curvature using discrete approximation
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product magnitude gives curvature numerator
            if len(v1) == 3 and len(v2) == 3:  # 3D case
                cross_product = np.cross(v1, v2)
                numerator = np.linalg.norm(cross_product)
                denominator = np.linalg.norm(v1) ** 3
                
                if denominator > 1e-6:
                    curvature = numerator / denominator
                    max_curvature = max(max_curvature, curvature)
        
        return max_curvature
    
    def _validate_kinematic_constraints(self, path: List[Tuple[float, float, float]]) -> bool:
        """
        Validate path satisfies kinematic constraints.
        
        Args:
            path: Path to validate
            
        Returns:
            True if constraints satisfied
        """
        if len(path) < 2:
            return True
        
        # Estimate time between waypoints
        dt = self.target_point_spacing / self.max_velocity  # Conservative estimate
        
        path_array = np.array(path)
        
        # Check velocity constraints
        velocities = np.diff(path_array, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        if np.any(velocity_magnitudes > self.max_velocity):
            return False
        
        # Check acceleration constraints
        if len(path) > 2:
            accelerations = np.diff(velocities, axis=0) / dt
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            if np.any(acceleration_magnitudes > self.max_acceleration):
                return False
        
        # Check jerk constraints
        if len(path) > 3:
            jerks = np.diff(accelerations, axis=0) / dt
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
            
            if np.any(jerk_magnitudes > self.max_jerk):
                return False
        
        return True
    
    def _enforce_kinematic_constraints(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Enforce kinematic constraints by adjusting path.
        
        Args:
            path: Original path
            
        Returns:
            Kinematically feasible path
        """
        if len(path) < 2:
            return path
        
        # Simple approach: increase point spacing to reduce velocities
        path_array = np.array(path)
        
        # Calculate required spacing to satisfy velocity constraint
        segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
        max_segment_length = np.max(segment_lengths)
        
        # If max segment > max_velocity * reasonable_dt, need more points
        reasonable_dt = 0.1  # 10Hz waypoint following
        max_allowed_length = self.max_velocity * reasonable_dt
        
        if max_segment_length > max_allowed_length:
            # Resample path with smaller spacing
            new_spacing = max_allowed_length / 2  # Conservative
            return self.resample_path(path, new_spacing)
        
        return path
    
    def resample_path(self, path: List[Tuple[float, float, float]], 
                     target_spacing: float) -> List[Tuple[float, float, float]]:
        """
        Resample path to uniform point spacing.
        
        Args:
            path: Original path
            target_spacing: Desired spacing between points
            
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
            return path
        
        # Create interpolation functions for each dimension
        interp_x = interp1d(cumulative_distances, path_array[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(cumulative_distances, path_array[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_z = interp1d(cumulative_distances, path_array[:, 2], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Generate resampled points
        num_points = int(np.ceil(total_length / target_spacing)) + 1
        sample_distances = np.linspace(0, total_length, num_points)
        
        resampled_path = []
        for dist in sample_distances:
            x = float(interp_x(dist))
            y = float(interp_y(dist))
            z = float(interp_z(dist))
            resampled_path.append((x, y, z))
        
        return resampled_path
    
    def calculate_path_quality(self, path: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Calculate path quality metrics.
        
        Args:
            path: Path to analyze
            
        Returns:
            Quality metrics dictionary
        """
        if len(path) < 2:
            return {'error': 'Path too short'}
        
        # Basic metrics
        total_length = self._calculate_path_length(path)
        max_curvature = self._calculate_max_curvature(path)
        
        # Smoothness metrics
        path_array = np.array(path)
        
        # Calculate segment length variation (lower is smoother)
        segment_lengths = np.linalg.norm(np.diff(path_array, axis=0), axis=1)
        length_variation = np.std(segment_lengths) / np.mean(segment_lengths) if len(segment_lengths) > 0 else 0
        
        # Calculate direction changes
        if len(path) > 2:
            directions = np.diff(path_array, axis=0)
            unit_directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
            
            direction_changes = []
            for i in range(len(unit_directions) - 1):
                dot_product = np.clip(np.dot(unit_directions[i], unit_directions[i+1]), -1, 1)
                angle_change = np.arccos(dot_product)
                direction_changes.append(angle_change)
            
            mean_direction_change = np.mean(direction_changes) if direction_changes else 0
            max_direction_change = np.max(direction_changes) if direction_changes else 0
        else:
            mean_direction_change = max_direction_change = 0
        
        return {
            'total_length': float(total_length),
            'max_curvature': float(max_curvature),
            'length_variation_coefficient': float(length_variation),
            'mean_direction_change': float(mean_direction_change),
            'max_direction_change': float(max_direction_change),
            'smoothness_score': float(1.0 / (1.0 + max_curvature + mean_direction_change)),
            'num_waypoints': len(path)
        }
    
    def get_smoother_info(self) -> Dict[str, Any]:
        """Get path smoother configuration information."""
        return {
            'method': self.method,
            'parameters': {
                'spline_degree': self.spline_degree,
                'smoothness_factor': self.smoothness_factor,
                'target_spacing': self.target_point_spacing,
                'transition_blend_distance': self.transition_blend_distance
            },
            'constraints': {
                'max_velocity': self.max_velocity,
                'max_acceleration': self.max_acceleration,
                'max_jerk': self.max_jerk
            },
            'methods_available': ['spline', 'bezier', 'gradient']
        }
