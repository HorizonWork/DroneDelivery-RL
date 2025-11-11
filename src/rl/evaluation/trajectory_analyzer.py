"""
Trajectory Analyzer
Comprehensive trajectory analysis for path quality and navigation performance.
Analyzes smoothness, efficiency, and adherence to planned paths.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.interpolate import interp1d
import json

@dataclass
class TrajectoryMetrics:
    """Comprehensive trajectory quality metrics."""
    path_length: float                    # Total distance traveled
    path_efficiency: float               # Ratio of straight-line to actual distance
    smoothness_score: float              # 0-1 score (1 = perfectly smooth)
    curvature_metrics: Dict[str, float]  # Max, mean, std curvature
    velocity_metrics: Dict[str, float]   # Velocity profile analysis
    acceleration_metrics: Dict[str, float] # Acceleration profile analysis
    waypoint_tracking_error: float       # Mean deviation from waypoints
    floor_transition_efficiency: float   # Efficiency of vertical movements
    obstacle_avoidance_quality: float    # Quality of obstacle avoidance maneuvers

@dataclass
class FlightPhase:
    """Individual flight phase analysis."""
    phase_name: str                      # takeoff, cruise, maneuvering, landing
    duration: float                      # seconds
    distance: float                      # meters
    energy_consumption: float            # Joules
    average_velocity: float              # m/s
    max_acceleration: float              # m/s²
    smoothness: float                    # 0-1 score

class TrajectoryAnalyzer:
    """
    Advanced trajectory analysis system.
    Provides detailed analysis of flight paths, smoothness, and navigation quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analysis parameters
        self.smoothness_window = config.get('smoothness_window', 5)      # Points for smoothness calc
        self.velocity_smoothing = config.get('velocity_smoothing', 0.1)   # Low-pass filter factor
        self.curvature_threshold = config.get('curvature_threshold', 0.5) # 1/m max acceptable curvature
        
        # Flight phase detection
        self.altitude_change_threshold = config.get('altitude_threshold', 0.5)  # m
        self.velocity_threshold = config.get('velocity_threshold', 0.3)         # m/s
        self.maneuver_acceleration_threshold = config.get('maneuver_threshold', 1.0)  # m/s²
        
        # Efficiency benchmarks
        self.efficiency_benchmarks = {
            'min_path_efficiency': 0.80,     # 80% efficiency minimum
            'max_curvature': 0.5,            # 1/m maximum curvature
            'max_jerk': 3.0,                 # m/s³ maximum jerk
            'tracking_error_threshold': 0.5   # m maximum tracking error
        }
        
        # Building dimensions for context
        self.building_dims = config.get('building_dims', {
            'length': 20.0, 'width': 40.0, 'height': 15.0  # 5 floors
        })
        
        self.logger.info("Trajectory Analyzer initialized")
        self.logger.info(f"Smoothness window: {self.smoothness_window} points")
        self.logger.info(f"Efficiency benchmarks: {self.efficiency_benchmarks}")
    
    def analyze_episodes(self, episode_results: List) -> Dict[str, Any]:
        """
        Analyze trajectories from multiple episodes.
        
        Args:
            episode_results: List of episode results with trajectory data
            
        Returns:
            Comprehensive trajectory analysis
        """
        if not episode_results:
            return {'error': 'No episode results provided'}
        
        # Filter episodes with trajectory data
        episodes_with_trajectories = [
            ep for ep in episode_results 
            if hasattr(ep, 'trajectory') and ep.trajectory and len(ep.trajectory) > 3
        ]
        
        if not episodes_with_trajectories:
            return {'error': 'No episodes with trajectory data'}
        
        self.logger.info(f"Analyzing {len(episodes_with_trajectories)} trajectories")
        
        # Analyze individual trajectories
        trajectory_metrics = []
        for episode in episodes_with_trajectories:
            metrics = self._analyze_single_trajectory(episode)
            trajectory_metrics.append(metrics)
        
        # Aggregate analysis
        analysis = self._aggregate_trajectory_analysis(trajectory_metrics)
        
        # Flight phase analysis
        phase_analysis = self._analyze_flight_phases(episodes_with_trajectories)
        analysis['flight_phases'] = phase_analysis
        
        # Multi-floor navigation analysis
        floor_analysis = self._analyze_floor_navigation(episodes_with_trajectories)
        analysis['floor_navigation'] = floor_analysis
        
        # Quality assessment
        quality_assessment = self._assess_trajectory_quality(trajectory_metrics)
        analysis['quality_assessment'] = quality_assessment
        
        return analysis
    
    def _analyze_single_trajectory(self, episode) -> TrajectoryMetrics:
        """
        Analyze single episode trajectory.
        
        Args:
            episode: Episode result with trajectory data
            
        Returns:
            Trajectory metrics for episode
        """
        trajectory = np.array(episode.trajectory)
        timestamps = getattr(episode, 'timestamps', None)
        
        if timestamps is None or len(timestamps) != len(trajectory):
            # Generate timestamps assuming 20Hz
            timestamps = np.linspace(0, episode.flight_time, len(trajectory))
        
        # Path length and efficiency
        path_length = self._calculate_path_length(trajectory)
        straight_line_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        path_efficiency = straight_line_distance / path_length if path_length > 0 else 0
        
        # Smoothness analysis
        smoothness_score = self._calculate_smoothness(trajectory, timestamps)
        
        # Curvature analysis
        curvature_metrics = self._analyze_curvature(trajectory)
        
        # Velocity and acceleration analysis
        velocity_metrics = self._analyze_velocity_profile(trajectory, timestamps)
        acceleration_metrics = self._analyze_acceleration_profile(trajectory, timestamps)
        
        # Waypoint tracking (if planned path available)
        waypoint_tracking_error = getattr(episode, 'path_deviation', 0.0)
        
        # Floor transition analysis
        floor_transition_efficiency = self._analyze_floor_transitions(trajectory)
        
        # Obstacle avoidance quality (simplified)
        obstacle_avoidance_quality = 1.0 - min(1.0, getattr(episode, 'clearance_violations', 0) * 0.1)
        
        return TrajectoryMetrics(
            path_length=path_length,
            path_efficiency=path_efficiency,
            smoothness_score=smoothness_score,
            curvature_metrics=curvature_metrics,
            velocity_metrics=velocity_metrics,
            acceleration_metrics=acceleration_metrics,
            waypoint_tracking_error=waypoint_tracking_error,
            floor_transition_efficiency=floor_transition_efficiency,
            obstacle_avoidance_quality=obstacle_avoidance_quality
        )
    
    def _calculate_path_length(self, trajectory: np.ndarray) -> float:
        """Calculate total path length."""
        if len(trajectory) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return float(np.sum(distances))
    
    def _calculate_smoothness(self, trajectory: np.ndarray, timestamps: np.ndarray) -> float:
        """
        Calculate trajectory smoothness score.
        
        Args:
            trajectory: Position trajectory
            timestamps: Time series
            
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(trajectory) < 3:
            return 1.0
        
        # Calculate jerk (third derivative of position)
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
        
        # First derivative (velocity)
        velocities = np.diff(trajectory, axis=0) / dt
        
        # Second derivative (acceleration)
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / dt
        else:
            return 1.0
        
        # Third derivative (jerk)
        if len(accelerations) > 1:
            jerks = np.diff(accelerations, axis=0) / dt
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
            
            # Smoothness inversely related to jerk
            mean_jerk = np.mean(jerk_magnitudes)
            smoothness = 1.0 / (1.0 + mean_jerk / 3.0)  # Normalize by typical jerk limit
            
            return float(np.clip(smoothness, 0.0, 1.0))
        
        return 1.0
    
    def _analyze_curvature(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Analyze trajectory curvature.
        
        Args:
            trajectory: Position trajectory
            
        Returns:
            Curvature metrics dictionary
        """
        if len(trajectory) < 3:
            return {'max_curvature': 0, 'mean_curvature': 0, 'std_curvature': 0}
        
        curvatures = []
        
        # Calculate curvature at each point using three consecutive points
        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i-1], trajectory[i], trajectory[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Curvature calculation using cross product
            if len(v1) == 3:  # 3D case
                cross_product = np.cross(v1, v2)
                cross_magnitude = np.linalg.norm(cross_product)
                v1_magnitude = np.linalg.norm(v1)
                
                if v1_magnitude > 1e-6:
                    curvature = cross_magnitude / (v1_magnitude ** 3)
                    curvatures.append(curvature)
        
        if curvatures:
            return {
                'max_curvature': float(np.max(curvatures)),
                'mean_curvature': float(np.mean(curvatures)),
                'std_curvature': float(np.std(curvatures)),
                'curvature_violations': int(np.sum(np.array(curvatures) > self.curvature_threshold))
            }
        
        return {'max_curvature': 0, 'mean_curvature': 0, 'std_curvature': 0, 'curvature_violations': 0}
    
    def _analyze_velocity_profile(self, trajectory: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """Analyze velocity profile."""
        if len(trajectory) < 2:
            return {'mean_velocity': 0, 'max_velocity': 0, 'velocity_variance': 0}
        
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
        velocities = np.diff(trajectory, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        return {
            'mean_velocity': float(np.mean(velocity_magnitudes)),
            'max_velocity': float(np.max(velocity_magnitudes)),
            'min_velocity': float(np.min(velocity_magnitudes)),
            'std_velocity': float(np.std(velocity_magnitudes)),
            'velocity_variance': float(np.var(velocity_magnitudes))
        }
    
    def _analyze_acceleration_profile(self, trajectory: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
        """Analyze acceleration profile."""
        if len(trajectory) < 3:
            return {'mean_acceleration': 0, 'max_acceleration': 0}
        
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
        velocities = np.diff(trajectory, axis=0) / dt
        
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0) / dt
            acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
            
            return {
                'mean_acceleration': float(np.mean(acceleration_magnitudes)),
                'max_acceleration': float(np.max(acceleration_magnitudes)),
                'std_acceleration': float(np.std(acceleration_magnitudes)),
                'high_acceleration_events': int(np.sum(acceleration_magnitudes > self.maneuver_acceleration_threshold))
            }
        
        return {'mean_acceleration': 0, 'max_acceleration': 0, 'std_acceleration': 0, 'high_acceleration_events': 0}
    
    def _analyze_floor_transitions(self, trajectory: np.ndarray) -> float:
        """
        Analyze efficiency of floor transitions.
        
        Args:
            trajectory: Position trajectory
            
        Returns:
            Floor transition efficiency score (0-1)
        """
        if len(trajectory) < 2:
            return 1.0
        
        # Calculate vertical movements
        altitudes = trajectory[:, 2]
        vertical_changes = np.abs(np.diff(altitudes))
        
        # Count significant floor transitions (>2.5m altitude change)
        floor_transitions = np.sum(vertical_changes > 2.5)
        
        if floor_transitions == 0:
            return 1.0  # Single floor - perfect efficiency
        
        # Calculate efficiency: actual vertical distance vs minimum possible
        total_vertical_distance = np.sum(vertical_changes)
        min_vertical_distance = abs(altitudes[-1] - altitudes[0])  # Direct vertical path
        
        if total_vertical_distance > 0:
            efficiency = min_vertical_distance / total_vertical_distance
            return float(np.clip(efficiency, 0.0, 1.0))
        
        return 1.0
    
    def _analyze_flight_phases(self, episodes_with_trajectories: List) -> Dict[str, Any]:
        """
        Analyze different flight phases across episodes.
        
        Args:
            episodes_with_trajectories: Episodes with trajectory data
            
        Returns:
            Flight phase analysis
        """
        phase_data = defaultdict(list)
        
        for episode in episodes_with_trajectories:
            trajectory = np.array(episode.trajectory)
            
            if len(trajectory) < 10:  # Need sufficient data points
                continue
            
            # Detect flight phases
            phases = self._detect_flight_phases(trajectory, episode.flight_time)
            
            for phase in phases:
                phase_data[phase.phase_name].append(phase)
        
        # Aggregate phase statistics
        phase_analysis = {}
        for phase_name, phase_list in phase_data.items():
            if phase_list:
                phase_analysis[phase_name] = {
                    'count': len(phase_list),
                    'mean_duration': float(np.mean([p.duration for p in phase_list])),
                    'mean_distance': float(np.mean([p.distance for p in phase_list])),
                    'mean_energy': float(np.mean([p.energy_consumption for p in phase_list])),
                    'mean_velocity': float(np.mean([p.average_velocity for p in phase_list])),
                    'mean_smoothness': float(np.mean([p.smoothness for p in phase_list]))
                }
        
        return phase_analysis
    
    def _detect_flight_phases(self, trajectory: np.ndarray, total_time: float) -> List[FlightPhase]:
        """
        Detect flight phases in trajectory.
        
        Args:
            trajectory: Position trajectory
            total_time: Total flight time
            
        Returns:
            List of detected flight phases
        """
        phases = []
        
        if len(trajectory) < 5:
            return phases
        
        # Simple phase detection based on altitude changes
        altitudes = trajectory[:, 2]
        dt = total_time / len(trajectory)
        
        # Calculate velocity profile
        velocities = np.diff(trajectory, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Detect takeoff (initial altitude increase)
        takeoff_end = self._find_takeoff_end(altitudes)
        if takeoff_end > 0:
            takeoff_trajectory = trajectory[:takeoff_end+1]
            takeoff_phase = FlightPhase(
                phase_name='takeoff',
                duration=takeoff_end * dt,
                distance=self._calculate_path_length(takeoff_trajectory),
                energy_consumption=0.0,  # Would calculate if power data available
                average_velocity=float(np.mean(velocity_magnitudes[:takeoff_end])) if takeoff_end > 0 else 0.0,
                max_acceleration=0.0,    # Would calculate if needed
                smoothness=self._calculate_smoothness(takeoff_trajectory, np.arange(len(takeoff_trajectory)) * dt)
            )
            phases.append(takeoff_phase)
        
        # Detect cruise phase (stable altitude and velocity)
        cruise_segments = self._find_cruise_segments(trajectory, velocity_magnitudes)
        for start_idx, end_idx in cruise_segments:
            cruise_trajectory = trajectory[start_idx:end_idx+1]
            cruise_phase = FlightPhase(
                phase_name='cruise',
                duration=(end_idx - start_idx) * dt,
                distance=self._calculate_path_length(cruise_trajectory),
                energy_consumption=0.0,
                average_velocity=float(np.mean(velocity_magnitudes[start_idx:end_idx])) if end_idx > start_idx else 0.0,
                max_acceleration=0.0,
                smoothness=self._calculate_smoothness(cruise_trajectory, np.arange(len(cruise_trajectory)) * dt)
            )
            phases.append(cruise_phase)
        
        return phases
    
    def _find_takeoff_end(self, altitudes: np.ndarray) -> int:
        """Find end of takeoff phase."""
        if len(altitudes) < 2:
            return 0
        
        # Look for sustained altitude (no longer climbing)
        for i in range(1, len(altitudes)):
            if i > 5:  # Minimum takeoff duration
                recent_change = altitudes[i] - altitudes[max(0, i-3)]
                if recent_change < self.altitude_change_threshold:
                    return i
        
        return len(altitudes) // 4  # Fallback: first quarter
    
    def _find_cruise_segments(self, trajectory: np.ndarray, velocities: np.ndarray) -> List[Tuple[int, int]]:
        """Find cruise segments (stable velocity and altitude)."""
        if len(trajectory) < 10:
            return []
        
        cruise_segments = []
        altitudes = trajectory[:, 2]
        
        # Find segments with stable altitude and velocity
        window_size = 5
        for i in range(len(trajectory) - window_size):
            window_altitudes = altitudes[i:i+window_size]
            window_velocities = velocities[i:i+window_size] if i+window_size <= len(velocities) else []
            
            if len(window_velocities) == 0:
                continue
            
            # Check stability criteria
            altitude_stable = np.std(window_altitudes) < self.altitude_change_threshold
            velocity_stable = np.std(window_velocities) < self.velocity_threshold
            
            if altitude_stable and velocity_stable:
                # Extend segment
                segment_start = i
                segment_end = i + window_size
                
                # Try to extend segment
                for j in range(i + window_size, min(len(trajectory), i + window_size + 10)):
                    if (j < len(velocities) and 
                        abs(altitudes[j] - np.mean(window_altitudes)) < self.altitude_change_threshold and
                        abs(velocities[j] - np.mean(window_velocities)) < self.velocity_threshold):
                        segment_end = j
                    else:
                        break
                
                if segment_end - segment_start >= window_size:
                    cruise_segments.append((segment_start, segment_end))
        
        return cruise_segments
    
    def _aggregate_trajectory_analysis(self, trajectory_metrics: List[TrajectoryMetrics]) -> Dict[str, Any]:
        """
        Aggregate analysis across all trajectories.
        
        Args:
            trajectory_metrics: List of individual trajectory metrics
            
        Returns:
            Aggregated trajectory analysis
        """
        if not trajectory_metrics:
            return {}
        
        # Path efficiency aggregation
        path_efficiencies = [tm.path_efficiency for tm in trajectory_metrics]
        
        # Smoothness aggregation  
        smoothness_scores = [tm.smoothness_score for tm in trajectory_metrics]
        
        # Curvature aggregation
        max_curvatures = [tm.curvature_metrics.get('max_curvature', 0) for tm in trajectory_metrics]
        mean_curvatures = [tm.curvature_metrics.get('mean_curvature', 0) for tm in trajectory_metrics]
        
        # Velocity aggregation
        mean_velocities = [tm.velocity_metrics.get('mean_velocity', 0) for tm in trajectory_metrics]
        max_velocities = [tm.velocity_metrics.get('max_velocity', 0) for tm in trajectory_metrics]
        
        aggregated = {
            'path_efficiency': {
                'mean': float(np.mean(path_efficiencies)),
                'std': float(np.std(path_efficiencies)),
                'min': float(np.min(path_efficiencies)),
                'max': float(np.max(path_efficiencies)),
                'meets_benchmark': np.mean(path_efficiencies) >= self.efficiency_benchmarks['min_path_efficiency']
            },
            'smoothness': {
                'mean_score': float(np.mean(smoothness_scores)),
                'std_score': float(np.std(smoothness_scores)),
                'smooth_trajectories_percent': float(np.mean(np.array(smoothness_scores) > 0.8) * 100)
            },
            'curvature': {
                'mean_max_curvature': float(np.mean(max_curvatures)),
                'overall_mean_curvature': float(np.mean(mean_curvatures)),
                'curvature_violations': int(np.sum(np.array(max_curvatures) > self.efficiency_benchmarks['max_curvature']))
            },
            'velocity': {
                'mean_velocity': float(np.mean(mean_velocities)),
                'mean_max_velocity': float(np.mean(max_velocities)),
                'velocity_consistency': 1.0 - float(np.std(mean_velocities) / (np.mean(mean_velocities) + 1e-6))
            }
        }
        
        return aggregated
    
    def _analyze_floor_navigation(self, episodes_with_trajectories: List) -> Dict[str, Any]:
        """
        Analyze multi-floor navigation patterns.
        
        Args:
            episodes_with_trajectories: Episodes with trajectory data
            
        Returns:
            Floor navigation analysis
        """
        floor_data = {
            'total_floor_transitions': 0,
            'transition_efficiencies': [],
            'floors_visited': defaultdict(int),
            'transition_times': [],
            'vertical_distances': []
        }
        
        for episode in episodes_with_trajectories:
            trajectory = np.array(episode.trajectory)
            altitudes = trajectory[:, 2]
            
            # Count floor transitions
            floor_changes = 0
            current_floor = int(altitudes[0] // 3.0)  # 3m per floor
            
            transition_start_idx = 0
            for i, altitude in enumerate(altitudes):
                floor = int(altitude // 3.0)
                if floor != current_floor:
                    floor_changes += 1
                    
                    # Calculate transition efficiency
                    transition_segment = trajectory[transition_start_idx:i+1]
                    if len(transition_segment) > 1:
                        vertical_distance = abs(altitudes[i] - altitudes[transition_start_idx])
                        path_distance = self._calculate_path_length(transition_segment)
                        
                        if path_distance > 0:
                            transition_efficiency = vertical_distance / path_distance
                            floor_data['transition_efficiencies'].append(transition_efficiency)
                    
                    current_floor = floor
                    transition_start_idx = i
                
                floor_data['floors_visited'][floor] += 1
            
            floor_data['total_floor_transitions'] += floor_changes
        
        # Aggregate floor navigation metrics
        analysis = {
            'total_episodes_analyzed': len(episodes_with_trajectories),
            'episodes_with_transitions': len([1 for ep in episodes_with_trajectories 
                                            if self._has_floor_transitions(np.array(ep.trajectory))]),
            'average_transitions_per_episode': floor_data['total_floor_transitions'] / len(episodes_with_trajectories),
            'floor_usage_distribution': dict(floor_data['floors_visited']),
        }
        
        if floor_data['transition_efficiencies']:
            analysis['transition_efficiency'] = {
                'mean': float(np.mean(floor_data['transition_efficiencies'])),
                'std': float(np.std(floor_data['transition_efficiencies'])),
                'min': float(np.min(floor_data['transition_efficiencies'])),
                'max': float(np.max(floor_data['transition_efficiencies']))
            }
        
        return analysis
    
    def _has_floor_transitions(self, trajectory: np.ndarray) -> bool:
        """Check if trajectory has floor transitions."""
        if len(trajectory) < 2:
            return False
        
        altitudes = trajectory[:, 2]
        max_altitude_change = np.max(np.abs(np.diff(altitudes)))
        
        return max_altitude_change > 2.5  # Threshold for floor transition
    
    def _assess_trajectory_quality(self, trajectory_metrics: List[TrajectoryMetrics]) -> Dict[str, Any]:
        """
        Assess overall trajectory quality.
        
        Args:
            trajectory_metrics: List of trajectory metrics
            
        Returns:
            Quality assessment results
        """
        if not trajectory_metrics:
            return {}
        
        # Quality criteria
        quality_scores = []
        
        for metrics in trajectory_metrics:
            # Weighted quality score
            path_eff_score = min(1.0, metrics.path_efficiency / 0.8)  # Target 80% efficiency
            smoothness_score = metrics.smoothness_score
            curvature_score = 1.0 - min(1.0, metrics.curvature_metrics.get('max_curvature', 0) / 1.0)
            
            # Combined quality score
            quality_score = (0.4 * path_eff_score + 
                           0.4 * smoothness_score + 
                           0.2 * curvature_score)
            
            quality_scores.append(quality_score)
        
        # Quality assessment
        mean_quality = np.mean(quality_scores)
        
        quality_levels = {
            'excellent': np.sum(np.array(quality_scores) > 0.85),
            'good': np.sum((np.array(quality_scores) > 0.70) & (np.array(quality_scores) <= 0.85)),
            'fair': np.sum((np.array(quality_scores) > 0.50) & (np.array(quality_scores) <= 0.70)),
            'poor': np.sum(np.array(quality_scores) <= 0.50)
        }
        
        return {
            'overall_quality_score': float(mean_quality),
            'quality_distribution': quality_levels,
            'trajectories_above_threshold': int(np.sum(np.array(quality_scores) > 0.70)),
            'quality_consistency': 1.0 - float(np.std(quality_scores)),
            'improvement_potential': max(0.0, 0.85 - mean_quality)  # Room for improvement
        }
    
    def export_trajectory_analysis(self, analysis: Dict[str, Any], filename: str = "trajectory_analysis.json"):
        """Export trajectory analysis to file."""
        output_path = Path(self.config.get('output_dir', '.')) / filename
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Trajectory analysis exported to {output_path}")
    
    def get_trajectory_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate trajectory analysis summary."""
        summary_lines = []
        summary_lines.append("TRAJECTORY ANALYSIS SUMMARY")
        summary_lines.append("=" * 35)
        
        # Path efficiency
        if 'path_efficiency' in analysis:
            eff = analysis['path_efficiency']
            summary_lines.append(f"Path Efficiency: {eff.get('mean', 0):.3f} ± {eff.get('std', 0):.3f}")
            summary_lines.append(f"Efficiency Target: {'✓ MET' if eff.get('meets_benchmark', False) else '✗ NOT MET'}")
        
        # Smoothness
        if 'smoothness' in analysis:
            smooth = analysis['smoothness']
            summary_lines.append(f"Smoothness Score: {smooth.get('mean_score', 0):.3f}")
            summary_lines.append(f"Smooth Trajectories: {smooth.get('smooth_trajectories_percent', 0):.1f}%")
        
        # Quality assessment
        if 'quality_assessment' in analysis:
            quality = analysis['quality_assessment']
            summary_lines.append(f"Overall Quality: {quality.get('overall_quality_score', 0):.3f}/1.0")
            
            dist = quality.get('quality_distribution', {})
            summary_lines.append(f"Quality Distribution - Excellent: {dist.get('excellent', 0)}, "
                                f"Good: {dist.get('good', 0)}, Fair: {dist.get('fair', 0)}, "
                                f"Poor: {dist.get('poor', 0)}")
        
        return "\n".join(summary_lines)
