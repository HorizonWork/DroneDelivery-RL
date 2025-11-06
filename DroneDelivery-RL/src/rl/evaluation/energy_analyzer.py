"""
Energy Analyzer
Detailed analysis of energy consumption patterns and efficiency.
Implements power modeling and energy optimization analysis.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class EnergyBreakdown:
    """Energy consumption breakdown."""
    total_energy: float              # J
    thrust_energy: float            # J - propulsion
    avionics_energy: float          # J - sensors, computation
    communication_energy: float     # J - radio, data transmission
    idle_energy: float              # J - hovering/stationary
    maneuver_energy: float          # J - acceleration/deceleration
    
@dataclass  
class PowerProfile:
    """Power consumption profile over time."""
    timestamps: List[float] = field(default_factory=list)
    power_values: List[float] = field(default_factory=list)  # Watts
    velocity_profile: List[float] = field(default_factory=list)
    acceleration_profile: List[float] = field(default_factory=list)
    
    def add_sample(self, timestamp: float, power: float, velocity: float = 0.0, acceleration: float = 0.0):
        """Add power sample to profile."""
        self.timestamps.append(timestamp)
        self.power_values.append(power)
        self.velocity_profile.append(velocity)
        self.acceleration_profile.append(acceleration)

class EnergyAnalyzer:
    """
    Comprehensive energy consumption analysis.
    Provides detailed energy efficiency metrics and optimization insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Energy model parameters (based on typical quadcopter specifications)
        self.base_power = config.get('base_power', 80.0)            # Watts (hovering)
        self.thrust_efficiency = config.get('thrust_efficiency', 0.65)  # Propulsion efficiency
        self.avionics_power = config.get('avionics_power', 15.0)    # Watts (constant)
        self.communication_power = config.get('comm_power', 5.0)    # Watts (constant)
        
        # Drone specifications
        self.drone_mass = config.get('drone_mass', 1.5)            # kg
        self.max_thrust = config.get('max_thrust', 25.0)           # N
        self.max_velocity = config.get('max_velocity', 5.0)        # m/s
        
        # Energy calculation parameters
        self.sampling_rate = config.get('sampling_rate', 20.0)     # Hz
        self.power_model = config.get('power_model', 'quadratic')  # linear, quadratic, cubic
        
        # Analysis thresholds
        self.efficiency_targets = {
            'energy_per_meter': 100.0,      # J/m target efficiency
            'average_power': 120.0,         # W average power target
            'peak_power': 200.0,            # W peak power limit
            'power_variance': 50.0          # W² power variance target
        }
        
        # Energy component weights for detailed modeling
        self.energy_weights = config.get('energy_weights', {
            'thrust': 0.70,     # 70% thrust
            'avionics': 0.20,   # 20% avionics  
            'communication': 0.05,  # 5% communication
            'other': 0.05       # 5% other systems
        })
        
        self.logger.info("Energy Analyzer initialized")
        self.logger.info(f"Base power: {self.base_power}W, Drone mass: {self.drone_mass}kg")
        self.logger.info(f"Power model: {self.power_model}")
        self.logger.info(f"Energy efficiency target: {self.efficiency_targets['energy_per_meter']}J/m")
    
    def analyze_episodes(self, episode_results: List) -> Dict[str, Any]:
        """
        Analyze energy consumption across multiple episodes.
        
        Args:
            episode_results: List of episode results with energy data
            
        Returns:
            Comprehensive energy analysis
        """
        if not episode_results:
            return {'error': 'No episode data provided'}
        
        successful_episodes = [ep for ep in episode_results if getattr(ep, 'success', False)]
        
        if not successful_episodes:
            return {'error': 'No successful episodes for energy analysis'}
        
        self.logger.info(f"Analyzing energy consumption for {len(successful_episodes)} successful episodes")
        
        # Basic energy statistics
        energies = [ep.energy_consumption for ep in successful_episodes]
        times = [ep.flight_time for ep in successful_episodes]
        distances = [ep.path_length for ep in successful_episodes]
        
        # Energy efficiency metrics
        energy_per_meter = [e/d for e, d in zip(energies, distances) if d > 0]
        average_power = [e/t for e, t in zip(energies, times) if t > 0]
        
        analysis = {
            'energy_consumption': {
                'mean_energy': float(np.mean(energies)),
                'std_energy': float(np.std(energies)),
                'min_energy': float(np.min(energies)),
                'max_energy': float(np.max(energies)),
                'median_energy': float(np.median(energies)),
                'energy_range': float(np.max(energies) - np.min(energies))
            },
            'energy_efficiency': {
                'mean_energy_per_meter': float(np.mean(energy_per_meter)) if energy_per_meter else 0.0,
                'std_energy_per_meter': float(np.std(energy_per_meter)) if energy_per_meter else 0.0,
                'meets_efficiency_target': np.mean(energy_per_meter) <= self.efficiency_targets['energy_per_meter'] if energy_per_meter else False
            },
            'power_analysis': {
                'mean_power': float(np.mean(average_power)) if average_power else 0.0,
                'std_power': float(np.std(average_power)) if average_power else 0.0,
                'meets_power_target': np.mean(average_power) <= self.efficiency_targets['average_power'] if average_power else False
            }
        }
        
        # Advanced analysis for episodes with detailed trajectories
        episodes_with_trajectories = [ep for ep in successful_episodes if hasattr(ep, 'trajectory') and ep.trajectory]
        
        if episodes_with_trajectories:
            advanced_analysis = self._advanced_energy_analysis(episodes_with_trajectories)
            analysis.update(advanced_analysis)
        
        # Energy breakdown estimation
        energy_breakdown = self._estimate_energy_breakdown(successful_episodes)
        analysis['energy_breakdown'] = energy_breakdown
        
        # Efficiency recommendations
        recommendations = self._generate_efficiency_recommendations(analysis)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _advanced_energy_analysis(self, episodes_with_trajectories: List) -> Dict[str, Any]:
        """
        Advanced energy analysis using trajectory data.
        
        Args:
            episodes_with_trajectories: Episodes with detailed trajectory data
            
        Returns:
            Advanced energy analysis results
        """
        flight_phases = {
            'takeoff': [],
            'cruise': [],
            'maneuvering': [],
            'landing': []
        }
        
        for episode in episodes_with_trajectories:
            if not hasattr(episode, 'trajectory') or len(episode.trajectory) < 3:
                continue
                
            # Analyze flight phases
            trajectory = np.array(episode.trajectory)
            
            # Simple phase detection based on altitude and velocity
            altitudes = trajectory[:, 2]
            
            # Takeoff phase (increasing altitude)
            takeoff_mask = np.diff(altitudes, prepend=altitudes[0]) > 0.1
            takeoff_energy = self._estimate_phase_energy(trajectory[takeoff_mask], episode.flight_time / len(trajectory))
            
            # Cruise phase (stable altitude, moderate velocity)
            altitude_std = np.std(altitudes)
            cruise_mask = np.abs(altitudes - np.mean(altitudes)) < altitude_std * 0.5
            cruise_energy = self._estimate_phase_energy(trajectory[cruise_mask], episode.flight_time / len(trajectory))
            
            flight_phases['takeoff'].append(takeoff_energy)
            flight_phases['cruise'].append(cruise_energy)
        
        # Aggregate phase analysis
        phase_analysis = {}
        for phase, energies in flight_phases.items():
            if energies:
                phase_analysis[f'{phase}_energy'] = {
                    'mean': float(np.mean(energies)),
                    'std': float(np.std(energies)),
                    'samples': len(energies)
                }
        
        return {'flight_phase_analysis': phase_analysis}
    
    def _estimate_phase_energy(self, trajectory_segment: np.ndarray, time_per_step: float) -> float:
        """
        Estimate energy consumption for trajectory segment.
        
        Args:
            trajectory_segment: Trajectory positions
            time_per_step: Time per trajectory step
            
        Returns:
            Estimated energy consumption for segment
        """
        if len(trajectory_segment) < 2:
            return 0.0
        
        # Calculate velocities and accelerations
        velocities = np.diff(trajectory_segment, axis=0) / time_per_step
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Energy model based on velocity (simplified quadratic model)
        # P = P_base + k * v²
        k_factor = 20.0  # Empirical factor
        
        power_samples = self.base_power + k_factor * (velocity_magnitudes ** 2)
        total_energy = np.sum(power_samples) * time_per_step
        
        return float(total_energy)
    
    def _estimate_energy_breakdown(self, successful_episodes: List) -> Dict[str, float]:
        """
        Estimate energy consumption breakdown by system.
        
        Args:
            successful_episodes: List of successful episodes
            
        Returns:
            Energy breakdown dictionary
        """
        if not successful_episodes:
            return {}
        
        total_energies = [ep.energy_consumption for ep in successful_episodes]
        mean_total_energy = np.mean(total_energies)
        
        # Estimate breakdown based on weights
        breakdown = {}
        for component, weight in self.energy_weights.items():
            breakdown[f'{component}_energy'] = mean_total_energy * weight
        
        # Calculate percentages
        breakdown_percent = {
            f'{component}_percent': weight * 100
            for component, weight in self.energy_weights.items()
        }
        
        breakdown.update(breakdown_percent)
        breakdown['total_energy'] = mean_total_energy
        
        return breakdown
    
    def _generate_efficiency_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate energy efficiency recommendations.
        
        Args:
            analysis: Energy analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check energy per meter efficiency
        energy_efficiency = analysis.get('energy_efficiency', {})
        mean_energy_per_meter = energy_efficiency.get('mean_energy_per_meter', 0)
        
        if mean_energy_per_meter > self.efficiency_targets['energy_per_meter']:
            recommendations.append(
                f"Energy per meter ({mean_energy_per_meter:.1f} J/m) exceeds target "
                f"({self.efficiency_targets['energy_per_meter']} J/m). Consider optimizing flight paths."
            )
        
        # Check average power
        power_analysis = analysis.get('power_analysis', {})
        mean_power = power_analysis.get('mean_power', 0)
        
        if mean_power > self.efficiency_targets['average_power']:
            recommendations.append(
                f"Average power ({mean_power:.1f}W) exceeds target "
                f"({self.efficiency_targets['average_power']}W). Consider reducing aggressive maneuvers."
            )
        
        # Check power variance (smooth vs jerky flight)
        power_std = power_analysis.get('std_power', 0)
        if power_std > self.efficiency_targets['power_variance']:
            recommendations.append(
                "High power variance detected. Smoother flight trajectories could improve efficiency."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Energy consumption is within optimal ranges.")
        
        recommendations.append("Consider implementing regenerative braking during descents.")
        recommendations.append("Optimize payload weight distribution for better efficiency.")
        
        return recommendations
    
    def calculate_energy_savings(self, rl_energy: float, baseline_energy: float) -> Dict[str, float]:
        """
        Calculate detailed energy savings metrics.
        
        Args:
            rl_energy: RL method energy consumption  
            baseline_energy: Baseline energy consumption
            
        Returns:
            Energy savings analysis
        """
        if baseline_energy <= 0:
            return {'error': 'Invalid baseline energy'}
        
        absolute_savings = baseline_energy - rl_energy
        percentage_savings = (absolute_savings / baseline_energy) * 100
        
        # Calculate annual savings (assuming daily missions)
        missions_per_day = self.config.get('missions_per_day', 10)
        days_per_year = 365
        
        annual_energy_savings = absolute_savings * missions_per_day * days_per_year  # Joules
        annual_energy_savings_kwh = annual_energy_savings / 3.6e6  # Convert to kWh
        
        # Cost savings (assuming electricity cost)
        electricity_cost_per_kwh = self.config.get('electricity_cost', 0.12)  # $/kWh
        annual_cost_savings = annual_energy_savings_kwh * electricity_cost_per_kwh
        
        return {
            'absolute_savings_per_mission': absolute_savings,           # J
            'percentage_savings': percentage_savings,                   # %
            'annual_energy_savings': annual_energy_savings,            # J/year
            'annual_energy_savings_kwh': annual_energy_savings_kwh,    # kWh/year
            'annual_cost_savings': annual_cost_savings,                # $/year
            'efficiency_improvement_ratio': rl_energy / baseline_energy,
            'meets_25_percent_target': percentage_savings >= 25.0
        }
    
    def analyze_power_profile(self, timestamps: List[float], 
                            positions: List[Tuple[float, float, float]],
                            actions: List[List[float]]) -> PowerProfile:
        """
        Analyze detailed power consumption profile.
        
        Args:
            timestamps: Time series
            positions: Position trajectory
            actions: Action sequence
            
        Returns:
            Power consumption profile
        """
        if len(timestamps) < 2 or len(positions) < 2:
            return PowerProfile()
        
        power_profile = PowerProfile()
        
        # Calculate velocities and accelerations
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0.05
        positions_array = np.array(positions)
        
        velocities = np.diff(positions_array, axis=0) / dt
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0) / dt if len(velocities) > 1 else np.zeros((1, 3))
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Power consumption model
        for i in range(len(timestamps) - 1):
            timestamp = timestamps[i]
            
            # Get velocity and acceleration
            velocity = velocity_magnitudes[i] if i < len(velocity_magnitudes) else 0.0
            acceleration = acceleration_magnitudes[i] if i < len(acceleration_magnitudes) else 0.0
            
            # Power model: P = P_base + P_thrust(v) + P_maneuver(a)
            if self.power_model == 'quadratic':
                # Quadratic velocity dependency
                thrust_power = self.base_power + 15.0 * (velocity ** 2)
            elif self.power_model == 'cubic':
                # Cubic velocity dependency (more realistic at high speeds)
                thrust_power = self.base_power + 10.0 * velocity + 8.0 * (velocity ** 3)
            else:  # linear
                thrust_power = self.base_power + 20.0 * velocity
            
            # Acceleration penalty
            maneuver_power = 5.0 * (acceleration ** 2)
            
            # Total power
            total_power = thrust_power + maneuver_power + self.avionics_power + self.communication_power
            
            power_profile.add_sample(timestamp, total_power, velocity, acceleration)
        
        return power_profile
    
    def compare_energy_efficiency(self, method_results: Dict[str, List]) -> Dict[str, Any]:
        """
        Compare energy efficiency between multiple methods.
        
        Args:
            method_results: Dictionary of method_name -> list of episode results
            
        Returns:
            Energy efficiency comparison
        """
        comparison = {}
        
        for method_name, episodes in method_results.items():
            successful = [ep for ep in episodes if getattr(ep, 'success', False)]
            
            if successful:
                energies = [ep.energy_consumption for ep in successful]
                times = [ep.flight_time for ep in successful] 
                distances = [ep.path_length for ep in successful]
                
                # Calculate efficiency metrics
                energy_per_meter = [e/d for e, d in zip(energies, distances) if d > 0]
                average_power = [e/t for e, t in zip(energies, times) if t > 0]
                
                comparison[method_name] = {
                    'mean_energy': float(np.mean(energies)),
                    'energy_per_meter': float(np.mean(energy_per_meter)) if energy_per_meter else 0,
                    'average_power': float(np.mean(average_power)) if average_power else 0,
                    'energy_efficiency_score': self._calculate_efficiency_score(energies, distances, times)
                }
        
        # Rank methods by efficiency
        if len(comparison) > 1:
            sorted_methods = sorted(comparison.items(), key=lambda x: x[1]['energy_efficiency_score'], reverse=True)
            comparison['efficiency_ranking'] = [method for method, _ in sorted_methods]
        
        return comparison
    
    def _calculate_efficiency_score(self, energies: List[float], distances: List[float], 
                                  times: List[float]) -> float:
        """
        Calculate composite energy efficiency score.
        
        Args:
            energies: Energy consumption values
            distances: Path distances
            times: Flight times
            
        Returns:
            Efficiency score (higher is better)
        """
        if not energies or not distances or not times:
            return 0.0
        
        # Normalize metrics
        mean_energy = np.mean(energies)
        mean_distance = np.mean(distances) 
        mean_time = np.mean(times)
        
        if mean_energy <= 0 or mean_distance <= 0 or mean_time <= 0:
            return 0.0
        
        # Composite score: distance achieved per unit energy per unit time
        # Higher score = more distance with less energy in less time
        efficiency_score = (mean_distance / mean_energy) * (1.0 / mean_time) * 1000  # Scale factor
        
        return float(efficiency_score)
    
    def generate_energy_report(self, analysis: Dict[str, Any]) -> str:
        """
        Generate detailed energy analysis report.
        
        Args:
            analysis: Energy analysis results
            
        Returns:
            Formatted energy report
        """
        report_lines = []
        report_lines.append("ENERGY CONSUMPTION ANALYSIS")
        report_lines.append("=" * 40)
        report_lines.append("")
        
        # Energy consumption summary
        energy_stats = analysis.get('energy_consumption', {})
        report_lines.append("Energy Consumption Summary:")
        report_lines.append(f"  Mean: {energy_stats.get('mean_energy', 0):.0f} J")
        report_lines.append(f"  Std:  {energy_stats.get('std_energy', 0):.0f} J") 
        report_lines.append(f"  Range: {energy_stats.get('min_energy', 0):.0f} - {energy_stats.get('max_energy', 0):.0f} J")
        report_lines.append("")
        
        # Energy efficiency  
        efficiency_stats = analysis.get('energy_efficiency', {})
        report_lines.append("Energy Efficiency:")
        report_lines.append(f"  Energy per meter: {efficiency_stats.get('mean_energy_per_meter', 0):.1f} J/m")
        report_lines.append(f"  Target (100 J/m): {'✓ MET' if efficiency_stats.get('meets_efficiency_target', False) else '✗ NOT MET'}")
        report_lines.append("")
        
        # Power analysis
        power_stats = analysis.get('power_analysis', {})
        report_lines.append("Power Consumption:")
        report_lines.append(f"  Average power: {power_stats.get('mean_power', 0):.1f} W")
        report_lines.append(f"  Target (120 W): {'✓ MET' if power_stats.get('meets_power_target', False) else '✗ NOT MET'}")
        report_lines.append("")
        
        # Energy breakdown
        breakdown = analysis.get('energy_breakdown', {})
        if breakdown:
            report_lines.append("Energy Breakdown:")
            report_lines.append(f"  Thrust: {breakdown.get('thrust_energy', 0):.0f} J ({breakdown.get('thrust_percent', 0):.1f}%)")
            report_lines.append(f"  Avionics: {breakdown.get('avionics_energy', 0):.0f} J ({breakdown.get('avionics_percent', 0):.1f}%)")
            report_lines.append(f"  Communication: {breakdown.get('communication_energy', 0):.0f} J ({breakdown.get('communication_percent', 0):.1f}%)")
            report_lines.append("")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report_lines.append("Efficiency Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"  {i}. {rec}")
        
        return "\n".join(report_lines)
    
    def export_energy_data(self, analysis: Dict[str, Any], filename: str = "energy_analysis.json"):
        """Export detailed energy analysis to file."""
        output_path = Path(self.config.get('output_dir', '.')) / filename
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.logger.info(f"Energy analysis exported to {output_path}")
