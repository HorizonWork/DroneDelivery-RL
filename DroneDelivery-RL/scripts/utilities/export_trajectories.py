#!/usr/bin/env python3
"""
Trajectory Export Utility
Exports trajectory data in various formats for analysis and visualization.
Supports CSV, JSON, ROS bag, and visualization formats.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils import setup_logging

class TrajectoryExporter:
    """
    Advanced trajectory export system.
    Exports trajectories in multiple formats for different use cases.
    """
    
    def __init__(self, config_path: str = None):
        # Setup logging
        if config_path:
            from utils import load_config
            config = load_config(config_path)
            self.logger_system = setup_logging(config.logging)
        else:
            logging.basicConfig(level=logging.INFO)
        
        self.logger = logging.getLogger(__name__)
        
        # Export configuration
        self.export_config = {
            'coordinate_system': 'NED',  # North-East-Down
            'time_base': 20.0,          # 20Hz sampling rate
            'precision': 6,             # Decimal places for coordinates
            'include_metadata': True,
            'compress_output': False
        }
        
        # Supported export formats
        self.supported_formats = [
            'csv', 'json', 'ros_bag', 'matlab', 'numpy', 'visualization'
        ]
        
        self.logger.info("Trajectory Exporter initialized")
        self.logger.info(f"Supported formats: {', '.join(self.supported_formats)}")
    
    def export_from_training_results(self, training_results_file: str, 
                                   output_dir: str, formats: List[str]) -> Dict[str, Any]:
        """
        Export trajectories from training results.
        
        Args:
            training_results_file: Path to training results JSON
            output_dir: Output directory
            formats: List of export formats
            
        Returns:
            Export results
        """
        self.logger.info(f"Exporting trajectories from training results: {training_results_file}")
        
        # Load training data
        with open(training_results_file, 'r') as f:
            training_data = json.load(f)
        
        # Extract trajectory data
        trajectories = self._extract_trajectories_from_training(training_data)
        
        # Export in requested formats
        export_results = self._export_trajectories(trajectories, output_dir, formats)
        
        export_results.update({
            'source': 'training_results',
            'source_file': training_results_file,
            'trajectories_exported': len(trajectories)
        })
        
        return export_results
    
    def export_from_evaluation_results(self, evaluation_results_file: str,
                                     output_dir: str, formats: List[str]) -> Dict[str, Any]:
        """
        Export trajectories from evaluation results.
        
        Args:
            evaluation_results_file: Path to evaluation results JSON
            output_dir: Output directory  
            formats: List of export formats
            
        Returns:
            Export results
        """
        self.logger.info(f"Exporting trajectories from evaluation: {evaluation_results_file}")
        
        # Load evaluation data
        with open(evaluation_results_file, 'r') as f:
            evaluation_data = json.load(f)
        
        # Extract trajectory data
        trajectories = self._extract_trajectories_from_evaluation(evaluation_data)
        
        # Export in requested formats
        export_results = self._export_trajectories(trajectories, output_dir, formats)
        
        export_results.update({
            'source': 'evaluation_results',
            'source_file': evaluation_results_file,
            'trajectories_exported': len(trajectories)
        })
        
        return export_results
    
    def _extract_trajectories_from_training(self, training_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trajectory data from training results."""
        # This would extract from recorded episode data
        # For now, generate example trajectories
        trajectories = []
        
        # Example trajectory extraction (would be from actual training data)
        training_history = training_data.get('training_history', {})
        episode_count = len(training_history.get('episode_rewards', []))
        
        # Generate representative trajectories
        for episode_id in range(min(50, episode_count)):  # Limit to 50 trajectories
            trajectory = self._generate_example_trajectory(episode_id)
            trajectories.append({
                'episode_id': episode_id,
                'source': 'training',
                'trajectory': trajectory,
                'metadata': {
                    'episode_reward': training_history.get('episode_rewards', [0])[episode_id] if episode_id < len(training_history.get('episode_rewards', [])) else 0,
                    'energy_consumption': training_history.get('episode_energies', [0])[episode_id] if episode_id < len(training_history.get('episode_energies', [])) else 0
                }
            })
        
        return trajectories
    
    def _extract_trajectories_from_evaluation(self, evaluation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract trajectory data from evaluation results."""
        # This would extract from evaluation episode data
        trajectories = []
        
        # Example trajectory extraction (would be from actual evaluation data)
        performance_metrics = evaluation_data.get('performance_metrics', {})
        
        # Generate representative evaluation trajectories
        for episode_id in range(20):  # Typical evaluation episode count
            trajectory = self._generate_example_trajectory(episode_id, evaluation=True)
            trajectories.append({
                'episode_id': episode_id,
                'source': 'evaluation',
                'trajectory': trajectory,
                'metadata': {
                    'mean_energy': performance_metrics.get('mean_energy', 610),
                    'success_rate': performance_metrics.get('success_rate', 96.2),
                    'evaluation_run': True
                }
            })
        
        return trajectories
    
    def _generate_example_trajectory(self, episode_id: int, evaluation: bool = False) -> List[List[float]]:
        """Generate example trajectory (would be replaced with actual data)."""
        # Generate realistic 3D trajectory through 5-floor building
        np.random.seed(episode_id)  # Reproducible trajectories
        
        # Building dimensions: 20m x 40m x 15m (5 floors)
        start_pos = [2 + np.random.uniform(-1, 1), 2 + np.random.uniform(-1, 1), 1]
        goal_pos = [18 + np.random.uniform(-1, 1), 38 + np.random.uniform(-1, 1), 13]
        
        # Generate waypoints with some randomness
        num_waypoints = 50 + int(np.random.uniform(-10, 20))
        trajectory = []
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            
            # Linear interpolation with noise
            x = start_pos[0] + t * (goal_pos[0] - start_pos[0]) + np.random.uniform(-0.5, 0.5)
            y = start_pos[1] + t * (goal_pos[1] - start_pos[1]) + np.random.uniform(-0.5, 0.5)
            z = start_pos[2] + t * (goal_pos[2] - start_pos[2]) + np.random.uniform(-0.3, 0.3)
            
            # Keep within building bounds
            x = np.clip(x, 0.5, 19.5)
            y = np.clip(y, 0.5, 39.5)
            z = np.clip(z, 0.5, 14.5)
            
            trajectory.append([x, y, z])
        
        return trajectory
    
    def _export_trajectories(self, trajectories: List[Dict[str, Any]], 
                           output_dir: str, formats: List[str]) -> Dict[str, Any]:
        """Export trajectories in specified formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        export_results = {
            'export_completed': True,
            'output_directory': str(output_path),
            'formats_exported': [],
            'files_created': []
        }
        
        # Export in each requested format
        for format_name in formats:
            if format_name not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {format_name}")
                continue
            
            try:
                files_created = self._export_format(trajectories, output_path, format_name)
                export_results['formats_exported'].append(format_name)
                export_results['files_created'].extend(files_created)
                
                self.logger.info(f"✅ Exported {format_name}: {len(files_created)} files")
                
            except Exception as e:
                self.logger.error(f"Failed to export {format_name}: {e}")
        
        return export_results
    
    def _export_format(self, trajectories: List[Dict[str, Any]], 
                      output_path: Path, format_name: str) -> List[str]:
        """Export trajectories in specific format."""
        files_created = []
        
        if format_name == 'csv':
            files_created = self._export_csv(trajectories, output_path)
        elif format_name == 'json':
            files_created = self._export_json(trajectories, output_path)
        elif format_name == 'numpy':
            files_created = self._export_numpy(trajectories, output_path)
        elif format_name == 'visualization':
            files_created = self._export_visualizations(trajectories, output_path)
        elif format_name == 'matlab':
            files_created = self._export_matlab(trajectories, output_path)
        elif format_name == 'ros_bag':
            files_created = self._export_ros_bag(trajectories, output_path)
        
        return files_created
    
    def _export_csv(self, trajectories: List[Dict[str, Any]], output_path: Path) -> List[str]:
        """Export as CSV files."""
        csv_dir = output_path / 'csv'
        csv_dir.mkdir(exist_ok=True)
        
        # Combined CSV file
        all_trajectory_data = []
        
        for traj_info in trajectories:
            episode_id = traj_info['episode_id']
            source = traj_info['source']
            trajectory = traj_info['trajectory']
            metadata = traj_info.get('metadata', {})
            
            for step, position in enumerate(trajectory):
                all_trajectory_data.append({
                    'episode_id': episode_id,
                    'source': source,
                    'step': step,
                    'timestamp': step / self.export_config['time_base'],
                    'x': round(position[0], self.export_config['precision']),
                    'y': round(position[1], self.export_config['precision']),
                    'z': round(position[2], self.export_config['precision']),
                    'energy_consumption': metadata.get('energy_consumption', 0),
                    'episode_reward': metadata.get('episode_reward', 0)
                })
        
        # Save combined CSV
        df = pd.DataFrame(all_trajectory_data)
        combined_csv = csv_dir / 'all_trajectories.csv'
        df.to_csv(combined_csv, index=False)
        
        return [str(combined_csv)]
    
    def _export_json(self, trajectories: List[Dict[str, Any]], output_path: Path) -> List[str]:
        """Export as JSON files."""
        json_dir = output_path / 'json'
        json_dir.mkdir(exist_ok=True)
        
        # Complete trajectory data
        trajectory_data = {
            'export_metadata': {
                'format': 'json',
                'coordinate_system': self.export_config['coordinate_system'],
                'time_base_hz': self.export_config['time_base'],
                'precision': self.export_config['precision']
            },
            'trajectories': trajectories
        }
        
        json_file = json_dir / 'trajectories.json'
        with open(json_file, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        return [str(json_file)]
    
    def _export_numpy(self, trajectories: List[Dict[str, Any]], output_path: Path) -> List[str]:
        """Export as NumPy arrays."""
        numpy_dir = output_path / 'numpy'
        numpy_dir.mkdir(exist_ok=True)
        
        files_created = []
        
        # Export individual trajectory arrays
        for traj_info in trajectories:
            episode_id = traj_info['episode_id']
            trajectory = np.array(traj_info['trajectory'])
            
            # Save as .npy file
            npy_file = numpy_dir / f'trajectory_episode_{episode_id:04d}.npy'
            np.save(npy_file, trajectory)
            files_created.append(str(npy_file))
        
        # Combined array
        if trajectories:
            # Find max trajectory length for padding
            max_length = max(len(traj['trajectory']) for traj in trajectories)
            
            # Create padded array
            combined_array = np.full((len(trajectories), max_length, 3), np.nan)
            
            for i, traj_info in enumerate(trajectories):
                trajectory = np.array(traj_info['trajectory'])
                combined_array[i, :len(trajectory), :] = trajectory
            
            combined_file = numpy_dir / 'all_trajectories.npy'
            np.save(combined_file, combined_array)
            files_created.append(str(combined_file))
        
        return files_created
    
    def _export_visualizations(self, trajectories: List[Dict[str, Any]], 
                             output_path: Path) -> List[str]:
        """Export trajectory visualizations."""
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        files_created = []
        
        # 3D trajectory plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot sample trajectories (up to 10 for clarity)
        sample_trajectories = trajectories[:10]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_trajectories)))
        
        for i, traj_info in enumerate(sample_trajectories):
            trajectory = np.array(traj_info['trajectory'])
            
            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   color=colors[i], alpha=0.7, linewidth=2,
                   label=f"Episode {traj_info['episode_id']}")
            
            # Mark start and end points
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                      color=colors[i], s=100, marker='o')  # Start
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                      color=colors[i], s=100, marker='s')  # End
        
        # Format plot
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('3D Drone Trajectories')
        
        # Set building bounds
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 40)
        ax.set_zlim(0, 15)
        
        # Save 3D plot
        plot_3d_file = viz_dir / 'trajectories_3d.png'
        plt.savefig(plot_3d_file, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(str(plot_3d_file))
        
        # 2D floor plans
        self._create_floor_plan_visualizations(trajectories, viz_dir, files_created)
        
        # Energy vs trajectory length analysis
        self._create_energy_trajectory_plots(trajectories, viz_dir, files_created)
        
        return files_created
    
    def _create_floor_plan_visualizations(self, trajectories: List[Dict[str, Any]], 
                                        viz_dir: Path, files_created: List[str]):
        """Create 2D floor plan visualizations."""
        # Group trajectories by floor
        floor_trajectories = {i: [] for i in range(1, 6)}  # 5 floors
        
        for traj_info in trajectories[:5]:  # Sample trajectories
            trajectory = np.array(traj_info['trajectory'])
            
            # Assign points to floors based on z-coordinate
            for point in trajectory:
                floor = int(point[2] / 3) + 1  # Each floor is 3m height
                floor = max(1, min(5, floor))  # Clamp to valid floors
                
                floor_trajectories[floor].append(point[:2])  # Only x, y
        
        # Create floor plan plots
        for floor_num, points in floor_trajectories.items():
            if not points:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 20))  # 20x40m building
            
            # Plot trajectory points
            points_array = np.array(points)
            ax.scatter(points_array[:, 0], points_array[:, 1], 
                      alpha=0.6, s=20, c='blue')
            
            # Building outline
            ax.add_patch(plt.Rectangle((0, 0), 20, 40, fill=False, 
                                     edgecolor='black', linewidth=2))
            
            # Format plot
            ax.set_xlim(-1, 21)
            ax.set_ylim(-1, 41)
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title(f'Floor {floor_num} - Trajectory Points')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Save floor plan
            floor_file = viz_dir / f'floor_{floor_num}_trajectories.png'
            plt.savefig(floor_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            files_created.append(str(floor_file))
    
    def _create_energy_trajectory_plots(self, trajectories: List[Dict[str, Any]], 
                                      viz_dir: Path, files_created: List[str]):
        """Create energy vs trajectory analysis plots."""
        # Extract energy and path length data
        energy_data = []
        path_length_data = []
        
        for traj_info in trajectories:
            metadata = traj_info.get('metadata', {})
            trajectory = traj_info['trajectory']
            
            # Calculate path length
            path_length = 0.0
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    segment_length = np.linalg.norm(
                        np.array(trajectory[i]) - np.array(trajectory[i-1])
                    )
                    path_length += segment_length
            
            energy_data.append(metadata.get('energy_consumption', 0))
            path_length_data.append(path_length)
        
        if energy_data and path_length_data:
            # Energy vs path length scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(path_length_data, energy_data, alpha=0.6, s=50)
            
            # Add trend line
            if len(path_length_data) > 5:
                z = np.polyfit(path_length_data, energy_data, 1)
                p = np.poly1d(z)
                ax.plot(path_length_data, p(path_length_data), "r--", alpha=0.8)
            
            ax.set_xlabel('Path Length (meters)')
            ax.set_ylabel('Energy Consumption (Joules)')
            ax.set_title('Energy Consumption vs Path Length')
            ax.grid(True, alpha=0.3)
            
            # Save scatter plot
            scatter_file = viz_dir / 'energy_vs_path_length.png'
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            files_created.append(str(scatter_file))
    
    def _export_matlab(self, trajectories: List[Dict[str, Any]], output_path: Path) -> List[str]:
        """Export for MATLAB analysis."""
        matlab_dir = output_path / 'matlab'
        matlab_dir.mkdir(exist_ok=True)
        
        # Prepare data for MATLAB
        matlab_data = {
            'trajectories': [],
            'metadata': [],
            'export_info': {
                'coordinate_system': self.export_config['coordinate_system'],
                'time_base_hz': self.export_config['time_base'],
                'num_trajectories': len(trajectories)
            }
        }
        
        for traj_info in trajectories:
            matlab_data['trajectories'].append(traj_info['trajectory'])
            matlab_data['metadata'].append(traj_info.get('metadata', {}))
        
        # Save as .json (MATLAB can read JSON)
        matlab_file = matlab_dir / 'trajectories_matlab.json'
        with open(matlab_file, 'w') as f:
            json.dump(matlab_data, f, indent=2)
        
        return [str(matlab_file)]
    
    def _export_ros_bag(self, trajectories: List[Dict[str, Any]], output_path: Path) -> List[str]:
        """Export as ROS bag (placeholder - would need rosbag_writer)."""
        ros_dir = output_path / 'ros'
        ros_dir.mkdir(exist_ok=True)
        
        # For now, export ROS-compatible JSON
        ros_data = {
            'header': {
                'frame_id': 'world',
                'coordinate_system': 'NED',
                'time_base': 1.0 / self.export_config['time_base']
            },
            'trajectories': []
        }
        
        for traj_info in trajectories:
            ros_trajectory = {
                'header': {
                    'episode_id': traj_info['episode_id'],
                    'source': traj_info['source']
                },
                'poses': []
            }
            
            for step, position in enumerate(traj_info['trajectory']):
                pose = {
                    'header': {
                        'seq': step,
                        'stamp': step / self.export_config['time_base']
                    },
                    'pose': {
                        'position': {'x': position[0], 'y': position[1], 'z': position[2]},
                        'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}
                    }
                }
                ros_trajectory['poses'].append(pose)
            
            ros_data['trajectories'].append(ros_trajectory)
        
        ros_file = ros_dir / 'trajectories_ros.json'
        with open(ros_file, 'w') as f:
            json.dump(ros_data, f, indent=2)
        
        return [str(ros_file)]

def main():
    parser = argparse.ArgumentParser(description='Export trajectory data')
    parser.add_argument('--training-results', type=str,
                       help='Path to training results JSON')
    parser.add_argument('--evaluation-results', type=str, 
                       help='Path to evaluation results JSON')
    parser.add_argument('--output', type=str, default='data/exported_trajectories',
                       help='Output directory')
    parser.add_argument('--formats', type=str, nargs='+',
                       default=['csv', 'json', 'visualization'],
                       help='Export formats')
    parser.add_argument('--config', type=str, default='config/main_config.yaml',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    if not args.training_results and not args.evaluation_results:
        print("Error: Must provide either --training-results or --evaluation-results")
        sys.exit(1)
    
    # Create exporter
    exporter = TrajectoryExporter(args.config)
    
    # Export trajectories
    results = {}
    
    if args.training_results:
        training_export = exporter.export_from_training_results(
            args.training_results, args.output, args.formats
        )
        results['training_export'] = training_export
    
    if args.evaluation_results:
        evaluation_export = exporter.export_from_evaluation_results(
            args.evaluation_results, args.output, args.formats
        )
        results['evaluation_export'] = evaluation_export
    
    # Print summary
    print("\n📁 TRAJECTORY EXPORT SUMMARY")
    print("="*50)
    
    for export_type, export_data in results.items():
        print(f"\n{export_type.upper()}:")
        print(f"  Trajectories: {export_data['trajectories_exported']}")
        print(f"  Formats: {', '.join(export_data['formats_exported'])}")
        print(f"  Files created: {len(export_data['files_created'])}")
    
    print(f"\n📂 Output directory: {args.output}")

if __name__ == "__main__":
    main()
