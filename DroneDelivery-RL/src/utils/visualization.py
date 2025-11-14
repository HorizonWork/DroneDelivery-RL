import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
import json

class SystemVisualizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 100)
        self.save_plots = config.get('save_plots', True)
        self.output_dir = Path(config.get('output_dir', 'plots'))

        self.colors = {
            'trajectory': '
            'goal': '
            'obstacle': '
            'collision': '
            'safe': '
            'warning': '
            'error': '
        }

        plt.style.use(config.get('plot_style', 'seaborn-v0_8'))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("System Visualizer initialized")

    def plot_building_map(self, building_config: Dict[str, Any],
                         obstacles: List[Dict[str, Any]] = None,
                         save_name: str = 'building_map.png') - plt.Figure:

        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)

        x_max = building_config.get('x_max', 20.0)
        y_max = building_config.get('y_max', 40.0)

        building_rect = patches.Rectangle(
            (0, 0), x_max, y_max,
            linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(building_rect)

        if obstacles:
            for obstacle in obstacles:
                obs_type = obstacle.get('type', 'box')

                if obs_type == 'box':
                    x, y = obstacle.get('position', [0, 0])[:2]
                    width = obstacle.get('size', [1, 1])[0]
                    height = obstacle.get('size', [1, 1])[1]

                    obs_rect = patches.Rectangle(
                        (x - width/2, y - height/2), width, height,
                        facecolor=self.colors['obstacle'], alpha=0.7
                    )
                    ax.add_patch(obs_rect)

                elif obs_type == 'circle':
                    x, y = obstacle.get('position', [0, 0])[:2]
                    radius = obstacle.get('radius', 0.5)

                    obs_circle = patches.Circle(
                        (x, y), radius,
                        facecolor=self.colors['obstacle'], alpha=0.7
                    )
                    ax.add_patch(obs_circle)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Building Map')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        if self.save_plots:
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig

    def plot_training_curves(self, training_data: Dict[str, List[float]],
                           save_name: str = 'training_curves.png') - plt.Figure:

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()

        plot_configs = [
            ('episode_rewards', 'Episode Reward', 'Episodes', 'Reward'),
            ('success_rates', 'Success Rate', 'Episodes', 'Success Rate ()'),
            ('energy_consumption', 'Energy Consumption', 'Episodes', 'Energy (J)'),
            ('policy_losses', 'Policy Loss', 'Updates', 'Loss')
        ]

        for i, (metric_name, title, xlabel, ylabel) in enumerate(plot_configs):
            if metric_name in training_data and training_data[metric_name]:
                data = training_data[metric_name]

                axes[i].plot(data, color=self.colors['trajectory'], linewidth=1.5)

                if len(data)  10:
                    window_size = min(50, len(data)
                    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
                    axes[i].plot(range(window_size-1, len(data)), moving_avg,
                               color=self.colors['goal'], linewidth=2, alpha=0.8)

                axes[i].set_title(title)
                axes[i].set_xlabel(xlabel)
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig

    def plot_evaluation_comparison(self, evaluation_results: Dict[str, Dict[str, float]],
                                 save_name: str = 'evaluation_comparison.png') - plt.Figure:

        methods = list(evaluation_results.keys())
        metrics = ['success_rate', 'mean_energy', 'mean_time', 'collision_rate']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [evaluation_results[method].get(metric, 0) for method in methods]

            bars = axes[i].bar(methods, values, color=self.colors['trajectory'], alpha=0.7)

            if metric in ['success_rate']:
                max_val = max(values) if values else 1
                for bar, val in zip(bars, values):
                    if val = 0.9  max_val:
                        bar.set_color(self.colors['safe'])
                    elif val = 0.7  max_val:
                        bar.set_color(self.colors['warning'])
                    else:
                        bar.set_color(self.colors['error'])

            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Value')

            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}', ha='center', va='bottom')

        plt.tight_layout()

        if self.save_plots:
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig

    def plot_system_metrics(self, metrics_data: Dict[str, Any],
                           save_name: str = 'system_metrics.png') - plt.Figure:

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=self.dpi)
        axes = axes.flatten()

        if 'cpu_usage' in metrics_data:
            axes[0].plot(metrics_data['cpu_usage'], color=self.colors['trajectory'])
            axes[0].set_title('CPU Usage')
            axes[0].set_ylabel('Usage ()')
            axes[0].grid(True, alpha=0.3)

        if 'memory_usage' in metrics_data:
            axes[1].plot(metrics_data['memory_usage'], color=self.colors['warning'])
            axes[1].set_title('Memory Usage')
            axes[1].set_ylabel('Usage (MB)')
            axes[1].grid(True, alpha=0.3)

        if 'processing_frequency' in metrics_data:
            axes[2].plot(metrics_data['processing_frequency'], color=self.colors['safe'])
            axes[2].set_title('Processing Frequency')
            axes[2].set_ylabel('Frequency (Hz)')
            axes[2].grid(True, alpha=0.3)

        if 'network_latency' in metrics_data:
            axes[3].plot(metrics_data['network_latency'], color=self.colors['error'])
            axes[3].set_title('Network Latency')
            axes[3].set_ylabel('Latency (ms)')
            axes[3].grid(True, alpha=0.3)

        if 'power_consumption' in metrics_data:
            axes[4].plot(metrics_data['power_consumption'], color=self.colors['obstacle'])
            axes[4].set_title('Power Consumption')
            axes[4].set_ylabel('Power (W)')
            axes[4].grid(True, alpha=0.3)

        if 'temperature' in metrics_data:
            axes[5].plot(metrics_data['temperature'], color=self.colors['collision'])
            axes[5].set_title('System Temperature')
            axes[5].set_ylabel('Temperature (C)')
            axes[5].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.save_plots:
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig

class TrajectoryPlotter:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.figure_size = config.get('figure_size', (12, 8))
        self.dpi = config.get('dpi', 100)
        self.output_dir = Path(config.get('output_dir', 'plots'))

        self.trajectory_colors = ['

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Trajectory Plotter initialized")

    def plot_3d_trajectory(self, trajectory: np.ndarray,
                          building_bounds: Dict[str, float],
                          obstacles: List[Dict[str, Any]] = None,
                          goal_position: Optional[np.ndarray] = None,
                          save_name: str = '3d_trajectory.png') - plt.Figure:

        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                color=self.trajectory_colors[0], linewidth=2, label='Trajectory')

        ax.scatter(trajectory[0], color='green', s=100, label='Start')
        ax.scatter(trajectory[-1], color='red', s=100, label='End')

        if goal_position is not None:
            ax.scatter(goal_position, color='gold', s=150, marker='', label='Goal')

        x_max = building_bounds.get('x_max', 20)
        y_max = building_bounds.get('y_max', 40)
        z_max = building_bounds.get('z_max', 15)

        corners = np.array([
            [0, 0, 0], [x_max, 0, 0], [x_max, y_max, 0], [0, y_max, 0],
            [0, 0, z_max], [x_max, 0, z_max], [x_max, y_max, z_max], [0, y_max, z_max]
        ])

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        for edge in edges:
            points = corners[edge]
            ax.plot3D(points.T, 'k-', alpha=0.3)

        for floor in range(1, 6):
            z = floor  3
            ax.plot([0, x_max, x_max, 0, 0], [0, 0, y_max, y_max, 0],
                   [z, z, z, z, z], 'k--', alpha=0.2)

        if obstacles:
            for obs in obstacles:
                if obs.get('type') == 'box':
                    pos = obs.get('position', [0, 0, 0])
                    size = obs.get('size', [1, 1, 1])

                    ax.scatter(pos, color='orange', s=50, alpha=0.7)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()

        max_range = max(x_max, y_max, z_max)
        ax.set_xlim(0, max_range)
        ax.set_ylim(0, max_range)
        ax.set_zlim(0, max_range)

        if self.config.get('save_plots', True):
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig

    def plot_trajectory_analysis(self, trajectory: np.ndarray,
                               velocities: np.ndarray = None,
                               accelerations: np.ndarray = None,
                               energy_consumption: np.ndarray = None,
                               save_name: str = 'trajectory_analysis.png') - plt.Figure:

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)

        time_axis = np.linspace(0, len(trajectory) / 20.0, len(trajectory))

        axes[0, 0].plot(time_axis, trajectory[:, 0], label='X', color=self.trajectory_colors[0])
        axes[0, 0].plot(time_axis, trajectory[:, 1], label='Y', color=self.trajectory_colors[1])
        axes[0, 0].plot(time_axis, trajectory[:, 2], label='Z', color=self.trajectory_colors[2])
        axes[0, 0].set_title('Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        if velocities is not None:
            vel_mag = np.linalg.norm(velocities, axis=1)
            axes[0, 1].plot(time_axis, vel_mag, color=self.trajectory_colors[0])
            axes[0, 1].plot(time_axis, velocities[:, 0], label='Vx', alpha=0.7)
            axes[0, 1].plot(time_axis, velocities[:, 1], label='Vy', alpha=0.7)
            axes[0, 1].plot(time_axis, velocities[:, 2], label='Vz', alpha=0.7)
            axes[0, 1].set_title('Velocity')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (m/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        if accelerations is not None:
            accel_mag = np.linalg.norm(accelerations, axis=1)
            axes[1, 0].plot(time_axis, accel_mag, color=self.trajectory_colors[3])
            axes[1, 0].set_title('Acceleration Magnitude')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Acceleration (m/s²)')
            axes[1, 0].grid(True, alpha=0.3)

        if energy_consumption is not None:
            axes[1, 1].plot(time_axis, energy_consumption, color=self.trajectory_colors[4])
            axes[1, 1].set_title('Energy Consumption')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Power (W)')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if self.config.get('save_plots', True):
            fig.savefig(self.output_dir / save_name, bbox_inches='tight')

        return fig
