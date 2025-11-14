import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

dataclass
class MetricSnapshot:

    timestamp: float
    value: Union[float, int, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

dataclass
class MetricSeries:

    name: str
    unit: str
    snapshots: List[MetricSnapshot] = field(default_factory=list)
    aggregation_method: str = 'mean'

    def add_value(self, value: Union[float, int, bool], timestamp: Optional[float] = None, metadata):

        snapshot = MetricSnapshot(
            timestamp=timestamp or time.time(),
            value=value,
            metadata=metadata
        )
        self.snapshots.append(snapshot)

    def get_aggregated_value(self) - float:

        if not self.snapshots:
            return 0.0

        values = [s.value for s in self.snapshots]

        if self.aggregation_method == 'mean':
            return float(np.mean(values))
        elif self.aggregation_method == 'sum':
            return float(np.sum(values))
        elif self.aggregation_method == 'max':
            return float(np.max(values))
        elif self.aggregation_method == 'min':
            return float(np.min(values))
        elif self.aggregation_method == 'last':
            return float(values[-1])
        else:
            return float(np.mean(values))

class MetricsCollector:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.collection_frequency = config.get('frequency', 20.0)
        self.max_history_length = config.get('max_history', 10000)
        self.auto_aggregation = config.get('auto_aggregation', True)

        self.metric_definitions = {
            'success_rate': MetricSeries('success_rate', '', aggregation_method='mean'),
            'energy_consumption': MetricSeries('energy_consumption', 'J', aggregation_method='sum'),
            'flight_time': MetricSeries('flight_time', 's', aggregation_method='last'),
            'collision_occurred': MetricSeries('collision_occurred', 'bool', aggregation_method='max'),
            'ate_error': MetricSeries('ate_error', 'm', aggregation_method='mean'),

            'distance_to_goal': MetricSeries('distance_to_goal', 'm', aggregation_method='last'),
            'path_efficiency': MetricSeries('path_efficiency', 'ratio', aggregation_method='mean'),
            'velocity_magnitude': MetricSeries('velocity_magnitude', 'm/s', aggregation_method='mean'),
            'acceleration_magnitude': MetricSeries('acceleration_magnitude', 'm/s²', aggregation_method='mean'),
            'jerk_magnitude': MetricSeries('jerk_magnitude', 'm/s³', aggregation_method='mean'),
            'control_smoothness': MetricSeries('control_smoothness', 'score', aggregation_method='mean'),
            'num_replans': MetricSeries('num_replans', 'count', aggregation_method='sum'),
            'clearance_violations': MetricSeries('clearance_violations', 'count', aggregation_method='sum'),

            'thrust_energy': MetricSeries('thrust_energy', 'J', aggregation_method='sum'),
            'avionics_energy': MetricSeries('avionics_energy', 'J', aggregation_method='sum'),
            'power_consumption': MetricSeries('power_consumption', 'W', aggregation_method='mean'),

            'waypoint_tracking_error': MetricSeries('waypoint_tracking_error', 'm', aggregation_method='mean'),
            'path_deviation': MetricSeries('path_deviation', 'm', aggregation_method='mean'),
            'planning_frequency': MetricSeries('planning_frequency', 'Hz', aggregation_method='mean')
        }

        self.current_episode_metrics: Dict[str, MetricSeries] = {}
        self.episode_start_time = 0.0
        self.is_collecting = False

        self.episode_summaries: List[Dict[str, float]] = []

        self.logger.info("Metrics Collector initialized")
        self.logger.info(f"Tracking {len(self.metric_definitions)} metric types")
        self.logger.info(f"Collection frequency: {self.collection_frequency}Hz")

    def start_episode_collection(self):

        self.current_episode_metrics = {
            name: MetricSeries(
                name=series.name,
                unit=series.unit,
                aggregation_method=series.aggregation_method
            )
            for name, series in self.metric_definitions.items()
        }

        self.episode_start_time = time.time()
        self.is_collecting = True

        self.logger.debug("Started episode metrics collection")

    def collect_step_metrics(self, step_data: Dict[str, Any]):

        if not self.is_collecting:
            return

        current_time = time.time()

        if 'energy_consumption' in step_data:
            self.current_episode_metrics['energy_consumption'].add_value(
                step_data['energy_consumption'], current_time
            )

        if 'power_consumption' in step_data:
            self.current_episode_metrics['power_consumption'].add_value(
                step_data['power_consumption'], current_time
            )

        if 'position' in step_data and 'goal_position' in step_data:
            distance = np.linalg.norm(
                np.array(step_data['position']) - np.array(step_data['goal_position'])
            )
            self.current_episode_metrics['distance_to_goal'].add_value(distance, current_time)

        if 'velocity' in step_data:
            vel_mag = np.linalg.norm(step_data['velocity'])
            self.current_episode_metrics['velocity_magnitude'].add_value(vel_mag, current_time)

        if 'acceleration' in step_data:
            accel_mag = np.linalg.norm(step_data['acceleration'])
            self.current_episode_metrics['acceleration_magnitude'].add_value(accel_mag, current_time)

        if 'ate_error' in step_data:
            self.current_episode_metrics['ate_error'].add_value(
                step_data['ate_error'], current_time
            )

        if 'collision' in step_data:
            self.current_episode_metrics['collision_occurred'].add_value(
                step_data['collision'], current_time
            )

        if 'min_clearance' in step_data:
            if step_data['min_clearance']  0.5:
                self.current_episode_metrics['clearance_violations'].add_value(1, current_time)

        if 'replan_triggered' in step_data and step_data['replan_triggered']:
            self.current_episode_metrics['num_replans'].add_value(1, current_time)

    def end_episode_collection(self, success: bool, final_position: Tuple[float, float, float],
                             goal_position: Tuple[float, float, float]) - Dict[str, float]:

        if not self.is_collecting:
            return {}

        self.is_collecting = False

        episode_duration = time.time() - self.episode_start_time
        final_distance = np.linalg.norm(np.array(final_position) - np.array(goal_position))

        self.current_episode_metrics['success_rate'].add_value(success)
        self.current_episode_metrics['flight_time'].add_value(episode_duration)

        episode_summary = {}
        for metric_name, metric_series in self.current_episode_metrics.items():
            aggregated_value = metric_series.get_aggregated_value()
            episode_summary[metric_name] = aggregated_value

        self.episode_summaries.append(episode_summary)

        if len(self.episode_summaries)  self.max_history_length:
            self.episode_summaries.pop(0)

        self.logger.debug(f"Episode metrics collected: {len(episode_summary)} metrics")

        return episode_summary

    def get_running_statistics(self, window_size: int = 20) - Dict[str, Dict[str, float]]:

        if not self.episode_summaries:
            return {}

        recent_episodes = self.episode_summaries[-window_size:]
        running_stats = {}

        for metric_name in self.metric_definitions.keys():
            values = [ep.get(metric_name, 0.0) for ep in recent_episodes]

            if values:
                running_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }

        return running_stats

    def export_metrics_json(self, filename: str = "metrics_detailed.json"):

        filepath = Path(self.config.get('output_dir', '.')) / filename

        export_data = {
            'metric_definitions': {
                name: {
                    'name': series.name,
                    'unit': series.unit,
                    'aggregation_method': series.aggregation_method
                }
                for name, series in self.metric_definitions.items()
            },
            'episode_summaries': self.episode_summaries,
            'collection_config': {
                'frequency': self.collection_frequency,
                'max_history': self.max_history_length
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Detailed metrics exported to {filepath}")
