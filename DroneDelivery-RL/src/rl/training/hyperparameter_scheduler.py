import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

class ScheduleType(Enum):

    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    ADAPTIVE = "adaptive"

dataclass
class HyperparameterSchedule:

    param_name: str
    schedule_type: ScheduleType
    initial_value: float
    final_value: float
    decay_steps: int
    step_size: Optional[float] = None
    adaptation_metric: Optional[str] = None
    adaptation_threshold: Optional[float] = None

class HyperparameterScheduler:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.schedules: Dict[str, HyperparameterSchedule] = {}
        self._initialize_default_schedules(config.get('schedules', {}))

        self.current_timestep = 0
        self.performance_history = {}

        self.adaptation_window = config.get('adaptation_window', 1000)
        self.min_lr = config.get('min_learning_rate', 1e-6)
        self.max_lr = config.get('max_learning_rate', 1e-2)

        self.logger.info("Hyperparameter Scheduler initialized")
        self.logger.info(f"Schedules: {list(self.schedules.keys())}")

    def _initialize_default_schedules(self, schedule_configs: Dict[str, Any]):

        lr_config = schedule_configs.get('learning_rate', {})
        self.schedules['learning_rate'] = HyperparameterSchedule(
            param_name='learning_rate',
            schedule_type=ScheduleType(lr_config.get('type', 'exponential')),
            initial_value=lr_config.get('initial', 3e-4),
            final_value=lr_config.get('final', 1e-5),
            decay_steps=lr_config.get('decay_steps', 3_000_000),
            adaptation_metric='policy_loss',
            adaptation_threshold=0.1
        )

        noise_config = schedule_configs.get('exploration_noise', {})
        self.schedules['exploration_noise'] = HyperparameterSchedule(
            param_name='exploration_noise',
            schedule_type=ScheduleType(noise_config.get('type', 'exponential')),
            initial_value=noise_config.get('initial', 0.6),
            final_value=noise_config.get('final', 0.1),
            decay_steps=noise_config.get('decay_steps', 2_000_000)
        )

        clip_config = schedule_configs.get('clip_range', {})
        self.schedules['clip_range'] = HyperparameterSchedule(
            param_name='clip_range',
            schedule_type=ScheduleType(clip_config.get('type', 'linear')),
            initial_value=clip_config.get('initial', 0.2),
            final_value=clip_config.get('final', 0.1),
            decay_steps=clip_config.get('decay_steps', 4_000_000)
        )

        entropy_config = schedule_configs.get('entropy_coefficient', {})
        self.schedules['entropy_coefficient'] = HyperparameterSchedule(
            param_name='entropy_coefficient',
            schedule_type=ScheduleType(entropy_config.get('type', 'exponential')),
            initial_value=entropy_config.get('initial', 0.01),
            final_value=entropy_config.get('final', 0.001),
            decay_steps=entropy_config.get('decay_steps', 3_000_000)
        )

    def update_timestep(self, timestep: int, performance_metrics: Dict[str, float] = None):

        self.current_timestep = timestep

        if performance_metrics:
            for metric_name, value in performance_metrics.items():
                if metric_name not in self.performance_history:
                    self.performance_history[metric_name] = []

                self.performance_history[metric_name].append({
                    'timestep': timestep,
                    'value': value
                })

                if len(self.performance_history[metric_name])  self.adaptation_window:
                    self.performance_history[metric_name].pop(0)

    def get_current_hyperparameters(self) - Dict[str, float]:

        current_params = {}

        for param_name, schedule in self.schedules.items():
            current_value = self._calculate_scheduled_value(schedule)
            current_params[param_name] = current_value

        return current_params

    def _calculate_scheduled_value(self, schedule: HyperparameterSchedule) - float:

        if schedule.schedule_type == ScheduleType.CONSTANT:
            return schedule.initial_value

        progress = min(1.0, self.current_timestep / schedule.decay_steps)

        if schedule.schedule_type == ScheduleType.LINEAR:
            return schedule.initial_value + progress  (schedule.final_value - schedule.initial_value)

        elif schedule.schedule_type == ScheduleType.EXPONENTIAL:
            decay_rate = np.log(schedule.final_value / schedule.initial_value)
            return schedule.initial_value  np.exp(decay_rate  progress)

        elif schedule.schedule_type == ScheduleType.COSINE:
            return schedule.final_value + 0.5  (schedule.initial_value - schedule.final_value)  (
                1 + np.cos(np.pi  progress)
            )

        elif schedule.schedule_type == ScheduleType.STEP:
            if schedule.step_size:
                steps = int(self.current_timestep / schedule.step_size)
                decay_factor = 0.9  steps
                return schedule.initial_value  decay_factor
            return schedule.initial_value

        elif schedule.schedule_type == ScheduleType.ADAPTIVE:
            return self._adaptive_adjustment(schedule)

        return schedule.initial_value

    def _adaptive_adjustment(self, schedule: HyperparameterSchedule) - float:

        if not schedule.adaptation_metric or schedule.adaptation_metric not in self.performance_history:
            progress = min(1.0, self.current_timestep / schedule.decay_steps)
            decay_rate = np.log(schedule.final_value / schedule.initial_value)
            return schedule.initial_value  np.exp(decay_rate  progress)

        recent_performance = self.performance_history[schedule.adaptation_metric][-10:]

        if len(recent_performance)  5:
            return schedule.initial_value

        values = [entry['value'] for entry in recent_performance]
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]

        current_value = self._calculate_scheduled_value_non_adaptive(schedule)

        if schedule.param_name == 'learning_rate':
            if schedule.adaptation_metric == 'policy_loss':
                if trend_slope  0.01:
                    adapted_value = current_value  0.9
                elif abs(trend_slope)  0.001 and values[-1]  schedule.adaptation_threshold:
                    adapted_value = min(current_value  1.05, schedule.initial_value)
                else:
                    adapted_value = current_value
            else:
                adapted_value = current_value

        else:
            adapted_value = current_value

        adapted_value = np.clip(adapted_value,
                              min(schedule.initial_value, schedule.final_value),
                              max(schedule.initial_value, schedule.final_value))

        return float(adapted_value)

    def _calculate_scheduled_value_non_adaptive(self, schedule: HyperparameterSchedule) - float:

        progress = min(1.0, self.current_timestep / schedule.decay_steps)

        if schedule.schedule_type == ScheduleType.EXPONENTIAL or schedule.schedule_type == ScheduleType.ADAPTIVE:
            decay_rate = np.log(schedule.final_value / schedule.initial_value)
            return schedule.initial_value  np.exp(decay_rate  progress)
        else:
            return schedule.initial_value + progress  (schedule.final_value - schedule.initial_value)

    def add_custom_schedule(self, param_name: str, schedule_config: Dict[str, Any]):

        self.schedules[param_name] = HyperparameterSchedule(
            param_name=param_name,
            schedule_type=ScheduleType(schedule_config.get('type', 'linear')),
            initial_value=schedule_config['initial'],
            final_value=schedule_config['final'],
            decay_steps=schedule_config.get('decay_steps', 1_000_000),
            step_size=schedule_config.get('step_size'),
            adaptation_metric=schedule_config.get('adaptation_metric'),
            adaptation_threshold=schedule_config.get('adaptation_threshold')
        )

        self.logger.info(f"Added custom schedule for {param_name}")

    def get_schedule_info(self) - Dict[str, Dict[str, Any]]:

        schedule_info = {}

        for param_name, schedule in self.schedules.items():
            current_value = self._calculate_scheduled_value(schedule)
            progress = min(1.0, self.current_timestep / schedule.decay_steps)

            schedule_info[param_name] = {
                'schedule_type': schedule.schedule_type.value,
                'initial_value': schedule.initial_value,
                'final_value': schedule.final_value,
                'current_value': current_value,
                'progress': progress  100,
                'decay_steps': schedule.decay_steps
            }

        return schedule_info

    def export_schedule_history(self, filepath: str):

        history_data = {
            'current_timestep': self.current_timestep,
            'schedules': {
                name: {
                    'type': schedule.schedule_type.value,
                    'initial': schedule.initial_value,
                    'final': schedule.final_value,
                    'current': self._calculate_scheduled_value(schedule)
                }
                for name, schedule in self.schedules.items()
            },
            'performance_history': self.performance_history
        }

        import json
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)

        self.logger.info(f"Schedule history exported to {filepath}")
