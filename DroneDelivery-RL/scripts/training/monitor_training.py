import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.utils.logger import setup_logging

class TrainingMonitor:

    def __init__(self, config_path: str = None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.monitor_config = {
            'update_interval': 30,
            'checkpoint_dir': 'models/checkpoints',
            'log_dir': 'logs',
            'tensorboard_dir': 'runs',
            'alert_thresholds': {
                'low_success_rate': 20.0,
                'high_memory_usage': 90.0,
                'training_stalled': 300,
                'loss_explosion': 100.0
            }
        }

        self.monitoring = False
        self.training_start_time = None
        self.last_checkpoint_time = None
        self.last_metrics = {}
        self.alerts_sent = []

        self.training_history = {
            'timestamps': [],
            'success_rates': [],
            'energy_consumptions': [],
            'policy_losses': [],
            'system_metrics': {
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_usage': []
            }
        }

        self.logger.info("Training Monitor initialized")

    def start_monitoring(self, experiment_name: str = None):

        self.logger.info("=== STARTING TRAINING MONITORING ===")
        self.monitoring = True
        self.training_start_time = time.time()

        if experiment_name:
            self.monitor_config['experiment_name'] = experiment_name

        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()

        self._run_interactive_dashboard()

    def _monitoring_loop(self):

        while self.monitoring:
            try:
                current_metrics = self._collect_current_metrics()

                self._update_training_history(current_metrics)

                self._check_alerts(current_metrics)

                self._display_status(current_metrics)

                time.sleep(self.monitor_config['update_interval'])

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _collect_current_metrics(self) - Dict[str, Any]:

        metrics = {
            'timestamp': time.time(),
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }

        checkpoint_info = self._get_latest_checkpoint()
        if checkpoint_info:
            metrics.update(checkpoint_info)

        system_metrics = self._get_system_metrics()
        metrics['system'] = system_metrics

        tensorboard_metrics = self._parse_tensorboard_logs()
        if tensorboard_metrics:
            metrics.update(tensorboard_metrics)

        if 'timestep' in metrics:
            total_timesteps = 5_000_000
            progress = (metrics['timestep'] / total_timesteps)  100
            metrics['progress_percent'] = progress

            if progress  0:
                elapsed_hours = metrics['training_time'] / 3600
                estimated_total_hours = elapsed_hours / (progress / 100)
                remaining_hours = estimated_total_hours - elapsed_hours
                metrics['estimated_remaining_hours'] = max(0, remaining_hours)

        return metrics

    def _get_latest_checkpoint(self) - Optional[Dict[str, Any]]:

        checkpoint_dir = Path(self.monitor_config['checkpoint_dir'])

        if not checkpoint_dir.exists():
            return None

        checkpoint_files = list(checkpoint_dir.glob('.pt'))
        if not checkpoint_files:
            return None

        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

        checkpoint_time = latest_checkpoint.stat().st_mtime
        if self.last_checkpoint_time and checkpoint_time = self.last_checkpoint_time:
            return self.last_metrics.get('checkpoint')

        self.last_checkpoint_time = checkpoint_time

        filename = latest_checkpoint.stem
        try:
            parts = filename.split('_')
            if len(parts) = 3 and parts[1] == 'step':
                timestep = int(parts[2])
                return {
                    'latest_checkpoint': str(latest_checkpoint),
                    'timestep': timestep,
                    'checkpoint_time': checkpoint_time
                }
        except (ValueError, IndexError):
            pass

        return {
            'latest_checkpoint': str(latest_checkpoint),
            'checkpoint_time': checkpoint_time
        }

    def _get_system_metrics(self) - Dict[str, Any]:

        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (10243)
            memory_total_gb = memory.total / (10243)

            disk = psutil.disk_usage('.')
            disk_free_gb = disk.free / (10243)

            gpu_info = self._get_gpu_usage()

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'disk_free_gb': disk_free_gb,
                'gpu_info': gpu_info
            }

        except Exception as e:
            self.logger.warning(f"System metrics collection failed: {e}")
            return {}

    def _get_gpu_usage(self) - Optional[Dict[str, Any]]:

        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None

                return {
                    'available': True,
                    'memory_allocated_gb': gpu_memory.get('allocated_bytes.all.current', 0) / (10243),
                    'memory_reserved_gb': gpu_memory.get('reserved_bytes.all.current', 0) / (10243),
                    'utilization_percent': gpu_utilization
                }
        except Exception:
            pass

        return {'available': False}

    def _parse_tensorboard_logs(self) - Optional[Dict[str, Any]]:

        return None

    def _update_training_history(self, metrics: Dict[str, Any]):

        self.training_history['timestamps'].append(metrics['timestamp'])

        if 'success_rate' in metrics:
            self.training_history['success_rates'].append(metrics['success_rate'])

        if 'energy_consumption' in metrics:
            self.training_history['energy_consumptions'].append(metrics['energy_consumption'])

        if 'policy_loss' in metrics:
            self.training_history['policy_losses'].append(metrics['policy_loss'])

        system = metrics.get('system', {})
        if 'cpu_percent' in system:
            self.training_history['system_metrics']['cpu_usage'].append(system['cpu_percent'])

        if 'memory_percent' in system:
            self.training_history['system_metrics']['memory_usage'].append(system['memory_percent'])

        max_history = 1000
        for key in self.training_history:
            if isinstance(self.training_history[key], list):
                self.training_history[key] = self.training_history[key][-max_history:]
            elif isinstance(self.training_history[key], dict):
                for subkey in self.training_history[key]:
                    if isinstance(self.training_history[key][subkey], list):
                        self.training_history[key][subkey] = self.training_history[key][subkey][-max_history:]

        self.last_metrics = metrics

    def _check_alerts(self, metrics: Dict[str, Any]):

        alerts = []

        if 'success_rate' in metrics:
            success_rate = metrics['success_rate']
            if success_rate  self.monitor_config['alert_thresholds']['low_success_rate']:
                alerts.append(f"  Low success rate: {success_rate:.1f}")

        system = metrics.get('system', {})
        if 'memory_percent' in system:
            memory_usage = system['memory_percent']
            if memory_usage  self.monitor_config['alert_thresholds']['high_memory_usage']:
                alerts.append(f"  High memory usage: {memory_usage:.1f}")

        if self.last_checkpoint_time:
            time_since_checkpoint = time.time() - self.last_checkpoint_time
            if time_since_checkpoint  self.monitor_config['alert_thresholds']['training_stalled']:
                alerts.append(f"  Training may be stalled (no checkpoint for {time_since_checkpoint/60:.1f} min)")

        if 'policy_loss' in metrics:
            policy_loss = metrics['policy_loss']
            if policy_loss  self.monitor_config['alert_thresholds']['loss_explosion']:
                alerts.append(f"  Policy loss explosion: {policy_loss:.2f}")

        for alert in alerts:
            if alert not in self.alerts_sent:
                self.logger.warning(alert)
                self.alerts_sent.append(alert)

        self.alerts_sent = self.alerts_sent[-10:]

    def _display_status(self, metrics: Dict[str, Any]):

        os.system('clear' if os.name == 'posix' else 'cls')

        print("="70)
        print(" DRONEDELIVERY-RL TRAINING MONITOR")
        print("="70)

        print(f"\n TRAINING PROGRESS")
        print("-"  30)

        if 'timestep' in metrics:
            print(f"Timestep: {metrics['timestep']:,} / 5,000,000")

        if 'progress_percent' in metrics:
            progress = metrics['progress_percent']
            progress_bar = ""  int(progress / 2) + ""  (50 - int(progress / 2))
            print(f"Progress: [{progress_bar}] {progress:.1f}")

        if 'estimated_remaining_hours' in metrics:
            remaining = metrics['estimated_remaining_hours']
            print(f"Estimated remaining: {remaining:.1f} hours")

        print(f"\n PERFORMANCE METRICS")
        print("-"  30)

        if 'success_rate' in metrics:
            print(f"Success Rate: {metrics['success_rate']:.1f}")

        if 'energy_consumption' in metrics:
            print(f"Energy Consumption: {metrics['energy_consumption']:.0f}J")

        if 'policy_loss' in metrics:
            print(f"Policy Loss: {metrics['policy_loss']:.4f}")

        print(f"\n SYSTEM RESOURCES")
        print("-"  30)

        system = metrics.get('system', {})
        if 'cpu_percent' in system:
            print(f"CPU Usage: {system['cpu_percent']:.1f}")

        if 'memory_percent' in system:
            print(f"Memory Usage: {system['memory_percent']:.1f} "
                  f"({system.get('memory_used_gb', 0):.1f}GB / {system.get('memory_total_gb', 0):.1f}GB)")

        if 'disk_free_gb' in system:
            print(f"Disk Free: {system['disk_free_gb']:.1f}GB")

        gpu_info = system.get('gpu_info', {})
        if gpu_info.get('available'):
            print(f"GPU Memory: {gpu_info.get('memory_allocated_gb', 0):.1f}GB allocated")

        print(f"\n LATEST CHECKPOINT")
        print("-"  30)

        if 'latest_checkpoint' in metrics:
            checkpoint_path = Path(metrics['latest_checkpoint'])
            checkpoint_time = datetime.fromtimestamp(metrics['checkpoint_time'])
            print(f"File: {checkpoint_path.name}")
            print(f"Time: {checkpoint_time.strftime('Y-m-d H:M:S')}")

        if 'training_time' in metrics:
            training_hours = metrics['training_time'] / 3600
            print(f"\n  Training Time: {training_hours:.1f} hours")

        print(f"\nLast Update: {datetime.now().strftime('H:M:S')}")
        print("="70)
        print("Press Ctrl+C to stop monitoring")

    def _run_interactive_dashboard(self):

        try:
            while self.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Stopping monitoring...")
            self.monitoring = False

    def save_monitoring_report(self, output_path: str):

        report = {
            'monitoring_summary': {
                'start_time': self.training_start_time,
                'end_time': time.time(),
                'total_monitoring_time_hours': (time.time() - self.training_start_time) / 3600 if self.training_start_time else 0,
                'last_metrics': self.last_metrics,
                'alerts_generated': len(set(self.alerts_sent))
            },
            'training_history': self.training_history,
            'configuration': self.monitor_config
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Monitoring report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Monitor PPO training progress')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name to monitor')
    parser.add_argument('--interval', type=int, default=30,
                       help='Update interval in seconds')
    parser.add_argument('--output', type=str, default='results/monitoring_report.json',
                       help='Output file for monitoring report')

    args = parser.parse_args()

    monitor = TrainingMonitor()
    monitor.monitor_config['update_interval'] = args.interval

    try:
        monitor.start_monitoring(args.experiment)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.save_monitoring_report(args.output)
        print(f"\n Monitoring report saved to {args.output}")

if __name__ == "__main__":
    main()
