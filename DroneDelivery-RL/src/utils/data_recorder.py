import numpy as np
import logging
import time
import json
import csv
import h5py
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import threading

dataclass
class FlightRecord:

    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    action: np.ndarray
    energy_consumption: float
    reward: float
    slam_pose: np.ndarray
    ate_error: float

class DataRecorder:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.output_dir = Path(config.get('output_dir', 'recorded_data'))
        self.recording_frequency = config.get('frequency', 20.0)
        self.buffer_size = config.get('buffer_size', 10000)
        self.auto_save_interval = config.get('auto_save_interval', 300.0)

        self.save_formats = config.get('formats', ['hdf5', 'csv'])
        self.compression = config.get('compression', True)

        self.is_recording = False
        self.recording_session_id = None

        self.flight_data_buffer: List[FlightRecord] = []
        self.system_metrics_buffer: List[Dict[str, Any]] = []

        self.save_thread: Optional[threading.Thread] = None
        self.save_lock = threading.Lock()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Data Recorder initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Recording frequency: {self.recording_frequency}Hz")

    def start_recording(self, session_id: Optional[str] = None):

        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return

        if session_id is None:
            timestamp = time.strftime("Ymd_HMS")
            session_id = f"flight_session_{timestamp}"

        self.recording_session_id = session_id
        self.is_recording = True

        self.flight_data_buffer.clear()
        self.system_metrics_buffer.clear()

        self.logger.info(f"Started recording session: {session_id}")

    def record_flight_data(self, position: np.ndarray, velocity: np.ndarray,
                          orientation: np.ndarray, angular_velocity: np.ndarray,
                          action: np.ndarray, energy_consumption: float,
                          reward: float, slam_pose: np.ndarray, ate_error: float):

        if not self.is_recording:
            return

        record = FlightRecord(
            timestamp=time.time(),
            position=position.copy(),
            velocity=velocity.copy(),
            orientation=orientation.copy(),
            angular_velocity=angular_velocity.copy(),
            action=action.copy(),
            energy_consumption=energy_consumption,
            reward=reward,
            slam_pose=slam_pose.copy(),
            ate_error=ate_error
        )

        with self.save_lock:
            self.flight_data_buffer.append(record)

            if len(self.flight_data_buffer) = self.buffer_size:
                self._async_save_buffer()

    def record_system_metrics(self, metrics: Dict[str, Any]):

        if not self.is_recording:
            return

        metric_record = {
            'timestamp': time.time(),
            metrics
        }

        with self.save_lock:
            self.system_metrics_buffer.append(metric_record)

    def stop_recording(self) - str:

        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return ""

        self.is_recording = False

        self._save_all_data()

        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()

        session_dir = self.output_dir / self.recording_session_id

        self.logger.info(f"Recording stopped. Data saved to: {session_dir}")
        self.logger.info(f"Recorded {len(self.flight_data_buffer)} flight samples, "
                        f"{len(self.system_metrics_buffer)} metric samples")

        return str(session_dir)

    def _async_save_buffer(self):

        if self.save_thread and self.save_thread.is_alive():
            return

        flight_data_copy = self.flight_data_buffer.copy()
        metrics_data_copy = self.system_metrics_buffer.copy()

        self.flight_data_buffer.clear()
        self.system_metrics_buffer.clear()

        self.save_thread = threading.Thread(
            target=self._save_data_batch,
            args=(flight_data_copy, metrics_data_copy),
            daemon=True
        )
        self.save_thread.start()

    def _save_data_batch(self, flight_data: List[FlightRecord], metrics_data: List[Dict[str, Any]]):

        if not flight_data and not metrics_data:
            return

        session_dir = self.output_dir / self.recording_session_id
        session_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("Ymd_HMS")

        if 'hdf5' in self.save_formats:
            self._save_hdf5(flight_data, metrics_data, session_dir / f"batch_{timestamp}.h5")

        if 'csv' in self.save_formats:
            self._save_csv(flight_data, session_dir / f"flight_data_{timestamp}.csv")

        if 'json' in self.save_formats:
            self._save_json(metrics_data, session_dir / f"metrics_{timestamp}.json")

    def _save_hdf5(self, flight_data: List[FlightRecord], metrics_data: List[Dict[str, Any]], filepath: Path):

        try:
            with h5py.File(filepath, 'w') as f:
                if flight_data:
                    flight_group = f.create_group('flight_data')

                    n_samples = len(flight_data)

                    flight_group.create_dataset('timestamps', (n_samples,), dtype='f8',
                                              data=[r.timestamp for r in flight_data])

                    flight_group.create_dataset('positions', (n_samples, 3), dtype='f4',
                                              data=np.array([r.position for r in flight_data]))

                    flight_group.create_dataset('velocities', (n_samples, 3), dtype='f4',
                                              data=np.array([r.velocity for r in flight_data]))

                    flight_group.create_dataset('orientations', (n_samples, 4), dtype='f4',
                                              data=np.array([r.orientation for r in flight_data]))

                    flight_group.create_dataset('actions', (n_samples, 4), dtype='f4',
                                              data=np.array([r.action for r in flight_data]))

                    flight_group.create_dataset('energy_consumption', (n_samples,), dtype='f4',
                                              data=[r.energy_consumption for r in flight_data])

                    flight_group.create_dataset('rewards', (n_samples,), dtype='f4',
                                              data=[r.reward for r in flight_data])

                    flight_group.create_dataset('ate_errors', (n_samples,), dtype='f4',
                                              data=[r.ate_error for r in flight_data])

                if metrics_data:
                    metrics_group = f.create_group('system_metrics')

                    timestamps = [m['timestamp'] for m in metrics_data]
                    metrics_group.create_dataset('timestamps', data=timestamps)

                    for key in metrics_data[0].keys():
                        if key != 'timestamp':
                            try:
                                values = [m.get(key, 0) for m in metrics_data]
                                metrics_group.create_dataset(key, data=values)
                            except Exception as e:
                                self.logger.warning(f"Failed to save metric {key}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to save HDF5 data: {e}")

    def _save_csv(self, flight_data: List[FlightRecord], filepath: Path):

        if not flight_data:
            return

        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'pos_x', 'pos_y', 'pos_z',
                    'vel_x', 'vel_y', 'vel_z',
                    'quat_x', 'quat_y', 'quat_z', 'quat_w',
                    'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                    'action_vx', 'action_vy', 'action_vz', 'action_yaw_rate',
                    'energy_consumption', 'reward', 'ate_error'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for record in flight_data:
                    writer.writerow({
                        'timestamp': record.timestamp,
                        'pos_x': record.position[0], 'pos_y': record.position[1], 'pos_z': record.position[2],
                        'vel_x': record.velocity[0], 'vel_y': record.velocity[1], 'vel_z': record.velocity[2],
                        'quat_x': record.orientation[0], 'quat_y': record.orientation[1],
                        'quat_z': record.orientation[2], 'quat_w': record.orientation[3],
                        'ang_vel_x': record.angular_velocity[0], 'ang_vel_y': record.angular_velocity[1],
                        'ang_vel_z': record.angular_velocity[2],
                        'action_vx': record.action[0], 'action_vy': record.action[1],
                        'action_vz': record.action[2], 'action_yaw_rate': record.action[3],
                        'energy_consumption': record.energy_consumption,
                        'reward': record.reward,
                        'ate_error': record.ate_error
                    })

        except Exception as e:
            self.logger.error(f"Failed to save CSV data: {e}")

    def _save_json(self, metrics_data: List[Dict[str, Any]], filepath: Path):

        try:
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save JSON data: {e}")

    def _save_all_data(self):

        if self.flight_data_buffer or self.system_metrics_buffer:
            self._save_data_batch(self.flight_data_buffer, self.system_metrics_buffer)

class FlightDataLogger:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.log_dir = Path(config.get('log_dir', 'flight_logs'))
        self.max_file_size = config.get('max_file_size_mb', 100)  1024  1024
        self.max_log_files = config.get('max_log_files', 10)

        self.current_log_file = None
        self.current_file_size = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Flight Data Logger initialized")

    def log_flight_event(self, event_type: str, data: Dict[str, Any]):

        log_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data
        }

        self._write_log_entry(log_entry)

    def _write_log_entry(self, entry: Dict[str, Any]):

        if (self.current_log_file is None or
            self.current_file_size  self.max_file_size):
            self._create_new_log_file()

        try:
            json_line = json.dumps(entry, default=str) + '\n'
            self.current_log_file.write(json_line)
            self.current_log_file.flush()

            self.current_file_size += len(json_line)

        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}")

    def _create_new_log_file(self):

        if self.current_log_file:
            self.current_log_file.close()

        timestamp = time.strftime("Ymd_HMS")
        log_filename = f"flight_log_{timestamp}.jsonl"
        log_path = self.log_dir / log_filename

        self.current_log_file = open(log_path, 'w')
        self.current_file_size = 0

        self._cleanup_old_logs()

        self.logger.info(f"Created new log file: {log_filename}")

    def _cleanup_old_logs(self):

        log_files = sorted(self.log_dir.glob("flight_log_.jsonl"))

        if len(log_files)  self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                try:
                    old_file.unlink()
                    self.logger.debug(f"Removed old log file: {old_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old log: {e}")

    def close(self):

        if self.current_log_file:
            self.current_log_file.close()
            self.current_log_file = None
