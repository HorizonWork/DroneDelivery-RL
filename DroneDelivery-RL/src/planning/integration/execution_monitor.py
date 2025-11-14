import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from src.bridges.airsim_bridge import AirSimBridge
from src.environment.sensor_interface import SensorInterface

class ExecutionStatus(Enum):

    IDLE = "idle"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    EMERGENCY = "emergency"

class ReplanTrigger(Enum):

    OBSTACLE_PROXIMITY = "obstacle_proximity"
    PATH_BLOCKED = "path_blocked"
    POSITION_DEVIATION = "position_deviation"
    TIMEOUT = "timeout"
    SAFETY_VIOLATION = "safety_violation"
    MANUAL_REQUEST = "manual_request"

dataclass
class ExecutionMetrics:

    start_time: float = 0.0
    current_time: float = 0.0
    progress_percentage: float = 0.0
    distance_traveled: float = 0.0
    distance_remaining: float = 0.0
    average_speed: float = 0.0
    current_waypoint: int = 0
    total_waypoints: int = 0
    replans_triggered: int = 0
    safety_violations: int = 0

dataclass
class MonitoringEvent:

    timestamp: float
    event_type: str
    severity: str
    message: str
    position: Optional[Tuple[float, float, float]] = None
    data: Dict[str, Any] = field(default_factory=dict)

class ExecutionMonitor:

    def __init__(self, config: Dict[str, Any],
                    airsim_bridge: AirSimBridge,
                    sensor_interface: SensorInterface):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.airsim_bridge = airsim_bridge
        self.sensor_interface = sensor_interface

        self.monitoring_frequency = config.get('monitoring_frequency', 10.0)
        self.waypoint_tolerance = config.get('waypoint_tolerance', 1.0)
        self.max_position_deviation = config.get('max_deviation', 2.0)

        self.obstacle_proximity_threshold = config.get('obstacle_proximity', 1.5)
        self.blocked_path_threshold = config.get('blocked_threshold', 0.8)
        self.execution_timeout = config.get('execution_timeout', 300.0)

        self.safety_check_frequency = config.get('safety_frequency', 5.0)
        self.min_safety_clearance = config.get('min_safety_clearance', 0.5)

        self.status = ExecutionStatus.IDLE
        self.current_path: List[Tuple[float, float, float]] = []
        self.current_waypoint_index = 0
        self.metrics = ExecutionMetrics()

        self.events: deque = deque(maxlen=1000)
        self.event_callbacks: List[Callable[[MonitoringEvent], None]] = []

        self.replan_callbacks: List[Callable[[ReplanTrigger, Dict[str, Any]], None]] = []

        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitor_lock = threading.Lock()

        self.position_history: deque = deque(maxlen=100)

        self.logger.info("Execution Monitor initialized")
        self.logger.info(f"Monitoring frequency: {self.monitoring_frequency} Hz")
        self.logger.info(f"Safety thresholds - Proximity: {self.obstacle_proximity_threshold}m, "
                        f"Deviation: {self.max_position_deviation}m")

    def start_execution(self, path: List[Tuple[float, float, float]]):

        with self.monitor_lock:
            self.current_path = path.copy()
            self.current_waypoint_index = 0
            self.status = ExecutionStatus.EXECUTING

            self.metrics = ExecutionMetrics(
                start_time=time.time(),
                current_time=time.time(),
                total_waypoints=len(path)
            )

            self.position_history.clear()

            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()

        self._log_event("EXECUTION_STARTED", "INFO", f"Started monitoring path with {len(path)} waypoints")
        self.logger.info(f"Started execution monitoring: {len(path)} waypoints")

    def stop_execution(self):

        with self.monitor_lock:
            self.status = ExecutionStatus.IDLE
            self.is_monitoring = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

        self._log_event("EXECUTION_STOPPED", "INFO", "Execution monitoring stopped")
        self.logger.info("Execution monitoring stopped")

    def _monitoring_loop(self):

        self.logger.info("Execution monitoring loop started")

        monitor_period = 1.0 / self.monitoring_frequency
        last_safety_check = 0.0

        while self.is_monitoring:
            try:
                loop_start = time.time()

                current_position = self._get_current_position()
                current_obstacles = self._get_current_obstacles()

                if current_position:
                    self._update_metrics(current_position)

                    self._check_progress(current_position)

                    if time.time() - last_safety_check  1.0 / self.safety_check_frequency:
                        self._check_safety(current_position, current_obstacles)
                        last_safety_check = time.time()

                    self._check_replan_triggers(current_position, current_obstacles)

                elapsed = time.time() - loop_start
                sleep_time = max(0, monitor_period - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(monitor_period)

        self.logger.info("Execution monitoring loop stopped")

    def _update_metrics(self, current_position: Tuple[float, float, float]):

        current_time = time.time()

        with self.monitor_lock:
            self.metrics.current_time = current_time

            self.position_history.append((current_position, current_time))

            if len(self.position_history)  1:
                prev_pos, prev_time = self.position_history[-2]
                distance_step = np.linalg.norm(np.array(current_position) - np.array(prev_pos))
                self.metrics.distance_traveled += distance_step

                elapsed_time = current_time - self.metrics.start_time
                if elapsed_time  0:
                    self.metrics.average_speed = self.metrics.distance_traveled / elapsed_time

            if self.current_path:
                self._calculate_progress(current_position)

    def _calculate_progress(self, current_position: Tuple[float, float, float]):

        if not self.current_path:
            return

        current_pos_array = np.array(current_position)

        min_distance = float('inf')
        closest_waypoint_idx = 0

        for i, waypoint in enumerate(self.current_path):
            distance = np.linalg.norm(current_pos_array - np.array(waypoint))
            if distance  min_distance:
                min_distance = distance
                closest_waypoint_idx = i

        self.metrics.current_waypoint = closest_waypoint_idx

        if len(self.current_path)  1:
            self.metrics.progress_percentage = (closest_waypoint_idx / (len(self.current_path) - 1))  100

        remaining_distance = 0.0
        for i in range(closest_waypoint_idx, len(self.current_path) - 1):
            segment_distance = np.linalg.norm(
                np.array(self.current_path[i + 1]) - np.array(self.current_path[i])
            )
            remaining_distance += segment_distance

        if closest_waypoint_idx  len(self.current_path):
            distance_to_next = np.linalg.norm(
                current_pos_array - np.array(self.current_path[closest_waypoint_idx])
            )
            remaining_distance += distance_to_next

        self.metrics.distance_remaining = remaining_distance

    def _check_progress(self, current_position: Tuple[float, float, float]):

        if not self.current_path or self.current_waypoint_index = len(self.current_path):
            return

        current_waypoint = self.current_path[self.current_waypoint_index]
        distance_to_waypoint = np.linalg.norm(
            np.array(current_position) - np.array(current_waypoint)
        )

        if distance_to_waypoint = self.waypoint_tolerance:
            self.current_waypoint_index += 1

            self._log_event("WAYPOINT_REACHED", "INFO",
                           f"Waypoint {self.current_waypoint_index - 1} reached",
                           current_position)

            if self.current_waypoint_index = len(self.current_path):
                self.status = ExecutionStatus.COMPLETED
                self._log_event("PATH_COMPLETED", "INFO", "Path execution completed successfully")

        elif distance_to_waypoint  self.max_position_deviation:
            self._log_event("POSITION_DEVIATION", "WARNING",
                           f"Position deviation: {distance_to_waypoint:.2f}m  {self.max_position_deviation}m",
                           current_position)

            self._trigger_replan(ReplanTrigger.POSITION_DEVIATION, {
                'current_position': current_position,
                'target_waypoint': current_waypoint,
                'deviation_distance': distance_to_waypoint
            })

    def _check_safety(self, current_position: Tuple[float, float, float],
                     obstacles: List[Tuple[float, float, float]]):

        if not obstacles:
            return

        current_pos_array = np.array(current_position)
        min_clearance = float('inf')

        for obstacle_pos in obstacles:
            obstacle_array = np.array(obstacle_pos)
            distance = np.linalg.norm(current_pos_array - obstacle_array)
            min_clearance = min(min_clearance, distance)

        if min_clearance  self.min_safety_clearance:
            self.metrics.safety_violations += 1

            self._log_event("SAFETY_VIOLATION", "ERROR",
                           f"Safety clearance violated: {min_clearance:.2f}m  {self.min_safety_clearance}m",
                           current_position, {'min_clearance': min_clearance})

            self._trigger_replan(ReplanTrigger.SAFETY_VIOLATION, {
                'current_position': current_position,
                'min_clearance': min_clearance,
                'obstacles': obstacles
            })

    def _check_replan_triggers(self, current_position: Tuple[float, float, float],
                             obstacles: List[Tuple[float, float, float]]):

        current_time = time.time()

        elapsed_time = current_time - self.metrics.start_time
        if elapsed_time  self.execution_timeout:
            self._log_event("EXECUTION_TIMEOUT", "ERROR",
                           f"Execution timeout: {elapsed_time:.1f}s  {self.execution_timeout}s")

            self._trigger_replan(ReplanTrigger.TIMEOUT, {
                'elapsed_time': elapsed_time,
                'timeout_limit': self.execution_timeout
            })
            return

        if self._check_upcoming_path_blocked(current_position, obstacles):
            self._log_event("PATH_BLOCKED", "WARNING", "Upcoming path blocked by obstacles")

            self._trigger_replan(ReplanTrigger.PATH_BLOCKED, {
                'current_position': current_position,
                'blocking_obstacles': obstacles
            })

    def _check_upcoming_path_blocked(self, current_position: Tuple[float, float, float],
                                   obstacles: List[Tuple[float, float, float]]) - bool:

        if not self.current_path or not obstacles:
            return False

        lookahead_waypoints = 5
        end_idx = min(self.current_waypoint_index + lookahead_waypoints, len(self.current_path))

        upcoming_path = self.current_path[self.current_waypoint_index:end_idx]

        for obstacle_pos in obstacles:
            obstacle_array = np.array(obstacle_pos)

            for waypoint in upcoming_path:
                waypoint_array = np.array(waypoint)
                distance = np.linalg.norm(waypoint_array - obstacle_array)

                if distance  self.blocked_path_threshold:
                    return True

        return False

    def _trigger_replan(self, trigger: ReplanTrigger, context: Dict[str, Any]):

        self.metrics.replans_triggered += 1

        self._log_event("REPLAN_TRIGGERED", "WARNING",
                       f"Replanning triggered: {trigger.value}",
                       context.get('current_position'), context)

        for callback in self.replan_callbacks:
            try:
                callback(trigger, context)
            except Exception as e:
                self.logger.error(f"Replan callback error: {e}")

    def _log_event(self, event_type: str, severity: str, message: str,
                  position: Optional[Tuple[float, float, float]] = None,
                  data: Dict[str, Any] = None):

        event = MonitoringEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            message=message,
            position=position,
            data=data or {}
        )

        self.events.append(event)

        if severity == "INFO":
            self.logger.info(f"[{event_type}] {message}")
        elif severity == "WARNING":
            self.logger.warning(f"[{event_type}] {message}")
        elif severity == "ERROR":
            self.logger.error(f"[{event_type}] {message}")
        elif severity == "CRITICAL":
            self.logger.critical(f"[{event_type}] {message}")

        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")

    def _get_current_position(self) - Optional[Tuple[float, float, float]]:

        try:
            drone_state = self.airsim_bridge.get_drone_state()

            if drone_state is None:
                self.logger.warning("Failed to get drone state from AirSim")
                return None

            position = drone_state.position

            if position is None or len(position) != 3:
                self.logger.warning(f"Invalid position data: {position}")
                return None

            return tuple(position)

        except Exception as e:
            self.logger.error(f"Error getting current position: {e}")
            return None

    def _get_current_obstacles(self) - List[Tuple[float, float, float]]:

        try:
            sensor_data = self.sensor_interface.get_latest_data()

            if sensor_data is None:
                self.logger.debug("No sensor data available")
                return []

            obstacles = []

            if hasattr(sensor_data, 'occupancy_histogram') and sensor_data.occupancy_histogram is not None:
                current_pos = self._get_current_position()
                if current_pos is None:
                    return []

                histogram = sensor_data.occupancy_histogram
                sector_angle = 2  np.pi / 24

                for i, distance in enumerate(histogram):
                    if 0.5  distance  5.0:
                        angle = i  sector_angle

                        obstacle_x = current_pos[0] + distance  np.cos(angle)
                        obstacle_y = current_pos[1] + distance  np.sin(angle)
                        obstacle_z = current_pos[2]

                        obstacles.append((obstacle_x, obstacle_y, obstacle_z))

            elif hasattr(sensor_data, 'depth_features') and sensor_data.depth_features is not None:
                depth_features = sensor_data.depth_features

                if 'min_distance' in depth_features and depth_features['min_distance']  2.0:
                    current_pos = self._get_current_position()
                    if current_pos:
                        min_dist = depth_features['min_distance']
                        obstacles.append((
                            current_pos[0] + min_dist,
                            current_pos[1],
                            current_pos[2]
                        ))

            self.logger.debug(f"Detected {len(obstacles)} obstacles")
            return obstacles

        except Exception as e:
            self.logger.error(f"Error getting current obstacles: {e}")
            return []

    def update_path(self, new_path: List[Tuple[float, float, float]]):

        with self.monitor_lock:
            self.current_path = new_path.copy()
            if self.current_waypoint_index = len(new_path):
                self.current_waypoint_index = max(0, len(new_path) - 1)

            self.metrics.total_waypoints = len(new_path)

        self._log_event("PATH_UPDATED", "INFO", f"Path updated: {len(new_path)} waypoints")

    def add_replan_callback(self, callback: Callable[[ReplanTrigger, Dict[str, Any]], None]):

        self.replan_callbacks.append(callback)

    def add_event_callback(self, callback: Callable[[MonitoringEvent], None]):

        self.event_callbacks.append(callback)

    def get_execution_status(self) - ExecutionStatus:

        return self.status

    def get_metrics(self) - ExecutionMetrics:

        with self.monitor_lock:
            return self.metrics

    def get_recent_events(self, count: int = 20) - List[MonitoringEvent]:

        return list(self.events)[-count:]

    def get_statistics(self) - Dict[str, Any]:

        with self.monitor_lock:
            event_counts = {}
            for event in self.events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

            return {
                'execution_status': self.status.value,
                'metrics': {
                    'progress_percentage': self.metrics.progress_percentage,
                    'distance_traveled': self.metrics.distance_traveled,
                    'distance_remaining': self.metrics.distance_remaining,
                    'average_speed': self.metrics.average_speed,
                    'waypoint_progress': f"{self.metrics.current_waypoint}/{self.metrics.total_waypoints}",
                    'execution_time': self.metrics.current_time - self.metrics.start_time,
                    'replans_triggered': self.metrics.replans_triggered,
                    'safety_violations': self.metrics.safety_violations
                },
                'events': {
                    'total_events': len(self.events),
                    'event_counts': event_counts,
                    'recent_events': len([e for e in self.events if time.time() - e.timestamp  60])
                },
                'configuration': {
                    'monitoring_frequency': self.monitoring_frequency,
                    'waypoint_tolerance': self.waypoint_tolerance,
                    'max_deviation': self.max_position_deviation,
                    'obstacle_proximity_threshold': self.obstacle_proximity_threshold
                }
            }

    def emergency_stop(self):

        with self.monitor_lock:
            self.status = ExecutionStatus.EMERGENCY

        self._log_event("EMERGENCY_STOP", "CRITICAL", "Emergency stop triggered")
        self.logger.critical("Emergency stop triggered")

    def pause_execution(self):

        with self.monitor_lock:
            if self.status == ExecutionStatus.EXECUTING:
                self.status = ExecutionStatus.PAUSED

        self._log_event("EXECUTION_PAUSED", "INFO", "Execution paused")

    def resume_execution(self):

        with self.monitor_lock:
            if self.status == ExecutionStatus.PAUSED:
                self.status = ExecutionStatus.EXECUTING

        self._log_event("EXECUTION_RESUMED", "INFO", "Execution resumed")

    def reset(self):

        self.stop_execution()

        with self.monitor_lock:
            self.current_path.clear()
            self.current_waypoint_index = 0
            self.metrics = ExecutionMetrics()
            self.position_history.clear()
            self.events.clear()

        self.logger.info("Execution monitor reset")
