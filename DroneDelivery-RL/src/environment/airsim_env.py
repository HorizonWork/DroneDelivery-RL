import gymnasium as gym
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

from src.environment.observation_space import ObservationSpace
from src.environment.action_space import ActionSpace
from src.environment.reward_function import RewardFunction
from src.environment.target_manager import TargetManager
from src.environment.curriculum_manager import CurriculumManager
from src.environment.sensor_interface import SensorInterface
from src.environment.drone_controller import DroneController
from src.environment.world_builder import WorldBuilder

from src.bridges.airsim_bridge import AirSimBridge
from src.bridges.slam_bridge import SLAMBridge
from src.bridges.sensor_bridge import SensorBridge

dataclass
class EpisodeInfo:

    episode_id: int = 0
    step_count: int = 0
    start_time: float = 0.0
    goal_reached: bool = False
    collision_occurred: bool = False
    timeout: bool = False
    energy_consumed: float = 0.0
    distance_traveled: float = 0.0

class AirSimEnvironment(gym.Env):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.observation_handler = ObservationSpace(config)
        self.action_handler = ActionSpace(config)
        self.reward_function = RewardFunction(config)
        self.target_manager = TargetManager(config)
        self.curriculum_manager = CurriculumManager(config)
        self.sensor_interface = SensorInterface(config)
        self.drone_controller = DroneController(config)

        self.world_builder = WorldBuilder(config)
        self.logger.info(" WorldBuilder created")
        self.logger.info("="50)
        self.logger.info("STARTING BRIDGE INITIALIZATION")
        self.logger.info("="50)

        import airsim
        self.logger.info("Creating test AirSim client...")
        try:
            test_client = airsim.MultirotorClient()
            self.logger.info("Confirming AirSim connection...")
            test_client.confirmConnection()
            self.logger.info("Pinging AirSim...")
            test_client.ping()
            self.logger.info(" AirSim health check PASSED")
        except Exception as e:
            self.logger.error(f" AirSim health check FAILED: {e}")
            raise

        self.logger.info("Creating AirSimBridge...")
        self.airsim_bridge = AirSimBridge(config.get('airsim', {}))
        if not self.airsim_bridge.connect():
            raise RuntimeError("AirSimBridge connection failed")
        self.logger.info("[OK] AirSimBridge connected")

        self.logger.info(" AirSimBridge created")
        self.logger.info("Creating SLAMBridge...")
        self.slam_bridge = SLAMBridge(config.get('slam', {}))
        self.logger.info(" SLAMBridge created")

        self.logger.info("Creating SensorBridge...")
        self.sensor_bridge = SensorBridge(config.get('sensor', {}))
        self.logger.info(" SensorBridge created")

        self.logger.info("="50)
        self.logger.info("ALL BRIDGES INITIALIZED SUCCESSFULLY")
        self.logger.info("="50)

        self.airsim_bridge = AirSimBridge(config.get('airsim', {}))
        self.slam_bridge = SLAMBridge(config.get('slam', {}))
        self.sensor_bridge = SensorBridge(config.get('sensor', {}))

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.action_handler.action_low,
            high=self.action_handler.action_high,
            dtype=np.float32
        )

        self.episode_info = EpisodeInfo()
        self.max_episode_steps = config.get('max_episode_steps', 6000)
        self.max_episode_time = config.get('max_episode_time', 300.0)

        self.current_state: Dict[str, Any] = {}
        self.previous_position: Optional[Tuple[float, float, float]] = None

        self.update_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.update_frequency = config.get('update_frequency', 20.0)

        self.step_times: List[float] = []
        self.max_step_time_history = 1000

        self.world_built = False

        self.logger.info("AirSim Environment initialized")
        self.logger.info(f"Observation space: {self.observation_space.shape}")
        self.logger.info(f"Action space: {self.action_space.shape}")
        self.logger.info(f"Max episode steps: {self.max_episode_steps}")

    def reset(self, seed: Optional[int] = None,
             options: Optional[Dict[str, Any]] = None) - Tuple[np.ndarray, Dict[str, Any]]:

        super().reset(seed=seed)

        start_time = time.time()
        self.logger.info(f"Resetting environment (episode {self.episode_info.episode_id + 1})")

        self.episode_info = EpisodeInfo(
            episode_id=self.episode_info.episode_id + 1,
            start_time=start_time
        )

        if not self.airsim_bridge.is_connected:
            if not self.airsim_bridge.connect():
                raise RuntimeError("Failed to connect to AirSim")

        self.airsim_bridge.reset_drone()

        self._start_subsystems()

        current_phase = self.curriculum_manager.get_current_phase()
        self.target_manager.set_curriculum_phase(current_phase)

        target = self.target_manager.select_target()
        self.logger.info(f"Selected target: {target.name} at {target.position}")

        self.reward_function.reset_episode()
        self.target_manager.reset_episode()
        self.sensor_interface.reset()
        self.drone_controller.reset()

        if not self.world_built:
            try:
                self.logger.info("Building world (first episode only)...")
                self.world_builder.update_world_state()
                self.world_built = True
                self.logger.info(" World built successfully")
                time.sleep(2.0)
            except Exception as e:
                self.logger.warning(f"Could not update world state: {e}")
                self.logger.info("Continuing without world update...")
        time.sleep(0.5)

        observation = self._get_observation()
        info = self._get_info()

        self.current_state = self._build_state_dict()
        self.previous_position = self.current_state.get('position')

        reset_time = time.time() - start_time
        self.logger.info(f"Environment reset completed in {reset_time:.2f}s")

        return observation, info

    def step(self, action: np.ndarray) - Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        step_start_time = time.time()

        if not self.action_handler.validate_action(action):
            self.logger.warning("Invalid action received, clipping to valid range")

        safe_action = self.action_handler.apply_safety_constraints(action, self.current_state)

        self.action_handler.record_action(safe_action)

        previous_state = self.current_state.copy()

        self.drone_controller.execute_action(safe_action)

        time.sleep(1.0 / self.update_frequency)

        self.current_state = self._build_state_dict()

        observation = self._get_observation()

        info = self._get_info()
        reward = self.reward_function.compute_reward(
            previous_state, safe_action, self.current_state, info
        )

        terminated = self._check_terminated()
        truncated = self._check_truncated()

        self.episode_info.step_count += 1
        self.episode_info.energy_consumed += self._calculate_step_energy(safe_action)

        if self.previous_position is not None:
            current_pos = self.current_state.get('position', (0, 0, 0))
            distance_step = np.linalg.norm(np.array(current_pos) - np.array(self.previous_position))
            self.episode_info.distance_traveled += distance_step
            self.previous_position = current_pos

        if terminated or truncated:
            self._finalize_episode()
            self.curriculum_manager.update_progress(
                success=self.episode_info.goal_reached,
                collision=self.episode_info.collision_occurred
            )

        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        if len(self.step_times)  self.max_step_time_history:
            self.step_times.pop(0)

        return observation, float(reward), terminated, truncated, info

    def _start_subsystems(self):

        try:
            if not self.slam_bridge.start_slam():
                self.logger.warning("Failed to start SLAM system")

            self.sensor_bridge.start_processing()

            self.sensor_interface.start()

            if not self.is_running:
                self.is_running = True
                self.update_thread = threading.Thread(target=self._update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()

        except Exception as e:
            self.logger.error(f"Failed to start subsystems: {e}")

    def _stop_subsystems(self):

        try:
            self.is_running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)

            self.sensor_interface.stop()
            self.sensor_bridge.stop_processing()
            self.slam_bridge.stop_slam()

        except Exception as e:
            self.logger.error(f"Error stopping subsystems: {e}")

    def _update_loop(self):

        self.logger.info("Starting real-time update loop")

        while self.is_running:
            try:
                sensor_data = self.airsim_bridge.get_sensor_data()

                if sensor_data.stereo_left is not None and sensor_data.stereo_right is not None:
                    processed_reading = self.sensor_bridge.process_sensor_data(
                        sensor_data.stereo_left,
                        sensor_data.stereo_right,
                        sensor_data.depth_image,
                        sensor_data.imu_data,
                        sensor_data.timestamp
                    )

                    self.sensor_interface.update_sensor_data(processed_reading)

                    if sensor_data.stereo_left is not None and sensor_data.stereo_right is not None:
                        self.slam_bridge.process_stereo_frame(
                            sensor_data.stereo_left,
                            sensor_data.stereo_right,
                            sensor_data.timestamp
                        )

                    if sensor_data.imu_data is not None:
                        self.slam_bridge.process_imu_data(
                            sensor_data.imu_data,
                            sensor_data.timestamp
                        )

                time.sleep(1.0 / 30.0)

            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                time.sleep(0.1)

    def _build_state_dict(self) - Dict[str, Any]:

        drone_state = self.airsim_bridge.get_drone_state()

        current_target = self.target_manager.get_current_target()

        sensor_data = self.sensor_interface.get_latest_data()

        state = {
            'position': drone_state.position,
            'orientation': drone_state.orientation,
            'linear_velocity': drone_state.linear_velocity,
            'angular_velocity': drone_state.angular_velocity,
            'battery_level': self.airsim_bridge.get_battery_level(),
            'goal_position': current_target.position if current_target else (0, 0, 0),
            'sensor_data': sensor_data,
            'timestamp': drone_state.timestamp
        }

        return state

    def _get_observation(self) - np.ndarray:

        slam_pose = self.slam_bridge.get_current_pose()
        slam_data = {
            'slam_pose': {
                'position': slam_pose.position,
                'orientation': slam_pose.orientation
            } if slam_pose else None,
            'ate': self.slam_bridge.get_ate()
        }

        current_target = self.target_manager.get_current_target()
        goal_position = current_target.position if current_target else (0, 0, 0)

        observation = self.observation_handler.build_observation(
            drone_state=self.current_state,
            goal_position=goal_position,
            sensor_data=self.current_state.get('sensor_data', {}),
            slam_data=slam_data
        )

        return observation

    def _get_info(self) - Dict[str, Any]:

        current_target = self.target_manager.get_current_target()

        goal_reached = False
        if current_target:
            current_pos = self.current_state.get('position', (0, 0, 0))
            goal_reached = self.target_manager.is_at_target(current_pos)

        collision = self.airsim_bridge.check_collision()

        self.episode_info.goal_reached = goal_reached
        self.episode_info.collision_occurred = collision

        info = {
            'episode_id': self.episode_info.episode_id,
            'step_count': self.episode_info.step_count,
            'goal_reached': goal_reached,
            'collision': collision,
            'current_target': current_target.name if current_target else None,
            'distance_to_goal': self.target_manager.get_distance_to_target(
                self.current_state.get('position', (0, 0, 0))
            ),
            'energy_consumed': self.episode_info.energy_consumed,
            'distance_traveled': self.episode_info.distance_traveled,
            'curriculum_phase': self.curriculum_manager.get_current_phase(),
            'slam_tracking': self.slam_bridge.get_slam_state().is_tracking,
            'battery_level': self.current_state.get('battery_level', 1.0)
        }

        return info

    def _check_terminated(self) - bool:

        return self.episode_info.goal_reached or self.episode_info.collision_occurred

    def _check_truncated(self) - bool:

        if self.episode_info.step_count = self.max_episode_steps:
            self.episode_info.timeout = True
            return True

        elapsed_time = time.time() - self.episode_info.start_time
        if elapsed_time = self.max_episode_time:
            self.episode_info.timeout = True
            return True

        return False

    def _calculate_step_energy(self, action: np.ndarray) - float:

        velocity_magnitude = np.linalg.norm(action[:3])
        energy = velocity_magnitude  2  self.config.get('energy_scale', 0.1)
        return energy

    def _finalize_episode(self):

        episode_duration = time.time() - self.episode_info.start_time

        self.logger.info(f"Episode {self.episode_info.episode_id} completed:")
        self.logger.info(f"  Steps: {self.episode_info.step_count}")
        self.logger.info(f"  Duration: {episode_duration:.2f}s")
        self.logger.info(f"  Goal reached: {self.episode_info.goal_reached}")
        self.logger.info(f"  Collision: {self.episode_info.collision_occurred}")
        self.logger.info(f"  Timeout: {self.episode_info.timeout}")
        self.logger.info(f"  Energy consumed: {self.episode_info.energy_consumed:.2f}")
        self.logger.info(f"  Distance traveled: {self.episode_info.distance_traveled:.2f}m")

    def render(self, mode: str = "rgb_array") - Optional[np.ndarray]:

        if mode == "rgb_array":
            sensor_data = self.airsim_bridge.get_sensor_data()
            if sensor_data.stereo_left is not None:
                return sensor_data.stereo_left

        return None

    def close(self):

        self.logger.info("Closing AirSim environment")

        self._stop_subsystems()

        self.airsim_bridge.disconnect()

        self._save_session_statistics()

    def _save_session_statistics(self):

        try:
            import json

            stats = {
                'total_episodes': self.episode_info.episode_id,
                'average_step_time': np.mean(self.step_times) if self.step_times else 0.0,
                'max_step_time': np.max(self.step_times) if self.step_times else 0.0,
                'action_statistics': self.action_handler.get_action_statistics(),
                'target_statistics': self.target_manager.get_target_statistics(),
                'curriculum_progress': self.curriculum_manager.get_progress_info()
            }

            stats_file = f"data/training/session_stats_{int(time.time())}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            self.logger.info(f"Session statistics saved to {stats_file}")

        except Exception as e:
            self.logger.error(f"Failed to save session statistics: {e}")

    def get_environment_info(self) - Dict[str, Any]:

        return {
            'observation_space': self.observation_handler.get_observation_info(),
            'action_space': self.action_handler.get_action_info(),
            'reward_function': self.reward_function.get_reward_info(),
            'target_system': self.target_manager.get_target_statistics(),
            'curriculum': self.curriculum_manager.get_progress_info(),
            'performance': {
                'average_step_time': np.mean(self.step_times) if self.step_times else 0.0,
                'max_step_time': np.max(self.step_times) if self.step_times else 0.0,
                'total_episodes': self.episode_info.episode_id
            }
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
