"""
Main AirSim Environment
Complete Gymnasium environment integrating all components.
"""

import gymnasium as gym
import numpy as np
import logging
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import cv2  # Import cv2 early
from scipy.spatial.transform import Rotation as R  # Import scipy early


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


@dataclass
class EpisodeInfo:
    """Information about current episode."""

    episode_id: int = 0
    step_count: int = 0
    start_time: float = 0.0
    goal_reached: bool = False
    collision_occurred: bool = False
    timeout: bool = False
    energy_consumed: float = 0.0
    distance_traveled: float = 0.0


class AirSimEnvironment(gym.Env):
    """
    Main AirSim Gymnasium Environment.
    Integrates all subsystems for complete drone delivery environment.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.observation_handler = ObservationSpace(config)
        self.action_handler = ActionSpace(config)
        self.reward_function = RewardFunction(config)
        self.target_manager = TargetManager(config)
        self.curriculum_manager = CurriculumManager(config)
        self.sensor_interface = SensorInterface(config)
        self.drone_controller = DroneController(config)
        self.world_builder = WorldBuilder(config)
        self.logger.info("✓ WorldBuilder created")
        self.logger.info("=" * 50)
        self.logger.info("STARTING BRIDGE INITIALIZATION")
        self.logger.info("=" * 50)

        # Check AirSim before proceeding
        import airsim

        self.logger.info("Creating test AirSim client...")
        try:
            test_client = airsim.MultirotorClient()
            self.logger.info("Confirming AirSim connection...")
            test_client.confirmConnection()
            self.logger.info("Pinging AirSim...")
            test_client.ping()
            self.logger.info("✓ AirSim health check PASSED")
        except Exception as e:
            self.logger.error(f"✗ AirSim health check FAILED: {e}")
            raise

        self.logger.info("Creating AirSimBridge...")
        self.logger.info("✓ AirSimBridge created")

        self.logger.info("Creating SLAMBridge...")
        self.slam_bridge = SLAMBridge(config.get("slam", {}))
        self.logger.info("✓ SLAMBridge created")

        self.logger.info("Creating SensorBridge...")
        self.sensor_bridge = SensorBridge(config.get("sensor", {}))
        self.logger.info("✓ SensorBridge created")

        self.logger.info("=" * 50)
        self.logger.info("ALL BRIDGES INITIALIZED SUCCESSFULLY")
        self.logger.info("=" * 50)

        # ✅ ADD PRINT TO STDOUT (bypasses logger):
        print("=" * 80)
        print("DEBUG: AFTER BRIDGES - STARTING GYM SPACES")
        print("=" * 80)
        import sys

        sys.stdout.flush()  # Force flush

        try:
            print("DEBUG: Creating observation space...")
            sys.stdout.flush()

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
            )

            print(f"DEBUG: Observation space created: {self.observation_space.shape}")
            sys.stdout.flush()

        except Exception as e:
            print(f"DEBUG: FAILED TO CREATE OBSERVATION SPACE: {e}")
            import traceback

            traceback.print_exc()
            sys.stdout.flush()
            raise

        # Initialize bridges
        self.airsim_bridge = AirSimBridge(config.get("airsim", {}))

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.action_handler.action_low,
            high=self.action_handler.action_high,
            dtype=np.float32,
        )

        # Episode management
        self.episode_info = EpisodeInfo()
        self.max_episode_steps = config.get(
            "max_episode_steps", 6000
        )  # 5 minutes at 20Hz
        self.max_episode_time = config.get("max_episode_time", 300.0)  # 5 minutes

        # State tracking
        self.current_state: Dict[str, Any] = {}
        self.previous_position: Optional[Tuple[float, float, float]] = None

        # Threading for real-time operation
        self.update_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.update_frequency = config.get("update_frequency", 20.0)  # Hz

        # Performance tracking
        self.step_times: List[float] = []
        self.max_step_time_history = 1000

        self.world_built = False

        self.logger.info("AirSim Environment initialized")
        self.logger.info(f"Observation space: {self.observation_space.shape}")
        self.logger.info(f"Action space: {self.action_space.shape}")
        self.logger.info(f"Max episode steps: {self.max_episode_steps}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)

        start_time = time.time()
        self.logger.info(
            f"Resetting environment (episode {self.episode_info.episode_id + 1})"
        )

        # Reset episode info
        self.episode_info = EpisodeInfo(
            episode_id=self.episode_info.episode_id + 1, start_time=start_time
        )

        # Connect to AirSim if not connected
        if not self.airsim_bridge.is_connected:
            if not self.airsim_bridge.connect():
                raise RuntimeError("Failed to connect to AirSim")

        # Reset drone to spawn location
        self.airsim_bridge.reset_drone()

        # Start subsystems
        self._start_subsystems()

        # Update curriculum phase
        current_phase = self.curriculum_manager.get_current_phase()
        self.target_manager.set_curriculum_phase(current_phase)

        # Select new target
        target = self.target_manager.select_target()
        self.logger.info(f"Selected target: {target.name} at {target.position}")

        # Reset components
        self.reward_function.reset_episode()
        self.target_manager.reset_episode()
        self.sensor_interface.reset()
        self.drone_controller.reset()

        # Build world if needed
        # self.world_builder.update_world_state()

        if not self.world_built:
            try:
                self.logger.info("Building world (first episode only)...")
                self.world_builder.update_world_state()
                self.world_built = True
                self.logger.info("✓ World built successfully")
                time.sleep(2.0)  # Extra time for world to stabilize
            except Exception as e:
                self.logger.warning(f"Could not update world state: {e}")
                self.logger.info("Continuing without world update...")
        # Wait for systems to stabilize
        self.logger.info("Waiting for AirSim to stabilize...")
        time.sleep(2.0)

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        # Reset state tracking
        self.current_state = self._build_state_dict()
        self.previous_position = self.current_state.get("position")

        reset_time = time.time() - start_time
        self.logger.info(f"Environment reset completed in {reset_time:.2f}s")

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Action vector [vx, vy, vz, yaw_rate]

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        step_start_time = time.time()

        # Validate and process action
        if not self.action_handler.validate_action(action):
            self.logger.warning("Invalid action received, clipping to valid range")

        # Apply safety constraints
        safe_action = self.action_handler.apply_safety_constraints(
            action, self.current_state
        )

        # Record action for statistics
        self.action_handler.record_action(safe_action)

        # Store previous state
        previous_state = self.current_state.copy()

        # Execute action through drone controller
        self.drone_controller.execute_action(safe_action)

        # Wait for control cycle
        time.sleep(1.0 / self.update_frequency)

        # Get new state
        self.current_state = self._build_state_dict()

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        info = self._get_info()
        reward = self.reward_function.compute_reward(
            previous_state, safe_action, self.current_state, info
        )

        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # Update episode info
        self.episode_info.step_count += 1
        self.episode_info.energy_consumed += self._calculate_step_energy(safe_action)

        if self.previous_position is not None:
            current_pos = self.current_state.get("position", (0, 0, 0))
            distance_step = np.linalg.norm(
                np.array(current_pos) - np.array(self.previous_position)
            )
            self.episode_info.distance_traveled += distance_step
            self.previous_position = current_pos

        # Update curriculum if episode completed
        if terminated or truncated:
            self._finalize_episode()
            self.curriculum_manager.update_progress(
                success=self.episode_info.goal_reached,
                collision=self.episode_info.collision_occurred,
            )

        # Track step performance
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        if len(self.step_times) > self.max_step_time_history:
            self.step_times.pop(0)

        return observation, float(reward), terminated, truncated, info

    def _start_subsystems(self):
        """Start all subsystems."""
        try:
            # Start SLAM system
            if not self.slam_bridge.start_slam():
                self.logger.warning("Failed to start SLAM system")

            # Start sensor processing
            self.sensor_bridge.start_processing()

            # Start sensor interface
            self.sensor_interface.start()

            # Start real-time update thread
            if not self.is_running:
                self.is_running = True
                self.update_thread = threading.Thread(target=self._update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()

        except Exception as e:
            self.logger.error(f"Failed to start subsystems: {e}")

    def _stop_subsystems(self):
        """Stop all subsystems."""
        try:
            # Stop update thread
            self.is_running = False
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)

            # Stop subsystems
            self.sensor_interface.stop()
            self.sensor_bridge.stop_processing()
            self.slam_bridge.stop_slam()

        except Exception as e:
            self.logger.error(f"Error stopping subsystems: {e}")

    def _update_loop(self):
        """Real-time update loop for sensor data processing."""
        self.logger.info("Starting real-time update loop")

        while self.is_running:
            try:
                # Get sensor data from AirSim
                sensor_data = self.airsim_bridge.get_sensor_data()

                # Process through sensor bridge
                if (
                    sensor_data.stereo_left is not None
                    and sensor_data.stereo_right is not None
                ):
                    processed_reading = self.sensor_bridge.process_sensor_data(
                        sensor_data.stereo_left,
                        sensor_data.stereo_right,
                        sensor_data.depth_image,
                        sensor_data.imu_data,
                        sensor_data.timestamp,
                    )

                    # Update sensor interface
                    self.sensor_interface.update_sensor_data(processed_reading)

                    # Feed to SLAM system
                    if (
                        sensor_data.stereo_left is not None
                        and sensor_data.stereo_right is not None
                    ):
                        self.slam_bridge.process_stereo_frame(
                            sensor_data.stereo_left,
                            sensor_data.stereo_right,
                            sensor_data.timestamp,
                        )

                    if sensor_data.imu_data is not None:
                        self.slam_bridge.process_imu_data(
                            sensor_data.imu_data, sensor_data.timestamp
                        )

                time.sleep(1.0 / 30.0)  # 30Hz update rate

            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                time.sleep(0.1)

    def _build_state_dict(self) -> Dict[str, Any]:
        """Build complete state dictionary."""
        # Get drone state from AirSim
        drone_state = self.airsim_bridge.get_drone_state()

        # Get current target
        current_target = self.target_manager.get_current_target()

        # Get sensor data
        sensor_data = self.sensor_interface.get_latest_data()

        state = {
            "position": drone_state.position,
            "orientation": drone_state.orientation,
            "linear_velocity": drone_state.linear_velocity,
            "angular_velocity": drone_state.angular_velocity,
            "battery_level": self.airsim_bridge.get_battery_level(),
            "goal_position": current_target.position if current_target else (0, 0, 0),
            "sensor_data": sensor_data,
            "timestamp": drone_state.timestamp,
        }

        return state

    def _get_observation(self) -> np.ndarray:
        """
        Get 40D observation vector.
        Returns: [pose(7), velocity(4), goal(3), battery(1), occupancy(24), localization(1)]
        """
        try:
            # Get drone state
            state = self.airsim_bridge.get_drone_state()

            # 1. Pose (7D) - position (3D) + orientation (4D)
            position = np.array(
                [
                    float(state.position[0]),
                    float(state.position[1]),
                    float(state.position[2]),
                ],
                dtype=np.float32,
            )

            orientation = np.array(
                [
                    float(state.orientation[0]),  # w
                    float(state.orientation[1]),  # x
                    float(state.orientation[2]),  # y
                    float(state.orientation[3]),  # z
                ],
                dtype=np.float32,
            )

            # 2. Velocity (4D) - linear (3D) + angular magnitude (1D)
            linear_vel = np.array(
                [
                    float(state.linear_velocity[0]),
                    float(state.linear_velocity[1]),
                    float(state.linear_velocity[2]),
                ],
                dtype=np.float32,
            )

            angular_vel = np.array(
                [
                    float(state.angular_velocity[0]),
                    float(state.angular_velocity[1]),
                    float(state.angular_velocity[2]),
                ],
                dtype=np.float32,
            )
            angular_vel_mag = np.array([np.linalg.norm(angular_vel)], dtype=np.float32)

            # 3. Goal relative (3D)
            if (
                hasattr(self, "current_target_position")
                and self.current_target_position is not None
            ):
                goal_pos = np.array(
                    [
                        float(self.current_target_position[0]),
                        float(self.current_target_position[1]),
                        float(self.current_target_position[2]),
                    ],
                    dtype=np.float32,
                )
                goal_rel = goal_pos - position
            else:
                goal_rel = np.zeros(3, dtype=np.float32)

            # 4. Battery (1D)
            battery = np.array([1.0], dtype=np.float32)

            # 5. Occupancy (24D) - placeholder for now
            occupancy = np.zeros(24, dtype=np.float32)

            # 6. Localization confidence (1D)
            localization = np.array([1.0], dtype=np.float32)

            # Concatenate all components
            observation = np.concatenate(
                [
                    position,  # 3
                    orientation,  # 4
                    linear_vel,  # 3
                    angular_vel_mag,  # 1
                    goal_rel,  # 3
                    battery,  # 1
                    occupancy,  # 24
                    localization,  # 1
                ],
                axis=0,
            )

            # Debug: verify size
            if observation.shape[0] != 40:
                self.logger.error(
                    f"Observation size mismatch! Got {observation.shape[0]}, expected 40"
                )
                self.logger.error(f"  position: {position.shape}")
                self.logger.error(f"  orientation: {orientation.shape}")
                self.logger.error(f"  linear_vel: {linear_vel.shape}")
                self.logger.error(f"  angular_vel_mag: {angular_vel_mag.shape}")
                self.logger.error(f"  goal_rel: {goal_rel.shape}")
                self.logger.error(f"  battery: {battery.shape}")
                self.logger.error(f"  occupancy: {occupancy.shape}")
                self.logger.error(f"  localization: {localization.shape}")
                # Pad with zeros if too short
                if observation.shape[0] < 40:
                    padding = np.zeros(40 - observation.shape[0], dtype=np.float32)
                    observation = np.concatenate([observation, padding])
                else:
                    observation = observation[:40]

            return observation.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Failed to get observation: {e}")
            import traceback

            traceback.print_exc()
            # Return zero observation
            return np.zeros(40, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get episode information dictionary."""
        current_target = self.target_manager.get_current_target()

        # Check goal reached
        goal_reached = False
        if current_target:
            current_pos = self.current_state.get("position", (0, 0, 0))
            goal_reached = self.target_manager.is_at_target(current_pos)

        # Check collision
        collision = self.airsim_bridge.check_collision()

        # Update episode info
        self.episode_info.goal_reached = goal_reached
        self.episode_info.collision_occurred = collision

        info = {
            "episode_id": self.episode_info.episode_id,
            "step_count": self.episode_info.step_count,
            "goal_reached": goal_reached,
            "collision": collision,
            "current_target": current_target.name if current_target else None,
            "distance_to_goal": self.target_manager.get_distance_to_target(
                self.current_state.get("position", (0, 0, 0))
            ),
            "energy_consumed": self.episode_info.energy_consumed,
            "distance_traveled": self.episode_info.distance_traveled,
            "curriculum_phase": self.curriculum_manager.get_current_phase(),
            "slam_tracking": self.slam_bridge.get_slam_state().is_tracking,
            "battery_level": self.current_state.get("battery_level", 1.0),
        }

        return info

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (goal reached or collision)."""
        return self.episode_info.goal_reached or self.episode_info.collision_occurred

    def _check_truncated(self) -> bool:
        """Check if episode should truncate (timeout)."""
        # Step limit
        if self.episode_info.step_count >= self.max_episode_steps:
            self.episode_info.timeout = True
            return True

        # Time limit
        elapsed_time = time.time() - self.episode_info.start_time
        if elapsed_time >= self.max_episode_time:
            self.episode_info.timeout = True
            return True

        return False

    def _calculate_step_energy(self, action: np.ndarray) -> float:
        """Calculate energy consumed in this step."""
        # Simplified energy model based on action magnitude
        velocity_magnitude = np.linalg.norm(action[:3])
        # Energy proportional to velocity squared (aerodynamic drag)
        energy = velocity_magnitude**2 * self.config.get("energy_scale", 0.1)
        return energy

    def _finalize_episode(self):
        """Finalize episode statistics and logging."""
        episode_duration = time.time() - self.episode_info.start_time

        self.logger.info(f"Episode {self.episode_info.episode_id} completed:")
        self.logger.info(f"  Steps: {self.episode_info.step_count}")
        self.logger.info(f"  Duration: {episode_duration:.2f}s")
        self.logger.info(f"  Goal reached: {self.episode_info.goal_reached}")
        self.logger.info(f"  Collision: {self.episode_info.collision_occurred}")
        self.logger.info(f"  Timeout: {self.episode_info.timeout}")
        self.logger.info(f"  Energy consumed: {self.episode_info.energy_consumed:.2f}")
        self.logger.info(
            f"  Distance traveled: {self.episode_info.distance_traveled:.2f}m"
        )

    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render environment (AirSim handles visualization)."""
        if mode == "rgb_array":
            # Get camera image from AirSim
            sensor_data = self.airsim_bridge.get_sensor_data()
            if sensor_data.stereo_left is not None:
                return sensor_data.stereo_left

        return None

    def close(self):
        """Close environment and cleanup resources."""
        self.logger.info("Closing AirSim environment")

        # Stop subsystems
        self._stop_subsystems()

        # Disconnect from AirSim
        self.airsim_bridge.disconnect()

        # Save final statistics
        self._save_session_statistics()

    def _save_session_statistics(self):
        """Save session-wide statistics."""
        try:
            import json

            stats = {
                "total_episodes": self.episode_info.episode_id,
                "average_step_time": (
                    np.mean(self.step_times) if self.step_times else 0.0
                ),
                "max_step_time": np.max(self.step_times) if self.step_times else 0.0,
                "action_statistics": self.action_handler.get_action_statistics(),
                "target_statistics": self.target_manager.get_target_statistics(),
                "curriculum_progress": self.curriculum_manager.get_progress_info(),
            }

            # Save to file
            stats_file = f"data/training/session_stats_{int(time.time())}.json"
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)

            self.logger.info(f"Session statistics saved to {stats_file}")

        except Exception as e:
            self.logger.error(f"Failed to save session statistics: {e}")

    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        return {
            "observation_space": self.observation_handler.get_observation_info(),
            "action_space": self.action_handler.get_action_info(),
            "reward_function": self.reward_function.get_reward_info(),
            "target_system": self.target_manager.get_target_statistics(),
            "curriculum": self.curriculum_manager.get_progress_info(),
            "performance": {
                "average_step_time": (
                    np.mean(self.step_times) if self.step_times else 0.0
                ),
                "max_step_time": np.max(self.step_times) if self.step_times else 0.0,
                "total_episodes": self.episode_info.episode_id,
            },
        }

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
