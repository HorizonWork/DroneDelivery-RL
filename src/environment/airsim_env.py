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
import cv2
from scipy.spatial.transform import Rotation as R

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
        
        # ✅ FIX: Configure DroneController to NOT scale actions
        controller_config = config.get("drone_controller", {}).copy()
        controller_config["expect_normalized_actions"] = False
        controller_config["strict_acceleration_check"] = False
        self.drone_controller = DroneController(controller_config)
        
        # WorldBuilder - Optional for synthetic environments
        world_config = config.get("world_builder", {})
        if world_config.get("enabled", False):
            self.world_builder = WorldBuilder(config)
            self.logger.info("WorldBuilder created - SYNTHETIC environment")
        else:
            self.world_builder = None
            self.logger.info("WorldBuilder DISABLED - using REAL UE environment")
        
        self.logger.info("=" * 50)
        self.logger.info("STARTING BRIDGE INITIALIZATION")
        self.logger.info("=" * 50)
        
        # Create AirSimBridge FIRST
        self.logger.info("Creating AirSimBridge...")
        self.airsim_bridge = AirSimBridge(config.get("airsim", {}))
        self.logger.info("AirSimBridge created")
        
        # ✅ CRITICAL: Connect DroneController to AirSimBridge
        self.drone_controller.set_airsim_bridge(self.airsim_bridge)
        
        # Check AirSim connection using bridge
        try:
            if not self.airsim_bridge.connect():
                raise RuntimeError("Failed to connect to AirSim")
            self.logger.info("AirSim health check PASSED")
        except Exception as e:
            self.logger.error(f"✗ AirSim health check FAILED: {e}")
            raise
        
        self.logger.info("Creating SLAMBridge...")
        self.slam_bridge = SLAMBridge(config.get("slam", {}))
        self.logger.info("SLAMBridge created")
        
        self.logger.info("Creating SensorBridge...")
        self.sensor_bridge = SensorBridge(config.get("sensor", {}))
        self.logger.info("SensorBridge created")
        
        self.logger.info("=" * 50)
        self.logger.info("ALL BRIDGES INITIALIZED SUCCESSFULLY")
        self.logger.info("=" * 50)
        
        # Initial altitude
        altitude_override = config.get(
            "initial_takeoff_altitude",
            config.get("airsim", {}).get("takeoff_altitude", 3.0),
        )
        try:
            altitude_override = float(altitude_override)
        except (TypeError, ValueError):
            altitude_override = 3.0
        self.initial_takeoff_altitude = max(0.0, altitude_override)
        
        # Spawn configuration
        spawn_source = (
            config.get("spawn_location")
            or config.get("airsim", {}).get("spawn_location")
            or self.airsim_bridge.spawn_location
        )
        self.base_spawn_location = tuple(float(v) for v in spawn_source)
        
        orientation_source = (
            config.get("spawn_orientation")
            or config.get("airsim", {}).get("spawn_orientation")
            or self.airsim_bridge.spawn_orientation
        )
        self.base_spawn_orientation = tuple(float(v) for v in orientation_source)
        
        spawn_rand_cfg = config.get("spawn_randomization", {})
        if isinstance(spawn_rand_cfg, dict):
            enabled = spawn_rand_cfg.get("enabled", False)
            xy_jitter = float(spawn_rand_cfg.get("xy_jitter_m", 2.0))
            max_radius = float(spawn_rand_cfg.get("max_radius_m", 200.0))
        else:
            enabled = bool(spawn_rand_cfg)
            xy_jitter = 2.0
            max_radius = 200.0
        
        if not enabled:
            enabled = bool(config.get("episode", {}).get("spawn_randomization", False))
        
        self.spawn_randomization_enabled = bool(enabled)
        self.spawn_xy_jitter = max(0.0, xy_jitter)
        self.spawn_max_radius = max(self.spawn_xy_jitter, max_radius)
        
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
        self.max_episode_steps = config.get("max_episode_steps", 6000)
        self.max_episode_time = config.get("max_episode_time", 300.0)
        
        # State tracking
        self.current_state: Dict[str, Any] = {}
        self.previous_position: Optional[Tuple[float, float, float]] = None
        self.current_target_position: Optional[Tuple[float, float, float]] = None
        
        # ✅ FIX: Disable sensor threads to avoid IOLoop conflicts
        self.enable_sensor_threads = config.get("enable_sensor_threads", False)
        self.update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.update_frequency = config.get("update_frequency", 20.0)
        
        # Performance tracking
        self.step_times: List[float] = []
        self.max_step_time_history = 1000
        
        self.world_built = False
        
        self.logger.info("AirSim Environment initialized")
        self.logger.info(f"Observation space: {self.observation_space.shape}")
        self.logger.info(f"Action space: {self.action_space.shape}")
        self.logger.info(f"Max episode steps: {self.max_episode_steps}")
        self.logger.info(f"Sensor threads enabled: {self.enable_sensor_threads}")
        self._logged_reset_obs = False
        self._logged_step_obs = False
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        start_time = time.time()
        self.logger.info(f"Resetting environment (episode {self.episode_info.episode_id + 1})")
        
        # Reset episode info
        self.episode_info = EpisodeInfo(
            episode_id=self.episode_info.episode_id + 1, start_time=start_time
        )
        
        # Connect to AirSim if not connected
        if not self.airsim_bridge.is_connected:
            if not self.airsim_bridge.connect():
                raise RuntimeError("Failed to connect to AirSim")
        
        # Select spawn pose
        episode_spawn = self._sample_spawn_location()
        self.airsim_bridge.spawn_location = episode_spawn
        self.airsim_bridge.spawn_orientation = self.base_spawn_orientation
        if self.world_builder:
            self.world_builder.update_spawn_reference(episode_spawn)
            self.world_builder.reset_world()
        self.logger.info(f"Episode spawn location: {episode_spawn}")
        
        # Reset drone to spawn location
        self.airsim_bridge.reset_drone()
        
        # Execute deterministic takeoff
        self._perform_initial_takeoff()
        self._log_initial_state()
        
        # ✅ FIX: Only start sensor threads if enabled
        if self.enable_sensor_threads:
            self._start_subsystems()
        else:
            self.logger.warning("Sensor threads DISABLED to avoid IOLoop conflicts")
        
        # Update curriculum phase
        current_phase = self.curriculum_manager.get_current_phase()
        self.target_manager.set_curriculum_phase(current_phase)
        
        # Select new target
        target = self.target_manager.select_target()
        self.current_target_position = target.position
        self.logger.info(f"Selected target: {target.name} at {target.position}")
        
        # Reset components
        self.reward_function.reset_episode()
        self.target_manager.reset_episode()
        self.sensor_interface.reset()
        self.drone_controller.reset()
        
        # Build world if needed
        if not self.world_built:
            try:
                self.logger.info("Building world (first episode only)...")
                self.world_builder.update_world_state()
                self.world_built = True
                self.logger.info("World built successfully")
                time.sleep(1.0)
            except Exception as e:
                self.logger.warning(f"Could not update world state: {e}")
        
        # Wait for stabilization
        self.logger.info("Waiting for AirSim to stabilize...")
        time.sleep(1.0)
        
        # Get initial observation
        observation = self._get_observation()
        if not self._logged_reset_obs:
            self.logger.info(f"Observation vector shape after reset: {observation.shape}")
            self._logged_reset_obs = True
        info = self._get_info()
        
        # Reset state tracking
        self.current_state = self._build_state_dict()
        self.previous_position = self.current_state.get("position")
        
        reset_time = time.time() - start_time
        self.logger.info(f"Environment reset completed in {reset_time:.2f}s")
        
        return observation, info
    
    def _sample_spawn_location(self) -> Tuple[float, float, float]:
        """Sample a valid spawn location within configured bounds."""
        base = np.array(self.base_spawn_location, dtype=np.float32)
        if (
            not self.spawn_randomization_enabled
            or self.spawn_xy_jitter <= 0.0
            or self.spawn_xy_jitter < 1e-6
        ):
            return tuple(base.tolist())
        
        noise = np.random.uniform(
            low=-self.spawn_xy_jitter, high=self.spawn_xy_jitter, size=2
        )
        candidate = base.copy()
        candidate[0] += noise[0]
        candidate[1] += noise[1]
        
        radial_distance = float(np.hypot(candidate[0], candidate[1]))
        if radial_distance > self.spawn_max_radius:
            self.logger.warning(
                "Spawn randomization produced |pos|=%.1fm beyond %.1fm limit; falling back to base spawn",
                radial_distance,
                self.spawn_max_radius,
            )
            return tuple(self.base_spawn_location)
        
        return tuple(candidate.tolist())
    
    def _perform_initial_takeoff(self) -> None:
        """Execute a deterministic takeoff after resetting the drone."""
        target_altitude = getattr(self, "initial_takeoff_altitude", 3.0)
        if target_altitude <= 0.0:
            self.logger.debug("Initial takeoff disabled (altitude <= 0)")
            return
        
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            if self.airsim_bridge.takeoff(altitude=target_altitude):
                self.logger.info(
                    "Takeoff stabilized at %.1fm on attempt %d",
                    target_altitude,
                    attempt,
                )
                time.sleep(0.5)
                return
            
            self.logger.warning("Takeoff attempt %d failed, resetting drone before retry", attempt)
            self.airsim_bridge.reset_drone()
        
        raise RuntimeError("Unable to complete takeoff after multiple attempts")
    
    def _log_initial_state(self) -> None:
        """Log spawn, orientation, and measured altitude for debugging."""
        try:
            drone_state = self.airsim_bridge.get_drone_state()
        except Exception as exc:
            self.logger.warning("Unable to query drone state after reset: %s", exc)
            return
        
        spawn_location = tuple(round(v, 3) for v in self.airsim_bridge.spawn_location)
        spawn_orientation = tuple(round(v, 3) for v in self.airsim_bridge.spawn_orientation)
        position = tuple(round(v, 3) for v in drone_state.position)
        altitude_m = -position[2]
        
        self.logger.info(
            "Episode %d spawn summary -> cmd_xyz=%s, cmd_orientation=%s, airsim_xyz=%s (altitude %.2fm)",
            self.episode_info.episode_id,
            spawn_location,
            spawn_orientation,
            position,
            altitude_m,
        )
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        step_start_time = time.time()
        
        # Validate and process action
        if not self.action_handler.validate_action(action):
            self.logger.warning("Invalid action received, clipping to valid range")
        
        # Apply safety constraints (clips action)
        safe_action = self.action_handler.apply_safety_constraints(action, self.current_state)
        
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
        if not self._logged_step_obs:
            self.logger.info(f"Observation vector shape during step: {observation.shape}")
            self._logged_step_obs = True
        
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
                timesteps_this_episode=self.episode_info.step_count,
            )
        
        # Track step performance
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        if len(self.step_times) > self.max_step_time_history:
            self.step_times.pop(0)
        
        return observation, float(reward), terminated, truncated, info
    
    def _start_subsystems(self):
        """Start all subsystems (DISABLED for IOLoop fix)."""
        try:
            # ✅ CRITICAL: Disable ALL async threads to fix IOLoop
            self.logger.warning("🔧 SENSOR THREADS DISABLED - Training in sync mode")
            
            # DO NOT START:
            # - slam_bridge threads
            # - sensor_bridge threads  
            # - sensor_interface threads
            # - update_loop thread
            
            # Only initialize non-threaded components
            # self.slam_bridge.start_slam()  # ❌ DISABLED
            # self.sensor_bridge.start_processing()  # ❌ DISABLED
            # self.sensor_interface.start()  # ❌ DISABLED
            # self._stop_event.clear()  # ❌ DISABLED
            # self.update_thread = ...  # ❌ DISABLED
            
        except Exception as e:
            self.logger.error(f"Failed to start subsystems: {e}")
    
    def _stop_subsystems(self):
        """Stop all subsystems."""
        try:
            # Signal stop
            self._stop_event.set()
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=2.0)
            
            # Stop subsystems
            self.sensor_interface.stop()
            self.sensor_bridge.stop_processing()
            self.slam_bridge.stop_slam()
            
        except Exception as e:
            self.logger.error(f"Error stopping subsystems: {e}")
    
    def _update_loop(self):
        """Real-time update loop (DISABLED)."""
        # This method won't be called when sensor threads disabled
        pass
    
    def _build_state_dict(self) -> Dict[str, Any]:
        """Build complete state dictionary."""
        # Get drone state from AirSim
        drone_state = self.airsim_bridge.get_drone_state()
        
        # Get current target
        current_target = self.target_manager.get_current_target()
        
        # Get sensor data (placeholder when threads disabled)
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
        """Get 40D observation vector."""
        try:
            state = self.airsim_bridge.get_drone_state()
            
            # 1. Pose (7D)
            position = np.array([
                float(state.position[0]),
                float(state.position[1]),
                float(state.position[2]),
            ], dtype=np.float32)
            
            orientation = np.array([
                float(state.orientation[0]),  # w
                float(state.orientation[1]),  # x
                float(state.orientation[2]),  # y
                float(state.orientation[3]),  # z
            ], dtype=np.float32)
            
            # 2. Velocity (4D)
            linear_vel = np.array([
                float(state.linear_velocity[0]),
                float(state.linear_velocity[1]),
                float(state.linear_velocity[2]),
            ], dtype=np.float32)
            
            angular_vel = np.array([
                float(state.angular_velocity[0]),
                float(state.angular_velocity[1]),
                float(state.angular_velocity[2]),
            ], dtype=np.float32)
            angular_vel_mag = np.array([np.linalg.norm(angular_vel)], dtype=np.float32)
            
            # 3. Goal relative (3D)
            if self.current_target_position is not None:
                goal_pos = np.array(self.current_target_position, dtype=np.float32)
                goal_rel = goal_pos - position
            else:
                goal_rel = np.zeros(3, dtype=np.float32)
            
            # 4. Battery (1D)
            battery = np.array([1.0], dtype=np.float32)
            
            # 5. Occupancy (24D)
            occupancy = np.zeros(24, dtype=np.float32)
            
            # 6. Localization confidence (1D)
            localization = np.array([1.0], dtype=np.float32)
            
            # Concatenate
            observation = np.concatenate([
                position,  # 3
                orientation,  # 4
                linear_vel,  # 3
                angular_vel_mag,  # 1
                goal_rel,  # 3
                battery,  # 1
                occupancy,  # 24
                localization,  # 1
            ], axis=0)
            
            if observation.shape[0] != 40:
                self.logger.error(f"Observation size mismatch! Got {observation.shape[0]}, expected 40")
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
            "success": goal_reached,
            "current_target": current_target.name if current_target else None,
            "distance_to_goal": self.target_manager.get_distance_to_target(
                self.current_state.get("position", (0, 0, 0))
            ),
            "energy_consumed": self.episode_info.energy_consumed,
            "energy_consumption": self.episode_info.energy_consumed,
            "distance_traveled": self.episode_info.distance_traveled,
            "curriculum_phase": self.curriculum_manager.get_current_phase(),
            "slam_tracking": self.slam_bridge.get_slam_state().is_tracking,
            "battery_level": self.current_state.get("battery_level", 1.0),
            "spawn_location": self.airsim_bridge.spawn_location,
            "spawn_orientation": self.airsim_bridge.spawn_orientation,
            "position": self.current_state.get("position", (0, 0, 0)),
        }
        
        try:
            phase_name = self.curriculum_manager.get_current_config().name
        except Exception:
            phase_name = "unknown"
        info["curriculum_phase_name"] = phase_name
        
        reward_breakdown = self.reward_function.get_last_components_dict()
        if reward_breakdown:
            info["reward_components"] = reward_breakdown
        
        return info
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        return self.episode_info.goal_reached or self.episode_info.collision_occurred
    
    def _check_truncated(self) -> bool:
        """Check if episode should truncate."""
        if self.episode_info.step_count >= self.max_episode_steps:
            self.episode_info.timeout = True
            return True
        
        elapsed_time = time.time() - self.episode_info.start_time
        if elapsed_time >= self.max_episode_time:
            self.episode_info.timeout = True
            return True
        
        return False
    
    def _calculate_step_energy(self, action: np.ndarray) -> float:
        """Calculate energy consumed in this step."""
        velocity_magnitude = np.linalg.norm(action[:3])
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
        self.logger.info(f"  Distance traveled: {self.episode_info.distance_traveled:.2f}m")
        
        reward_totals = self.reward_function.get_episode_totals()
        if reward_totals:
            self.logger.info(f"  Reward totals: {reward_totals}")
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render environment."""
        if mode == "rgb_array":
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
                "average_step_time": np.mean(self.step_times) if self.step_times else 0.0,
                "max_step_time": np.max(self.step_times) if self.step_times else 0.0,
                "action_statistics": self.action_handler.get_action_statistics(),
                "target_statistics": self.target_manager.get_target_statistics(),
                "curriculum_progress": self.curriculum_manager.get_progress_info(),
            }
            
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
                "average_step_time": np.mean(self.step_times) if self.step_times else 0.0,
                "max_step_time": np.max(self.step_times) if self.step_times else 0.0,
                "total_episodes": self.episode_info.episode_id,
            },
        }
    
    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
