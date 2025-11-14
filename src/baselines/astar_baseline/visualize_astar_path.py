"""
A* Path Visualization in Unreal Engine
Queries DroneSpawn and Landing_XXX actors from UE, plans A* path, 
and visualizes trajectory in real-time using AirSim plotting API.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import random
import time

import airsim
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.astar_baseline.astar_controller import AStarController
from src.baselines.astar_baseline.pid_controller import PIDController
from src.baselines.astar_baseline.hierarchical_planner import HierarchicalPlanner


class AStarPathVisualizer:
    """
    Visualizes A* paths in Unreal Engine using actual spawn points and landing targets.
    Uses hierarchical planning for efficient multi-floor navigation.
    """
    
    def __init__(self, map_file: str, config: Dict[str, Any]):
        """
        Args:
            map_file: Path to map metadata JSON
            config: Configuration dictionary
        """
        self.config = config
        self.map_file = map_file
        
        # AirSim connection
        self.client = None
        
        # Controllers
        self.astar_controller = AStarController(config, map_file=map_file)
        self.hierarchical_planner = HierarchicalPlanner(self.astar_controller)  # NEW
        self.pid_controller = PIDController(config)
        
        # Target positions (loaded from UE)
        self.spawn_position: Optional[Tuple[float, float, float]] = None
        self.actual_start_position: Optional[Tuple[float, float, float]] = None
        self.landing_targets: Dict[str, Tuple[float, float, float]] = {}
        
        print("✅ A* Path Visualizer initialized")
        print(f"   Map file: {map_file}")
    
    def connect_airsim(self):
        """Connect to AirSim simulator"""
        print("\n🔗 Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ Connected to AirSim")
    
    def load_ue_actors(self):
        """
        Load spawn point and landing target positions from Unreal Engine actors.
        Uses AirSim's simGetObjectPose() to query actor positions.
        """
        print("\n📍 Loading actor positions from Unreal Engine...")
        
        # ========================================
        # Load DroneSpawn position
        # ========================================
        try:
            spawn_pose = self.client.simGetObjectPose("DroneSpawn")
            self.spawn_position = (
                spawn_pose.position.x_val,
                spawn_pose.position.y_val,
                spawn_pose.position.z_val
            )
            print(f"✅ DroneSpawn: ({self.spawn_position[0]:.2f}, "
                  f"{self.spawn_position[1]:.2f}, {self.spawn_position[2]:.2f})")
        except Exception as e:
            print(f"⚠️  Could not find DroneSpawn actor: {e}")
            print("   Using default spawn position (0, 0, 0)")
            self.spawn_position = (0.0, 0.0, 0.0)
        
        # ========================================
        # Load all Landing_XXX positions
        # ========================================
        floors = [1, 2, 3, 4, 5]
        targets_per_floor = [101, 102, 103, 104, 105, 106,
                             201, 202, 203, 204, 205, 206,
                             301, 302, 303, 304, 305, 306,
                             401, 402, 403, 404, 405, 406,
                             501, 502, 503, 504, 505, 506]
        
        loaded_count = 0
        failed_count = 0
        
        # Safety offset: fly 1.2m above landing pad to avoid collision with pillar
        safety_offset_z = -1.2  # -1.2m in NED coordinates (negative = up)
        
        for target_id in targets_per_floor:
            target_name = f"Landing_{target_id}"
            try:
                target_pose = self.client.simGetObjectPose(target_name)
                position = (
                    target_pose.position.x_val,
                    target_pose.position.y_val,
                    target_pose.position.z_val + safety_offset_z  # Add 90cm up
                )
                self.landing_targets[target_name] = position
                loaded_count += 1
            except Exception as e:
                failed_count += 1
                # Don't print every failure to reduce noise
                if failed_count <= 3:
                    print(f"   ⚠️  Could not find {target_name}: {e}")
        
        print(f"\n📊 Actor Loading Summary:")
        print(f"   ✅ Successfully loaded: {loaded_count} landing targets")
        print(f"   🛡️  Safety offset applied: +1.2m above each landing pad")
        if failed_count > 0:
            print(f"   ⚠️  Failed to load: {failed_count} targets")
            print(f"   💡 Make sure Landing_XXX actors exist in your UE scene")
        
        if loaded_count == 0:
            print("\n❌ ERROR: No landing targets found!")
            print("   Please ensure your UE scene contains Landing_XXX actors")
            print("   Example names: Landing_101, Landing_102, ..., Landing_506")
            return False
        
        return True
    
    def prepare_drone(self):
        """Enable control and arm drone"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    
    def reset_to_spawn(self):
        """
        Reset drone to DroneSpawn position for consistent mission start.
        
        Research rationale:
        - Ensures reproducible initial conditions across all missions
        - Eliminates variability from random spawn positions or falling
        - Provides known safe starting position
        """
        if self.spawn_position is None:
            print("⚠️  DroneSpawn position not loaded, using current position")
            return
        
        print(f"\n📍 Resetting drone to DroneSpawn position...")
        print(f"   Target: ({self.spawn_position[0]:.2f}, "
              f"{self.spawn_position[1]:.2f}, {self.spawn_position[2]:.2f})")
        
        # Reset to spawn position using simSetVehiclePose
        # This provides clean initial state without gravity effects
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(
                    self.spawn_position[0],
                    self.spawn_position[1],
                    self.spawn_position[2]
                ),
                airsim.Quaternionr(0, 0, 0, 1)  # Level orientation
            ),
            True  # Ignore collision (for reset)
        )
        time.sleep(1.0)  # Allow physics to stabilize
        
        # Verify position after reset
        state = self.client.getMultirotorState()
        actual_pos = state.kinematics_estimated.position
        print(f"   Actual: ({actual_pos.x_val:.2f}, "
              f"{actual_pos.y_val:.2f}, {actual_pos.z_val:.2f})")
        print("✅ Drone reset to spawn position")
    
    def takeoff(self):
        """Takeoff drone from current position"""
        print("\n🛫 Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        
        # CRITICAL: Reset collision info after takeoff
        # AirSim bug: collision flag may be set when drone is on ground
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print("⚠️  Collision flag was set at spawn (AirSim bug)")
            print("   Resetting collision state...")
            # Move slightly to clear collision state
            current_state = self.client.getMultirotorState()
            pos = current_state.kinematics_estimated.position
            self.client.moveToPositionAsync(
                pos.x_val, pos.y_val, pos.z_val - 0.5, 1
            ).join()
            time.sleep(0.5)
        
        # Get position after takeoff for mission start
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        self.actual_start_position = (pos.x_val, pos.y_val, pos.z_val)
        
        # Verify no collision after takeoff
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print("⚠️  WARNING: Collision still detected after takeoff!")
        else:
            print(f"✅ Airborne at ({self.actual_start_position[0]:.2f}, "
                  f"{self.actual_start_position[1]:.2f}, "
                  f"{self.actual_start_position[2]:.2f})")
            print("✅ Collision state: CLEAR")
    
    def land(self):
        """Land drone"""
        print("\n🛬 Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("✅ Landed")
    
    def get_drone_state(self) -> Dict[str, Any]:
        """Get current drone state from AirSim"""
        state = self.client.getMultirotorState()
        
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        orientation = state.kinematics_estimated.orientation
        
        current_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
        current_vel = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        # Get yaw from quaternion
        yaw = airsim.to_eularian_angles(orientation)[2]
        
        return {
            'position': current_pos,
            'velocity': current_vel,
            'yaw': yaw
        }
    
    def check_collision(self) -> bool:
        """Check if drone has collided"""
        collision_info = self.client.simGetCollisionInfo()
        return collision_info.has_collided
    
    def visualize_path_in_ue(
        self, 
        path: List[Tuple[float, float, float]], 
        color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        thickness: float = 5.0,
        duration: float = -1.0
    ):
        """
        Draw A* path in Unreal Engine using AirSim plotting API.
        
        Args:
            path: List of waypoints (x, y, z)
            color_rgb: RGB color (0-1 range)
            thickness: Line thickness
            duration: How long to display (-1 = permanent until cleared)
        """
        if len(path) < 2:
            print("⚠️  Path too short to visualize")
            return
        
        print(f"\n🎨 Drawing path in UE ({len(path)} waypoints)...")
        
        # Convert path to AirSim Vector3r format
        vector_list = [
            airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))
            for p in path
        ]
        
        # Draw line strip
        try:
            self.client.simPlotLineStrip(
                points=vector_list,
                color_rgba=[color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
                thickness=thickness,
                duration=duration,
                is_persistent=True
            )
            print(f"✅ Path visualized in UE (red line)")
        except Exception as e:
            print(f"⚠️  Could not draw path: {e}")
            print("   (This is OK - visualization is optional)")
    
    def visualize_trajectory_in_ue(
        self,
        trajectory: List[Tuple[float, float, float]],
        color_rgb: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        thickness: float = 3.0
    ):
        """
        Draw actual trajectory in UE (what the drone actually flew).
        
        Args:
            trajectory: List of actual positions
            color_rgb: RGB color (0-1 range)
            thickness: Line thickness
        """
        if len(trajectory) < 2:
            return
        
        print(f"\n🎨 Drawing actual trajectory in UE ({len(trajectory)} points)...")
        
        vector_list = [
            airsim.Vector3r(float(p[0]), float(p[1]), float(p[2]))
            for p in trajectory
        ]
        
        try:
            self.client.simPlotLineStrip(
                points=vector_list,
                color_rgba=[color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
                thickness=thickness,
                duration=-1.0,
                is_persistent=True
            )
            print(f"✅ Trajectory visualized in UE (green line)")
        except Exception as e:
            print(f"⚠️  Could not draw trajectory: {e}")
    
    def mark_waypoints_in_ue(
        self,
        waypoints: List[Tuple[float, float, float]],
        color_rgb: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        scale: float = 0.5
    ):
        """
        Mark key waypoints with spheres in UE.
        
        Args:
            waypoints: List of waypoint positions
            color_rgb: RGB color (0-1 range)
            scale: Marker size
        """
        print(f"\n📍 Marking {len(waypoints)} waypoints in UE...")
        
        for i, wp in enumerate(waypoints):
            try:
                self.client.simPlotPoints(
                    points=[airsim.Vector3r(float(wp[0]), float(wp[1]), float(wp[2]))],
                    color_rgba=[color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
                    size=scale * 10,
                    duration=-1.0,
                    is_persistent=True
                )
            except:
                pass
    
    def clear_visualization(self):
        """Clear all visualizations in UE"""
        try:
            self.client.simFlushPersistentMarkers()
            print("🧹 Cleared previous visualizations")
        except:
            pass
    
    def run_mission(
        self,
        start_name: str = "DroneSpawn",
        target_name: str = None,
        visualize: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete A* mission from spawn to landing target.
        
        Args:
            start_name: Name of start actor (default: DroneSpawn)
            target_name: Name of target actor (e.g., "Landing_301")
                         If None, picks random target
            visualize: Whether to visualize path in UE
            
        Returns:
            Mission results dictionary
        """
        # ========================================
        # PHASE 0: Setup
        # ========================================
        if not self.spawn_position:
            print("❌ Spawn position not loaded. Call load_ue_actors() first.")
            return {}
        
        if not self.landing_targets:
            print("❌ No landing targets loaded. Call load_ue_actors() first.")
            return {}
        
        # Pick target
        if target_name is None or target_name not in self.landing_targets:
            # Random target
            target_name = random.choice(list(self.landing_targets.keys()))
            print(f"🎲 Selected random target: {target_name}")
        
        goal_position = self.landing_targets[target_name]
        
        print(f"\n{'='*70}")
        print(f"🚀 A* MISSION: {start_name} → {target_name}")
        print(f"   Spawn: ({self.spawn_position[0]:.2f}, {self.spawn_position[1]:.2f}, {self.spawn_position[2]:.2f})")
        
        # Use actual airborne position as start (after takeoff)
        mission_start_pos = self.actual_start_position if self.actual_start_position else self.spawn_position
        print(f"   Start: ({mission_start_pos[0]:.2f}, {mission_start_pos[1]:.2f}, {mission_start_pos[2]:.2f})")
        print(f"   Goal:  ({goal_position[0]:.2f}, {goal_position[1]:.2f}, {goal_position[2]:.2f})")
        print(f"{'='*70}")
        
        # Clear previous visualizations
        if visualize:
            self.clear_visualization()
        
        # ========================================
        # PHASE 1: A* Path Planning
        # ========================================
        print("\n🗺️  Planning path with hierarchical A*...")
        start_time = time.time()
        
        # Use actual airborne position as A* start point
        mission_start_pos = self.actual_start_position if self.actual_start_position else self.spawn_position
        path = self.hierarchical_planner.plan_path(mission_start_pos, goal_position)
        
        planning_time = time.time() - start_time
        
        # CRITICAL: Reset drone position after planning
        # During A* search (especially long searches), drone drifts down due to gravity
        print(f"\n📍 Resetting drone position (compensate for drift during planning)...")
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(
                    mission_start_pos[0],
                    mission_start_pos[1],
                    mission_start_pos[2]
                ),
                airsim.Quaternionr(0, 0, 0, 1)
            ),
            True
        )
        time.sleep(0.5)  # Allow physics to stabilize
        
        # Verify position after reset
        state = self.client.getMultirotorState()
        actual_pos = state.kinematics_estimated.position
        print(f"   Reset to: ({actual_pos.x_val:.2f}, {actual_pos.y_val:.2f}, {actual_pos.z_val:.2f})")
        
        if not path or len(path) == 0:
            print("❌ A* planning failed! No path found.")
            return {
                'success': False,
                'planning_time': planning_time,
                'message': 'No path found'
            }
        
        print(f"✅ Path planned: {len(path)} waypoints in {planning_time:.3f}s")
        
        # Visualize planned path
        if visualize:
            self.visualize_path_in_ue(
                path, 
                color_rgb=(1.0, 0.0, 0.0),  # Red for planned path
                thickness=5.0
            )
            # Mark spawn, start, and goal
            mission_start_pos = self.actual_start_position if self.actual_start_position else self.spawn_position
            self.mark_waypoints_in_ue(
                [mission_start_pos, goal_position],
                color_rgb=(1.0, 1.0, 0.0),  # Yellow markers
                scale=1.0
            )
        
        # ========================================
        # PHASE 2: PID Trajectory Execution
        # ========================================
        print("\n🎯 Executing path with PID controller...")
        
        # Drone is already airborne from takeoff, no need to move to start
        # Just reset PID and start following the path
        self.pid_controller.reset()
        self.astar_controller.set_path(path)
        
        # Execution parameters
        control_dt = self.config.get('control_dt', 0.05)
        max_time = self.config.get('max_episode_time', 300.0)
        goal_tolerance = self.config.get('goal_tolerance', 0.5)
        waypoint_tolerance = self.config.get('waypoint_tolerance', 1.0)
        
        # Execute
        waypoint_idx = 0
        trajectory = []
        execution_start = time.time()
        prev_vel = np.zeros(3)
        energy_consumed = 0.0
        
        while waypoint_idx < len(path):
            # Timeout check
            elapsed = time.time() - execution_start
            if elapsed > max_time:
                print(f"⏱️  Timeout after {elapsed:.1f}s")
                break
            
            # Get state
            drone_state = self.get_drone_state()
            current_pos = drone_state['position']
            current_yaw = drone_state['yaw']
            current_vel = drone_state['velocity']
            
            # Target waypoint
            target_waypoint = path[waypoint_idx]
            
            # PID control
            vx, vy, vz, yaw_rate = self.pid_controller.compute_control(
                current_pos=tuple(current_pos),
                current_yaw=current_yaw,
                target_pos=target_waypoint,
                target_yaw=0.0
            )
            
            # Execute
            self.client.moveByVelocityAsync(
                vx, vy, vz,
                control_dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_rate)
            )
            
            time.sleep(control_dt)
            
            # Log trajectory
            trajectory.append(current_pos.copy())
            
            # Energy calculation
            velocity_mag = np.linalg.norm(current_vel)
            acceleration = np.linalg.norm(current_vel - prev_vel) / control_dt
            kinetic_energy = 0.5 * self.config.get('drone_mass', 1.5) * velocity_mag**2
            accel_energy = self.config.get('drone_mass', 1.5) * acceleration * velocity_mag * control_dt
            energy_consumed += (kinetic_energy + accel_energy) * control_dt
            prev_vel = current_vel.copy()
            
            # Collision check
            if self.check_collision():
                print("❌ Collision detected!")
                break
            
            # Waypoint reached?
            dist_to_waypoint = np.linalg.norm(current_pos - np.array(target_waypoint))
            if dist_to_waypoint < waypoint_tolerance:
                waypoint_idx += 1
                if waypoint_idx < len(path):
                    progress = waypoint_idx / len(path) * 100
                    print(f"  ✓ Waypoint {waypoint_idx}/{len(path)} ({progress:.0f}%)")
        
        execution_time = time.time() - execution_start
        
        # Visualize actual trajectory
        if visualize and len(trajectory) > 1:
            self.visualize_trajectory_in_ue(
                trajectory,
                color_rgb=(0.0, 1.0, 0.0),  # Green for actual trajectory
                thickness=3.0
            )
        
        # ========================================
        # PHASE 3: Results
        # ========================================
        final_pos = trajectory[-1] if len(trajectory) > 0 else np.array(self.spawn_position)
        final_distance = float(np.linalg.norm(final_pos - np.array(goal_position)))
        
        # Path length
        path_length = 0.0
        if len(trajectory) > 1:
            path_length = sum(
                np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i]))
                for i in range(len(trajectory)-1)
            )
        
        # ATE (Average Trajectory Error)
        ate_error = 0.0
        if len(trajectory) > 0 and len(path) > 0:
            ate_errors = []
            for traj_point in trajectory:
                min_dist = min(
                    np.linalg.norm(np.array(traj_point) - np.array(wp))
                    for wp in path
                )
                ate_errors.append(min_dist)
            ate_error = float(np.mean(ate_errors))
        
        success = (final_distance < goal_tolerance)
        
        results = {
            'success': success,
            'start': start_name,
            'target': target_name,
            'spawn_position': self.spawn_position,
            'mission_start_position': self.actual_start_position,
            'planning_time': planning_time,
            'execution_time': execution_time,
            'total_time': planning_time + execution_time,
            'path_waypoints': len(path),
            'path_length': path_length,
            'energy_consumed': energy_consumed / 1000.0,  # kJ
            'final_distance_to_goal': final_distance,
            'ate_error': ate_error,
            'trajectory_points': len(trajectory)
        }
        
        # Print results
        battery_capacity = self.config.get('battery_capacity', 1.0)
        battery_used_percent = (results['energy_consumed'] / battery_capacity) * 100
        battery_remaining = battery_capacity - results['energy_consumed']
        
        print(f"\n{'='*70}")
        print(f"📈 MISSION RESULTS")
        print(f"{'='*70}")
        print(f"✓ Success: {'YES' if success else 'NO'}")
        print(f"⏱️  Planning time: {planning_time:.3f}s")
        print(f"⏱️  Execution time: {execution_time:.2f}s")
        print(f"⏱️  Total time: {results['total_time']:.2f}s")
        print(f"🗺️  Path waypoints: {len(path)}")
        print(f"📏 Path length: {path_length:.2f} m")
        print(f"⚡ Energy consumed: {results['energy_consumed']:.2f} kJ ({battery_used_percent:.1f}% of battery)")
        print(f"🔋 Battery remaining: {battery_remaining:.2f} kJ ({100-battery_used_percent:.1f}%)")
        print(f"🎯 ATE error: {ate_error:.3f} m")
        print(f"📍 Distance to goal: {final_distance:.3f} m")
        print(f"{'='*70}")
        
        return results
    
    def run_multiple_missions(self, num_missions: int = 5, visualize: bool = True):
        """
        Run multiple missions to different targets for evaluation.
        
        Args:
            num_missions: Number of missions to run
            visualize: Whether to visualize paths
            
        Returns:
            List of mission results
        """
        if not self.landing_targets:
            print("❌ No landing targets loaded")
            return []
        
        print(f"\n{'='*70}")
        print(f"🚀 RUNNING {num_missions} A* MISSIONS")
        print(f"{'='*70}")
        
        # Select random targets
        available_targets = list(self.landing_targets.keys())
        selected_targets = random.sample(
            available_targets, 
            min(num_missions, len(available_targets))
        )
        
        results = []
        
        for i, target in enumerate(selected_targets):
            print(f"\n\n{'='*70}")
            print(f"Mission {i+1}/{len(selected_targets)}: {target}")
            print(f"{'='*70}")
            
            result = self.run_mission(
                start_name="DroneSpawn",
                target_name=target,
                visualize=visualize
            )
            
            results.append(result)
            
            # Brief pause between missions
            time.sleep(2)
        
        # Aggregate statistics
        self._print_aggregate_stats(results)
        
        return results
    
    def _print_aggregate_stats(self, results: List[Dict[str, Any]]):
        """Print aggregate statistics from multiple missions"""
        if not results:
            return
        
        success_count = sum(1 for r in results if r.get('success', False))
        success_rate = success_count / len(results) * 100
        
        planning_times = [r['planning_time'] for r in results]
        execution_times = [r['execution_time'] for r in results]
        energies = [r['energy_consumed'] for r in results]
        ate_errors = [r['ate_error'] for r in results]
        
        print(f"\n\n{'='*70}")
        print(f"📊 AGGREGATE STATISTICS ({len(results)} missions)")
        print(f"{'='*70}")
        print(f"Success rate: {success_rate:.1f}% ({success_count}/{len(results)})")
        print(f"\nPlanning time:")
        print(f"  Mean: {np.mean(planning_times):.3f}s ± {np.std(planning_times):.3f}s")
        print(f"  Range: [{np.min(planning_times):.3f}s, {np.max(planning_times):.3f}s]")
        print(f"\nExecution time:")
        print(f"  Mean: {np.mean(execution_times):.2f}s ± {np.std(execution_times):.2f}s")
        print(f"  Range: [{np.min(execution_times):.2f}s, {np.max(execution_times):.2f}s]")
        print(f"\nEnergy consumption:")
        print(f"  Mean: {np.mean(energies):.2f} kJ ± {np.std(energies):.2f} kJ")
        print(f"\nTracking accuracy (ATE):")
        print(f"  Mean: {np.mean(ate_errors):.3f} m ± {np.std(ate_errors):.3f} m")
        print(f"{'='*70}")


def main():
    """Main visualization script"""
    
    print("="*70)
    print("🎨 A* PATH VISUALIZATION IN UNREAL ENGINE")
    print("="*70)
    
    # Configuration
    config = {
        # A* parameters (ULTRA-AGGRESSIVE for demo speed)
        'floor_penalty': 5.0,
        'heuristic_weight': 5.0,  # Ultra-greedy A* for maximum speed (1.0=optimal, 3.0=ultra-fast)
        
        # PID parameters (PhD-level tuning for altitude stability)
        'position_kp': 2.5,      # Increased for faster response
        'position_ki': 0.5,      # 5x increase for better altitude hold (gravity compensation)
        'position_kd': 0.8,      # Increased damping to reduce oscillation
        'yaw_kp': 1.5,
        'yaw_ki': 0.05,
        'yaw_kd': 0.3,
        'max_velocity': 5.0,
        'max_yaw_rate': 1.0,
        'integral_limit': 15.0,  # Increased for stronger integral action
        
        # Z-axis specific gains (critical for anti-gravity)
        'z_feedforward': 9.81,   # Gravity compensation (m/s²)
        'z_ki_boost': 2.0,       # Extra Ki multiplier for Z-axis only
        
        # Execution parameters
        'control_dt': 0.02,      # 50Hz control (was 20Hz) for faster reaction
        'max_episode_time': 300.0,
        'goal_tolerance': 0.5,
        'waypoint_tolerance': 1.0,
        'drone_mass': 1.5,       # kg
        'battery_capacity': 50.0, # 50x original capacity (kJ)
    }
    
    # Map file (must exist)
    map_file = "data/maps/building_5floors_metadata.json"
    
    if not Path(map_file).exists():
        print(f"\n❌ Map file not found: {map_file}")
        print("Please generate map first:")
        print("  python src/environment/airsim_navigation.py")
        return
    
    # Create visualizer
    visualizer = AStarPathVisualizer(map_file, config)
    
    try:
        # Connect to AirSim
        visualizer.connect_airsim()
        
        # Load actor positions from UE
        if not visualizer.load_ue_actors():
            print("\n❌ Failed to load UE actors. Exiting.")
            return
        
        # Prepare drone and reset to consistent start position
        visualizer.prepare_drone()
        visualizer.reset_to_spawn()  # Reset to DroneSpawn before takeoff
        visualizer.takeoff()
        
        # ========================================
        # OPTION 1: Single mission to specific target
        # ========================================
        # Test hierarchical planner with previously slow target
        result = visualizer.run_mission(
            start_name="DroneSpawn",
            target_name="Landing_201",  # Was taking 65+ seconds with direct A*
            visualize=True
        )
        
        # ========================================
        # OPTION 2: Multiple missions (random targets)
        # ========================================
        # results = visualizer.run_multiple_missions(
        #     num_missions=3,  # Change this number
        #     visualize=True
        # )
        
        # # Land
        # visualizer.land()
        
        # print("\n✅ Visualization completed!")
        # print("\n💡 Tips:")
        # print("  - Red lines: Planned A* path")
        # print("  - Green lines: Actual drone trajectory")
        # print("  - Yellow spheres: Start and goal positions")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        visualizer.land()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            visualizer.land()
        except:
            pass


if __name__ == "__main__":
    main()
