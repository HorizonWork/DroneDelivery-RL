"""
A* Baseline Evaluation with AirSim
Evaluates A* + PID performance using map from AirSim environment.
"""

import airsim
import numpy as np
import time
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.baselines.astar_baseline.astar_controller import AStarController
from src.baselines.astar_baseline.pid_controller import PIDController
from src.baselines.astar_baseline.evaluator import AStarEvaluator, EpisodeResult


class AirSimBaselineRunner:
    """
    Runs A* + PID baseline evaluation in AirSim.
    Uses pre-generated map from airsim_navigation.py
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
        self.pid_controller = PIDController(config)
        
        # Evaluator
        self.evaluator = AStarEvaluator(config)
        
        print("✅ Baseline runner initialized")
        print(f"   Map file: {map_file}")
    
    def connect_airsim(self):
        """Connect to AirSim simulator"""
        print("\n🔗 Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ Connected to AirSim")
        
    def prepare_drone(self):
        """Enable control and arm drone"""
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def takeoff(self):
        """Takeoff drone"""
        print("🛫 Taking off...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        print("✅ Airborne")
    
    def land(self):
        """Land drone"""
        print("🛬 Landing...")
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
        
        # Convert to standard coordinates
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
    
    def run_episode(
        self, 
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        episode_num: int = 1
    ) -> EpisodeResult:
        """
        Run single evaluation episode.
        
        Args:
            start_pos: Starting position (x, y, z) in meters
            goal_pos: Goal position (x, y, z) in meters
            episode_num: Episode number for logging
            
        Returns:
            EpisodeResult with metrics
        """
        print(f"\n{'='*70}")
        print(f"📊 EPISODE {episode_num}")
        print(f"   Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f})")
        print(f"   Goal:  ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}, {goal_pos[2]:.1f})")
        print(f"{'='*70}")
        
        result = EpisodeResult()
        start_time = time.time()
        
        # Reset drone to start position
        print(f"📍 Moving to start position...")
        self.client.simSetVehiclePose(
            airsim.Pose(
                airsim.Vector3r(start_pos[0], start_pos[1], start_pos[2]),
                airsim.Quaternionr(0, 0, 0, 1)
            ),
            True
        )
        time.sleep(2)
        
        # Reset PID controller
        self.pid_controller.reset()
        
        # PHASE 1: A* Planning
        print("\n🗺️  Planning path with A*...")
        path = self.astar_controller.plan_path(start_pos, goal_pos)
        
        if not path or len(path) == 0:
            print("❌ A* planning failed!")
            result.success = False
            result.timeout = True
            result.flight_time = time.time() - start_time
            return result
        
        print(f"✅ Path planned: {len(path)} waypoints")
        self.astar_controller.set_path(path)
        
        # PHASE 2: PID Trajectory Tracking
        print("\n🎯 Executing path with PID controller...")
        
        waypoint_idx = 0
        trajectory = []
        energy_consumed = 0.0
        
        control_dt = self.config.get('control_dt', 0.05)
        max_time = self.config.get('max_episode_time', 300.0)
        goal_tolerance = self.config.get('goal_tolerance', 0.5)
        waypoint_tolerance = self.config.get('waypoint_tolerance', 1.0)
        
        prev_vel = np.zeros(3)
        
        while waypoint_idx < len(path):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_time:
                print(f"⏱️  Timeout after {elapsed:.1f}s")
                result.timeout = True
                break
            
            # Get current state
            drone_state = self.get_drone_state()
            current_pos = drone_state['position']
            current_yaw = drone_state['yaw']
            current_vel = drone_state['velocity']
            
            # Get target waypoint
            target_waypoint = path[waypoint_idx]
            
            # Compute PID control
            vx, vy, vz, yaw_rate = self.pid_controller.compute_control(
                current_pos=tuple(current_pos),
                current_yaw=current_yaw,
                target_pos=target_waypoint,
                target_yaw=0.0
            )
            
            # Execute velocity command
            self.client.moveByVelocityAsync(
                vx, vy, vz,
                control_dt,
                airsim.DrivetrainType.MaxDegreeOfFreedom,
                airsim.YawMode(False, yaw_rate)
            )
            
            time.sleep(control_dt)
            
            # Log trajectory
            trajectory.append(current_pos.copy())
            
            # Compute energy (kinetic energy + acceleration cost)
            velocity_mag = np.linalg.norm(current_vel)
            acceleration = np.linalg.norm(current_vel - prev_vel) / control_dt
            
            # Energy model: E = 0.5*m*v^2 + m*a*d
            kinetic_energy = 0.5 * self.config.get('drone_mass', 1.5) * velocity_mag**2
            accel_energy = self.config.get('drone_mass', 1.5) * acceleration * velocity_mag * control_dt
            energy_consumed += (kinetic_energy + accel_energy) * control_dt
            
            prev_vel = current_vel.copy()
            
            # Check collision
            if self.check_collision():
                print("❌ Collision detected!")
                result.collision = True
                break
            
            # Check waypoint reached
            dist_to_waypoint = np.linalg.norm(current_pos - np.array(target_waypoint))
            if dist_to_waypoint < waypoint_tolerance:
                waypoint_idx += 1
                if waypoint_idx < len(path):
                    progress = waypoint_idx / len(path) * 100
                    print(f"  ✓ Waypoint {waypoint_idx}/{len(path)} ({progress:.0f}%)")
        
        # Compute final metrics
        result.flight_time = time.time() - start_time
        result.energy_consumed = energy_consumed / 1000.0  # Convert to kJ
        
        # Path length
        if len(trajectory) > 1:
            result.path_length = sum(
                np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i]))
                for i in range(len(trajectory)-1)
            )
        
        # Compute ATE (Average Trajectory Error)
        if len(trajectory) > 0 and len(path) > 0:
            ate_errors = []
            for traj_point in trajectory:
                min_dist = min(
                    np.linalg.norm(np.array(traj_point) - np.array(wp))
                    for wp in path
                )
                ate_errors.append(min_dist)
            result.ate_error = float(np.mean(ate_errors))
        
        # Check success
        final_pos = trajectory[-1] if len(trajectory) > 0 else np.array(start_pos)
        result.final_distance_to_goal = float(np.linalg.norm(final_pos - np.array(goal_pos)))
        result.success = (
            result.final_distance_to_goal < goal_tolerance 
            and not result.collision 
            and not result.timeout
        )
        
        # Print episode summary
        print(f"\n{'='*70}")
        print(f"📈 Episode {episode_num} Results:")
        print(f"   ✓ Success: {'YES' if result.success else 'NO'}")
        print(f"   ⏱️  Time: {result.flight_time:.2f}s")
        print(f"   ⚡ Energy: {result.energy_consumed:.2f} kJ")
        print(f"   📏 Path length: {result.path_length:.2f} m")
        print(f"   🎯 ATE: {result.ate_error:.3f} m")
        print(f"   📍 Distance to goal: {result.final_distance_to_goal:.3f} m")
        if result.collision:
            print(f"   ⚠️  COLLISION")
        if result.timeout:
            print(f"   ⏱️  TIMEOUT")
        print(f"{'='*70}")
        
        return result
    
    def generate_test_scenarios(
        self, 
        num_episodes: int
    ) -> list[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Generate test start/goal pairs based on REAL building bounds from map.
        NO hard-coded dimensions!
        
        Args:
            num_episodes: Number of test scenarios to generate
            
        Returns:
            List of (start_pos, goal_pos) tuples
        """
        print(f"\n🎲 Generating {num_episodes} random test scenarios...")
        
        scenarios = []
        
        # Get bounds from map (loaded via AStarController)
        bounds = self.astar_controller.world_bounds
        
        print(f"   Using map bounds:")
        print(f"   X: [{bounds[0, 0]:.1f}, {bounds[0, 1]:.1f}] m")
        print(f"   Y: [{bounds[1, 0]:.1f}, {bounds[1, 1]:.1f}] m")
        print(f"   Z: [{bounds[2, 0]:.1f}, {bounds[2, 1]:.1f}] m")
        
        # Safety margin to avoid spawning in walls
        margin = 2.0  # meters
        
        np.random.seed(42)  # Reproducible scenarios for research
        
        for i in range(num_episodes):
            # Random start (lower 30% of building height)
            z_range = bounds[2, 1] - bounds[2, 0]
            start = (
                np.random.uniform(bounds[0, 0] + margin, bounds[0, 1] - margin),
                np.random.uniform(bounds[1, 0] + margin, bounds[1, 1] - margin),
                np.random.uniform(bounds[2, 0] + 1, bounds[2, 0] + z_range * 0.3)
            )
            
            # Random goal (upper 30% of building height)
            goal = (
                np.random.uniform(bounds[0, 0] + margin, bounds[0, 1] - margin),
                np.random.uniform(bounds[1, 0] + margin, bounds[1, 1] - margin),
                np.random.uniform(bounds[2, 0] + z_range * 0.7, bounds[2, 1] - 1)
            )
            
            scenarios.append((start, goal))
        
        print(f"✅ Generated {len(scenarios)} scenarios (seed=42 for reproducibility)")
        
        return scenarios
    
    def run_evaluation(self, num_episodes: int = 10):
        """
        Run full evaluation campaign.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            EvaluationMetrics
        """
        print("\n" + "="*70)
        print(f"🚀 A* + PID BASELINE EVALUATION")
        print(f"   Episodes: {num_episodes}")
        print(f"   Map: {self.map_file}")
        print("="*70)
        
        # Generate test scenarios
        scenarios = self.generate_test_scenarios(num_episodes)
        
        # Run episodes
        for i, (start, goal) in enumerate(scenarios):
            try:
                result = self.run_episode(start, goal, episode_num=i+1)
                self.evaluator.episode_results.append(result)
            except Exception as e:
                print(f"❌ Episode {i+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute aggregate metrics
        metrics = self.evaluator._compute_metrics()
        
        # Print final results
        print("\n" + "="*70)
        print("📊 FINAL EVALUATION RESULTS")
        print("="*70)
        print(f"Episodes completed: {metrics.episodes_completed}/{num_episodes}")
        print(f"\n🎯 Success Metrics:")
        print(f"   Success rate: {metrics.success_rate:.1f}% (target: {metrics.target_success_rate:.1f}%)")
        print(f"   Collision rate: {metrics.collision_rate:.1f}% (target: <{metrics.target_collision_rate:.1f}%)")
        print(f"   Timeout rate: {metrics.timeout_rate:.1f}%")
        
        print(f"\n⚡ Energy Consumption:")
        print(f"   Mean: {metrics.mean_energy:.1f} ± {metrics.std_energy:.1f} kJ")
        print(f"   Target: {metrics.target_energy_mean:.1f} kJ")
        
        print(f"\n⏱️  Flight Time:")
        print(f"   Mean: {metrics.mean_time:.1f} ± {metrics.std_time:.1f} s")
        print(f"   Target: {metrics.target_time_mean:.1f} s")
        
        print(f"\n📏 Tracking Accuracy (ATE):")
        print(f"   Mean: {metrics.mean_ate:.3f} ± {metrics.std_ate:.3f} m")
        print(f"   Target: {metrics.target_ate_mean:.3f} m")
        print("="*70)
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _save_results(self, metrics):
        """Save evaluation results to file"""
        results_dir = Path("results/baseline_evaluation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"astar_baseline_{timestamp}.json"
        
        # Convert metrics to dict
        results = {
            'timestamp': timestamp,
            'map_file': self.map_file,
            'config': self.config,
            'metrics': {
                'success_rate': metrics.success_rate,
                'collision_rate': metrics.collision_rate,
                'timeout_rate': metrics.timeout_rate,
                'mean_energy': metrics.mean_energy,
                'std_energy': metrics.std_energy,
                'mean_time': metrics.mean_time,
                'std_time': metrics.std_time,
                'mean_ate': metrics.mean_ate,
                'std_ate': metrics.std_ate,
                'episodes_completed': metrics.episodes_completed
            },
            'episodes': [
                {
                    'success': r.success,
                    'collision': r.collision,
                    'timeout': r.timeout,
                    'energy': r.energy_consumed,
                    'time': r.flight_time,
                    'ate': r.ate_error,
                    'path_length': r.path_length
                }
                for r in self.evaluator.episode_results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main evaluation script - Research compliant version"""
    
    # Configuration - ONLY control & evaluation parameters
    # Building dimensions come from map scanning, NOT config
    config = {
        # ==========================================
        # A* ALGORITHM PARAMETERS
        # ==========================================
        'floor_penalty': 5.0,  # Cost penalty for floor transitions
        
        # ==========================================
        # PID CONTROL PARAMETERS (tuned for stability)
        # ==========================================
        'position_kp': 2.0,
        'position_ki': 0.1,
        'position_kd': 0.5,
        'yaw_kp': 1.5,
        'yaw_ki': 0.05,
        'yaw_kd': 0.3,
        'max_velocity': 5.0,
        'max_yaw_rate': 1.0,
        'integral_limit': 10.0,
        
        # ==========================================
        # EVALUATION PARAMETERS
        # ==========================================
        'control_dt': 0.05,  # 20 Hz control loop
        'max_episode_time': 300.0,  # 5 minutes max
        'goal_tolerance': 0.5,  # meters
        'waypoint_tolerance': 1.0,  # meters
        'drone_mass': 1.5,  # kg (for energy calculation)
        
        # ==========================================
        # TARGET METRICS (Table 3 from paper)
        # ==========================================
        'num_episodes': 10,
        'target_success_rate': 96.0,
        'target_energy_mean': 820.0,
        'target_time_mean': 32.0,
        'target_collision_rate': 1.2,
        'target_ate_mean': 0.11
    }
    
    # Map file (must be generated first with airsim_navigation.py)
    map_file = "data/maps/building_5floors_metadata.json"
    
    # Check if map exists
    if not Path(map_file).exists():
        print(f"❌ Map file not found: {map_file}")
        print("Please generate map first using:")
        print("  python src/environment/airsim_navigation.py")
        return
    
    # ==========================================
    # LOAD MAP METADATA (building info from AirSim scan)
    # ==========================================
    print("\n" + "="*70)
    print("📂 Loading building information from map...")
    print("="*70)
    
    with open(map_file, 'r') as f:
        map_metadata = json.load(f)
    
    building_bounds = map_metadata['bounds']
    grid_dims = map_metadata['dimensions']
    cell_size = map_metadata['resolution']
    occupied_cells = map_metadata['occupied_cells']
    
    print(f"✅ Building bounds:")
    print(f"   X: [{building_bounds['x_min']:.1f}, {building_bounds['x_max']:.1f}] m")
    print(f"   Y: [{building_bounds['y_min']:.1f}, {building_bounds['y_max']:.1f}] m")
    print(f"   Z: [{building_bounds['z_min']:.1f}, {building_bounds['z_max']:.1f}] m")
    print(f"✅ Grid dimensions: {grid_dims['x']}×{grid_dims['y']}×{grid_dims['z']} cells")
    print(f"✅ Resolution: {cell_size} m/cell")
    print(f"✅ Occupied cells: {occupied_cells:,}")
    print("="*70)
    
    # Create runner
    runner = AirSimBaselineRunner(map_file, config)
    
    try:
        # Connect to AirSim
        runner.connect_airsim()
        runner.prepare_drone()
        runner.takeoff()
        
        # Run evaluation
        metrics = runner.run_evaluation(num_episodes=config['num_episodes'])
        
        # Land
        runner.land()
        
        print("\n✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        runner.land()
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            runner.land()
        except:
            pass


if __name__ == "__main__":
    main()
