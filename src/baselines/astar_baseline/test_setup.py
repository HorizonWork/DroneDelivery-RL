"""
Test script to verify A* baseline setup
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.baselines.astar_baseline.astar_controller import AStarController
        print("  ✓ AStarController")
    except Exception as e:
        print(f"  ✗ AStarController: {e}")
        return False
    
    try:
        from src.baselines.astar_baseline.pid_controller import PIDController
        print("  ✓ PIDController")
    except Exception as e:
        print(f"  ✗ PIDController: {e}")
        return False
    
    try:
        from src.baselines.astar_baseline.evaluator import AStarEvaluator
        print("  ✓ AStarEvaluator")
    except Exception as e:
        print(f"  ✗ AStarEvaluator: {e}")
        return False
    
    try:
        from src.environment.airsim_navigation import MapGenerator
        print("  ✓ MapGenerator")
    except Exception as e:
        print(f"  ✗ MapGenerator: {e}")
        return False
    
    return True

def test_astar_without_map():
    """Test A* with config-based initialization"""
    print("\nTesting A* controller (config-based)...")
    
    from src.baselines.astar_baseline.astar_controller import AStarController
    
    config = {
        'floors': 5,
        'floor_length': 20.0,
        'floor_width': 40.0,
        'floor_height': 3.0,
        'cell_size': 0.5,
        'floor_penalty': 5.0
    }
    
    try:
        controller = AStarController(config)
        print(f"  ✓ Grid size: {controller.grid_x}×{controller.grid_y}×{controller.grid_z}")
        
        # Test path planning
        start = (1.0, 1.0, 1.0)
        goal = (18.0, 38.0, 13.0)
        
        path = controller.plan_path(start, goal)
        
        if path:
            print(f"  ✓ Path found: {len(path)} waypoints")
            return True
        else:
            print(f"  ✗ Path planning failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_astar_with_map():
    """Test A* with map file"""
    print("\nTesting A* controller (map-based)...")
    
    from src.baselines.astar_baseline.astar_controller import AStarController
    
    map_file = "data/maps/building_5floors_metadata.json"
    
    if not Path(map_file).exists():
        print(f"  ⚠️  Map file not found: {map_file}")
        print(f"  → Run: python src/environment/airsim_navigation.py")
        return None
    
    config = {'floor_penalty': 5.0}
    
    try:
        controller = AStarController(config, map_file=map_file)
        print(f"  ✓ Loaded map: {controller.grid_x}×{controller.grid_y}×{controller.grid_z}")
        print(f"  ✓ Bounds: X[{controller.world_bounds[0,0]:.1f}, {controller.world_bounds[0,1]:.1f}]")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pid():
    """Test PID controller"""
    print("\nTesting PID controller...")
    
    from src.baselines.astar_baseline.pid_controller import PIDController
    
    config = {
        'position_kp': 2.0,
        'position_ki': 0.1,
        'position_kd': 0.5,
        'yaw_kp': 1.5,
        'yaw_ki': 0.05,
        'yaw_kd': 0.3,
        'max_velocity': 5.0,
        'max_yaw_rate': 1.0,
        'integral_limit': 10.0
    }
    
    try:
        controller = PIDController(config)
        print(f"  ✓ PID initialized")
        
        # Test control computation
        current_pos = (0.0, 0.0, -2.0)
        target_pos = (5.0, 5.0, -5.0)
        
        vx, vy, vz, yaw_rate = controller.compute_control(
            current_pos, 0.0, target_pos, 0.0
        )
        
        print(f"  ✓ Control output: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw={yaw_rate:.2f}")
        
        # Check velocity limits
        import numpy as np
        vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        if vel_mag <= config['max_velocity']:
            print(f"  ✓ Velocity within limits: {vel_mag:.2f} m/s")
            return True
        else:
            print(f"  ✗ Velocity exceeds limit: {vel_mag:.2f} > {config['max_velocity']}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_airsim_connection():
    """Test AirSim connection (optional)"""
    print("\nTesting AirSim connection...")
    
    try:
        import airsim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("  ✓ Connected to AirSim")
        return True
    except Exception as e:
        print(f"  ⚠️  Cannot connect to AirSim: {e}")
        print(f"  → Make sure AirSim/UE is running")
        return None

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 A* Baseline Setup Tests")
    print("="*60)
    
    results = {
        'Imports': test_imports(),
        'A* (config)': test_astar_without_map(),
        'A* (map)': test_astar_with_map(),
        'PID': test_pid(),
        'AirSim': test_airsim_connection()
    }
    
    print("\n" + "="*60)
    print("📊 Test Results")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"{test_name:20s} {status}")
    
    print("="*60)
    
    # Overall status
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    
    if failed == 0:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Generate map: python src/environment/airsim_navigation.py")
        print("2. Run evaluation: python src/baselines/astar_baseline/run_airsim_evaluation.py")
    else:
        print(f"❌ {failed} test(s) failed")
        print("Please fix errors before proceeding")

if __name__ == "__main__":
    main()
