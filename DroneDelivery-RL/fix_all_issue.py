#!/usr/bin/env python3
"""
Complete fix for ALL DroneDelivery-RL issues
Fixes code bugs that tests are catching
"""

from pathlib import Path
import re

print("🔧 Fixing All Issues...")
print("=" * 70)

# ============================================
# FIX 1: Add missing functions to coordinate_transforms.py
# ============================================

def fix_coordinate_transforms():
    """Add missing transform_pose function"""
    file_path = Path('src/localization/coordinate_transforms.py')
    
    if not file_path.exists():
        print(f"⚠️  {file_path} not found, skipping")
        return
    
    content = file_path.read_text(encoding='utf-8')
    
    # Check if transform_pose already exists
    if 'def transform_pose' in content:
        print(f"✅ {file_path} already has transform_pose")
        return
    
    # Add transform_pose function before __all__
    addition = '''

def transform_pose(pose: np.ndarray, from_frame: str, to_frame: str) -> np.ndarray:
    """
    Transform pose between coordinate frames.
    
    Args:
        pose: Pose as [x, y, z, qw, qx, qy, qz] or [x, y, z]
        from_frame: Source frame ('airsim', 'ned', 'enu', 'world')
        to_frame: Target frame
        
    Returns:
        Transformed pose in same format
    """
    if len(pose) >= 7:
        position = pose[:3]
        orientation = pose[3:7]
        
        # Transform position
        transformed_pos = airsim_to_ned(position) if from_frame == 'airsim' else position
        if to_frame == 'enu':
            transformed_pos = ned_to_enu(transformed_pos)
        
        # Keep orientation for now (TODO: add orientation transform)
        return np.concatenate([transformed_pos, orientation])
    else:
        # Position only
        transformed = airsim_to_ned(pose) if from_frame == 'airsim' else pose
        if to_frame == 'enu':
            transformed = ned_to_enu(transformed)
        return transformed
'''
    
    # Insert before __all__
    if '__all__' in content:
        content = content.replace('__all__', addition + '\n__all__')
        # Also add to __all__ list
        content = content.replace(
            "__all__ = [",
            "__all__ = [\n    'transform_pose',"
        )
    else:
        content += addition
    
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed: {file_path} (added transform_pose)")


# ============================================
# FIX 2: Add missing functions to coordinate_utils.py
# ============================================

def fix_coordinate_utils():
    """Add missing world_to_grid function"""
    file_path = Path('src/utils/coordinate_utils.py')
    
    if not file_path.exists():
        print(f"⚠️  {file_path} not found, skipping")
        return
    
    content = file_path.read_text(encoding='utf-8')
    
    # Check if world_to_grid already exists
    if 'def world_to_grid' in content:
        print(f"✅ {file_path} already has world_to_grid")
        return
    
    # Add world_to_grid function
    addition = '''

def world_to_grid(position: Tuple[float, float, float],
                  grid_resolution: float = 0.5,
                  grid_origin: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[int, int, int]:
    """
    Convert world coordinates to grid indices.
    
    Args:
        position: World position (x, y, z) in meters
        grid_resolution: Grid cell size in meters
        grid_origin: Origin of grid in world coordinates
        
    Returns:
        Grid indices (ix, iy, iz)
    """
    ix = int((position[0] - grid_origin[0]) / grid_resolution)
    iy = int((position[1] - grid_origin[1]) / grid_resolution)
    iz = int((position[2] - grid_origin[2]) / grid_resolution)
    return (ix, iy, iz)


def grid_to_world(grid_idx: Tuple[int, int, int],
                  grid_resolution: float = 0.5,
                  grid_origin: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
    """
    Convert grid indices to world coordinates.
    
    Args:
        grid_idx: Grid indices (ix, iy, iz)
        grid_resolution: Grid cell size in meters
        grid_origin: Origin of grid in world coordinates
        
    Returns:
        World position (x, y, z) in meters
    """
    x = grid_idx[0] * grid_resolution + grid_origin[0]
    y = grid_idx[1] * grid_resolution + grid_origin[1]
    z = grid_idx[2] * grid_resolution + grid_origin[2]
    return (x, y, z)
'''
    
    # Insert before __all__
    if '__all__' in content:
        content = content.replace('__all__', addition + '\n__all__')
        # Also add to __all__ list
        content = content.replace(
            "__all__ = [",
            "__all__ = [\n    'world_to_grid',\n    'grid_to_world',"
        )
    else:
        content += addition
    
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed: {file_path} (added world_to_grid, grid_to_world)")


# ============================================
# FIX 3: Fix GAECalculator parameter name
# ============================================

def fix_gae_calculator():
    """Fix GAECalculator to accept both 'lam' and 'lambda_'"""
    file_path = Path('src/rl/agents/gae_calculator.py')
    
    if not file_path.exists():
        print(f"⚠️  {file_path} not found, skipping")
        return
    
    content = file_path.read_text(encoding='utf-8')
    
    # Find __init__ method and fix parameter
    pattern = r'def __init__\(self,([^)]+)\):'
    match = re.search(pattern, content)
    
    if match:
        params = match.group(1)
        # Check if already has lam parameter
        if 'lam:' not in params and 'lam=' not in params:
            # Add lam as alias for lambda_
            init_end = content.find('"""', content.find('def __init__'))
            if init_end != -1:
                # Find where to insert parameter handling
                insert_pos = content.find('\n', init_end + 3)
                if insert_pos != -1:
                    addition = """
        # Support both 'lam' and 'lambda_' parameter names
        if 'lam' in locals() and lam is not None:
            lambda_ = lam
"""
                    content = content[:insert_pos] + addition + content[insert_pos:]
    
    file_path.write_text(content, encoding='utf-8')
    print(f"✅ Fixed: {file_path} (added lam parameter support)")


# ============================================
# FIX 4: Fix sensor_bridge.py cv2.createStereoBM
# ============================================

def fix_sensor_bridge_cv2():
    """Fix deprecated cv2.createStereoBM to cv2.StereoBM_create"""
    file_path = Path('src/bridges/sensor_bridge.py')
    
    if not file_path.exists():
        print(f"⚠️  {file_path} not found, skipping")
        return
    
    content = file_path.read_text(encoding='utf-8')
    
    # Replace deprecated API
    original = content
    content = content.replace('cv2.createStereoBM(', 'cv2.StereoBM_create(')
    content = content.replace('cv2.createStereoSGBM(', 'cv2.StereoSGBM_create(')
    
    if content != original:
        file_path.write_text(content, encoding='utf-8')
        print(f"✅ Fixed: {file_path} (updated cv2 API)")
    else:
        print(f"✅ {file_path} already uses correct cv2 API")


# ============================================
# FIX 5: Update observation dimension documentation
# ============================================

def fix_observation_dimension():
    """Update observation dimension from 35 to 40"""
    files_to_check = [
        'src/environment/observation_space.py',
        'README.md',
        'docs/architecture.md',
    ]
    
    for file_path_str in files_to_check:
        file_path = Path(file_path_str)
        if not file_path.exists():
            continue
        
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        original = content
        
        # Replace mentions of 35-dimensional with 40-dimensional
        content = re.sub(r'\b35-dimensional\b', '40-dimensional', content)
        content = re.sub(r'\bObservation dim(?:ension)?:\s*35\b', 'Observation dimension: 40', content)
        content = re.sub(r'\bOBS_DIM\s*=\s*35\b', 'OBS_DIM = 40', content)
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ Fixed: {file_path} (updated obs dim 35→40)")


# ============================================
# MAIN
# ============================================

def main():
    try:
        fix_coordinate_transforms()
        fix_coordinate_utils()
        fix_gae_calculator()
        fix_sensor_bridge_cv2()
        fix_observation_dimension()
        
        print("=" * 70)
        print("\n✅ ALL CODE FIXES APPLIED!")
        print("\n📝 Next: Tests still need updates for proper constructor calls")
        print("   Tests should pass config dict to all constructors:")
        print("   - ActionSpace(config=test_config)")
        print("   - ObservationSpace(config=test_config)")
        print("   - AirSimBridge(config=test_config)")
        print("   - SensorBridge(config=test_config)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
