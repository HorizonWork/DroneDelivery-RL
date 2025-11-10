#!/usr/bin/env python3
"""
COMPLETE FIX - Fix all 5 remaining issues
"""

from pathlib import Path
import re

print("🔧 COMPLETE FIX - Fixing all 5 issues...")
print("=" * 80)

# ================================================================
# FIX 1: Fix SensorInterface __del__ error
# ================================================================
def fix_sensor_interface():
    file_path = Path('src/environment/sensor_interface.py')
    
    if not file_path.exists():
        return False, f"{file_path} not found"
    
    content = file_path.read_text(encoding='utf-8')
    
    # Fix __del__ to check attribute exists
    old_del = '''    def __del__(self):
        """Cleanup on destruction."""
        self.stop()'''
    
    new_del = '''    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, 'is_running'):
                self.stop()
        except:
            pass'''
    
    if old_del in content:
        content = content.replace(old_del, new_del)
        file_path.write_text(content, encoding='utf-8')
        return True, "Fixed SensorInterface __del__"
    
    return True, "SensorInterface __del__ already OK"


# ================================================================
# FIX 2: Add transform_pose
# ================================================================
def add_transform_pose():
    file_path = Path('src/localization/coordinate_transforms.py')
    
    if not file_path.exists():
        return False, f"{file_path} not found"
    
    content = file_path.read_text(encoding='utf-8')
    
    if 'def transform_pose' in content:
        return True, "transform_pose already exists"
    
    addition = '''

def transform_pose(pose, from_frame='airsim', to_frame='ned'):
    """Transform pose between coordinate frames."""
    import numpy as np
    if len(pose) >= 7:
        position = pose[:3]
        orientation = pose[3:7]
        if from_frame == 'airsim' and to_frame == 'ned':
            position = airsim_to_ned(position)
        elif from_frame == 'ned' and to_frame == 'enu':
            position = ned_to_enu(position)
        elif from_frame == 'enu' and to_frame == 'ned':
            position = enu_to_ned(position)
        elif from_frame == 'ned' and to_frame == 'airsim':
            position = ned_to_airsim(position)
        return np.concatenate([position, orientation])
    else:
        if from_frame == 'airsim' and to_frame == 'ned':
            return airsim_to_ned(pose)
        elif from_frame == 'ned' and to_frame == 'enu':
            return ned_to_enu(pose)
        elif from_frame == 'enu' and to_frame == 'ned':
            return enu_to_ned(pose)
        elif from_frame == 'ned' and to_frame == 'airsim':
            return ned_to_airsim(pose)
        return pose
'''
    content += addition
    file_path.write_text(content, encoding='utf-8')
    return True, "Added transform_pose"


# ================================================================
# FIX 3: Add world_to_grid
# ================================================================
def add_world_to_grid():
    file_path = Path('src/utils/coordinate_utils.py')
    
    if not file_path.exists():
        return False, f"{file_path} not found"
    
    content = file_path.read_text(encoding='utf-8')
    
    if 'def world_to_grid' in content:
        return True, "world_to_grid already exists"
    
    addition = '''

def world_to_grid(position, grid_resolution=0.5, grid_origin=(0.0, 0.0, 0.0)):
    """Convert world coordinates to grid indices."""
    ix = int((position[0] - grid_origin[0]) / grid_resolution)
    iy = int((position[1] - grid_origin[1]) / grid_resolution)
    iz = int((position[2] - grid_origin[2]) / grid_resolution)
    return (ix, iy, iz)


def grid_to_world(grid_idx, grid_resolution=0.5, grid_origin=(0.0, 0.0, 0.0)):
    """Convert grid indices to world coordinates."""
    x = grid_idx[0] * grid_resolution + grid_origin[0] + grid_resolution / 2
    y = grid_idx[1] * grid_resolution + grid_origin[1] + grid_resolution / 2
    z = grid_idx[2] * grid_resolution + grid_origin[2] + grid_resolution / 2
    return (x, y, z)
'''
    content += addition
    file_path.write_text(content, encoding='utf-8')
    return True, "Added world_to_grid"


# ================================================================
# FIX 4: Fix GAECalculator
# ================================================================
def fix_gae_calculator():
    file_path = Path('src/rl/agents/gae_calculator.py')
    
    if not file_path.exists():
        return False, f"{file_path} not found"
    
    content = file_path.read_text(encoding='utf-8')
    
    if 'lam: float = None' in content or 'lam=None' in content:
        return True, "GAECalculator already accepts lam"
    
    # Fix __init__ signature
    content = re.sub(
        r'def __init__\(self,\s*gamma:\s*float\s*=\s*0\.99,\s*lambda_:\s*float\s*=\s*0\.95\):',
        'def __init__(self, gamma: float = 0.99, lambda_: float = 0.95, lam: float = None):',
        content
    )
    
    # Add lam handling
    content = re.sub(
        r'(\s+)(self\.gamma\s*=\s*gamma)',
        r'\1# Support both "lam" and "lambda_"\n\1if lam is not None:\n\1    lambda_ = lam\n\1\2',
        content,
        count=1
    )
    
    file_path.write_text(content, encoding='utf-8')
    return True, "Fixed GAECalculator"


# ================================================================
# MAIN
# ================================================================
def main():
    results = []
    
    # Apply all fixes
    results.append(fix_sensor_interface())
    results.append(add_transform_pose())
    results.append(add_world_to_grid())
    results.append(fix_gae_calculator())
    
    print()
    success = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    print(f"✅ SUCCESS: {len(success)}/4")
    for _, msg in success:
        print(f"  • {msg}")
    
    if failed:
        print(f"\n❌ FAILED: {len(failed)}/4")
        for _, msg in failed:
            print(f"  • {msg}")
    
    print("\n" + "=" * 80)
    print("\n🚀 CRITICAL NEXT STEPS:")
    print("\n1. CLEAR ALL CACHE (MANDATORY!):")
    print("   Get-ChildItem -Path . -Filter '__pycache__' -Recurse -Directory | Remove-Item -Recurse -Force")
    print("   Get-ChildItem -Path . -Filter '*.pyc' -Recurse -File | Remove-Item -Force")
    print("\n2. Run tests:")
    print("   pytest tests/ -v --ignore=tests/unit/test_sensor_bridge_unit.py --ignore=tests/unit/test_airsim_bridge_unit.py")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
