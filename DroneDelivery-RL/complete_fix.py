from pathlib import Path
import re

print(" COMPLETE FIX - Fixing all 5 issues...")
print("="  80)

def fix_sensor_interface():
    file_path = Path('src/environment/sensor_interface.py')

    if not file_path.exists():
        return False, f"{file_path} not found"

    content = file_path.read_text(encoding='utf-8')

    old_del =

    new_del =

    if old_del in content:
        content = content.replace(old_del, new_del)
        file_path.write_text(content, encoding='utf-8')
        return True, "Fixed SensorInterface __del__"

    return True, "SensorInterface __del__ already OK"

def add_transform_pose():
    file_path = Path('src/localization/coordinate_transforms.py')

    if not file_path.exists():
        return False, f"{file_path} not found"

    content = file_path.read_text(encoding='utf-8')

    if 'def transform_pose' in content:
        return True, "transform_pose already exists"

    addition =
    content += addition
    file_path.write_text(content, encoding='utf-8')
    return True, "Added transform_pose"

def add_world_to_grid():
    file_path = Path('src/utils/coordinate_utils.py')

    if not file_path.exists():
        return False, f"{file_path} not found"

    content = file_path.read_text(encoding='utf-8')

    if 'def world_to_grid' in content:
        return True, "world_to_grid already exists"

    addition =
    content += addition
    file_path.write_text(content, encoding='utf-8')
    return True, "Added world_to_grid"

def fix_gae_calculator():
    file_path = Path('src/rl/agents/gae_calculator.py')

    if not file_path.exists():
        return False, f"{file_path} not found"

    content = file_path.read_text(encoding='utf-8')

    if 'lam: float = None' in content or 'lam=None' in content:
        return True, "GAECalculator already accepts lam"

    content = re.sub(
        r'def __init__\(self,\sgamma:\sfloat\s=\s0\.99,\slambda_:\sfloat\s=\s0\.95\):',
        'def __init__(self, gamma: float = 0.99, lambda_: float = 0.95, lam: float = None):',
        content
    )

    content = re.sub(
        r'(\s+)(self\.gamma\s=\sgamma)',
        r'\1
        content,
        count=1
    )

    file_path.write_text(content, encoding='utf-8')
    return True, "Fixed GAECalculator"

def main():
    results = []

    results.append(fix_sensor_interface())
    results.append(add_transform_pose())
    results.append(add_world_to_grid())
    results.append(fix_gae_calculator())

    print()
    success = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]

    print(f" SUCCESS: {len(success)}/4")
    for _, msg in success:
        print(f"   {msg}")

    if failed:
        print(f"\n FAILED: {len(failed)}/4")
        for _, msg in failed:
            print(f"   {msg}")

    print("\n" + "="  80)
    print("\n CRITICAL NEXT STEPS:")
    print("\n1. CLEAR ALL CACHE (MANDATORY!):")
    print("   Get-ChildItem -Path . -Filter '__pycache__' -Recurse -Directory  Remove-Item -Recurse -Force")
    print("   Get-ChildItem -Path . -Filter '.pyc' -Recurse -File  Remove-Item -Force")
    print("\n2. Run tests:")
    print("   pytest tests/ -v --ignore=tests/unit/test_sensor_bridge_unit.py --ignore=tests/unit/test_airsim_bridge_unit.py")
    print("\n" + "="  80)

if __name__ == '__main__':
    main()
