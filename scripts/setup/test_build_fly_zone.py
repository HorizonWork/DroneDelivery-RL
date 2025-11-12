#!/usr/bin/env python3
"""
Demo script to test build_fly_zone.py functionality
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.build_fly_zone import FlyZoneBuilder, GridConfig
import numpy as np


def test_small_grid():
    print("Testing small grid generation (10x10x10)...")
    
    config = GridConfig(
        origin=(0.0, 0.0, 5.0),
        size=(5.0, 5.0, 5.0),
        cell_size=0.5,
        drone_radius=0.3,
        use_airsim_voxel=False,
        checkpoint_enabled=False
    )
    
    builder = FlyZoneBuilder(config)
    grid = builder.build_grid()
    
    print(f"Grid shape: {grid.shape}")
    print(f"Occupied cells: {np.sum(grid)}")
    print(f"Free cells: {np.prod(grid.shape) - np.sum(grid)}")
    print(f"Free ratio: {(np.prod(grid.shape) - np.sum(grid)) / np.prod(grid.shape) * 100:.1f}%")
    
    assert grid.shape == (10, 10, 10), f"Expected (10,10,10), got {grid.shape}"
    assert np.sum(grid) > 0, "Grid should have some occupied cells"
    
    print("✓ Small grid test passed\n")
    return True


def test_config_validation():
    print("Testing config validation...")
    
    try:
        config = GridConfig(cell_size=-0.5)
        print("✗ Should have raised ValueError for negative cell_size")
        return False
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        config = GridConfig(size=(0, 10, 10))
        print("✗ Should have raised ValueError for zero size")
        return False
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    try:
        config = GridConfig(cell_size=10.0, size=(5, 5, 5))
        print("✗ Should have raised ValueError for cell_size too large")
        return False
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")
    
    print("✓ Config validation test passed\n")
    return True


def test_coordinate_conversion():
    print("Testing coordinate conversion...")
    
    config = GridConfig(
        origin=(0.0, 0.0, 10.0),
        size=(10.0, 10.0, 10.0),
        cell_size=1.0
    )
    
    builder = FlyZoneBuilder(config)
    
    x, y, z = builder._get_cell_center(0, 0, 0)
    print(f"Cell (0,0,0) center: ({x:.1f}, {y:.1f}, {z:.1f})")
    assert abs(x - (-4.5)) < 0.1, f"Expected x=-4.5, got {x}"
    assert abs(y - (-4.5)) < 0.1, f"Expected y=-4.5, got {y}"
    assert abs(z - 10.5) < 0.1, f"Expected z=10.5, got {z}"
    
    x, y, z = builder._get_cell_center(9, 9, 9)
    print(f"Cell (9,9,9) center: ({x:.1f}, {y:.1f}, {z:.1f})")
    assert abs(x - 4.5) < 0.1, f"Expected x=4.5, got {x}"
    assert abs(y - 4.5) < 0.1, f"Expected y=4.5, got {y}"
    assert abs(z - 19.5) < 0.1, f"Expected z=19.5, got {z}"
    
    print("✓ Coordinate conversion test passed\n")
    return True


def test_boundary_detection():
    print("Testing boundary detection...")
    
    config = GridConfig(
        origin=(0.0, 0.0, 10.0),
        size=(10.0, 10.0, 10.0),
        cell_size=1.0
    )
    
    builder = FlyZoneBuilder(config)
    
    assert builder._in_flyable_region(0, 0, 15), "Center should be in region"
    assert not builder._in_flyable_region(-10, 0, 15), "Far left should be out"
    assert not builder._in_flyable_region(0, 0, 5), "Below should be out"
    assert not builder._in_flyable_region(0, 0, 25), "Above should be out"
    
    print("✓ Boundary detection test passed\n")
    return True


def test_synthetic_obstacles():
    print("Testing synthetic obstacle detection...")
    
    config = GridConfig(
        origin=(0.0, 0.0, 15.0),
        size=(20.0, 20.0, 15.0),
        cell_size=0.5
    )
    
    builder = FlyZoneBuilder(config)
    
    floor_z = 15.0
    assert builder._has_obstacle_synthetic(0, 0, floor_z), "Floor should be occupied"
    assert not builder._has_obstacle_synthetic(0, 0, floor_z + 1.5), "Above floor should be free"
    
    boundary_x = 10.0
    assert builder._has_obstacle_synthetic(boundary_x, 0, 20), "Boundary wall should be occupied"
    
    print("✓ Synthetic obstacle test passed\n")
    return True


def main():
    print("=" * 60)
    print("Build Fly Zone - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_config_validation,
        test_coordinate_conversion,
        test_boundary_detection,
        test_synthetic_obstacles,
        test_small_grid
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
