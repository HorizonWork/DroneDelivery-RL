#!/usr/bin/env python3
"""
Build Fly Zone Script
Generates 3D occupancy grid for drone navigation in AirSim/UE environment.
Supports both AirSim voxel grid API and synthetic obstacle generation.
"""

import sys
import json
import time
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
import airsim
from dataclasses import dataclass


@dataclass
class GridConfig:
    origin: Tuple[float, float, float] = (0.0, 0.0, 15.0)
    size: Tuple[float, float, float] = (20.0, 20.0, 30.0)
    cell_size: float = 0.5
    drone_radius: float = 0.3
    use_airsim_voxel: bool = False
    checkpoint_enabled: bool = True
    
    def __post_init__(self):
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be > 0, got {self.cell_size}")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"size dimensions must be > 0, got {self.size}")
        if self.cell_size > min(self.size) / 10:
            raise ValueError(f"cell_size {self.cell_size} too large for size {self.size}")


class FlyZoneBuilder:
    def __init__(self, config: GridConfig):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent
        
        self.grid_dims = self._calculate_grid_dimensions()
        self.grid = None
        
        self.client = None
        self.airsim_available = False
        
        self.logger.info(f"FlyZone initialized: origin={config.origin}, size={config.size}, cell={config.cell_size}")
        self.logger.info(f"Grid dimensions: {self.grid_dims}")

    def _setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        file_handler = logging.FileHandler("build_fly_zone.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def _calculate_grid_dimensions(self) -> Tuple[int, int, int]:
        x_cells = int(self.config.size[0] / self.config.cell_size)
        y_cells = int(self.config.size[1] / self.config.cell_size)
        z_cells = int(self.config.size[2] / self.config.cell_size)
        return (x_cells, y_cells, z_cells)

    def connect_airsim(self) -> bool:
        if not self.config.use_airsim_voxel:
            self.logger.info("AirSim integration disabled by config")
            return False
        
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            
            self.client.simGetVehiclePose()
            
            self.airsim_available = True
            self.logger.info("Connected to AirSim successfully")
            return True
            
        except ImportError:
            self.logger.warning("airsim package not installed")
            return False
        except ConnectionRefusedError:
            self.logger.warning("AirSim connection refused (check port 41451)")
            return False
        except Exception as e:
            error_msg = str(e)
            
            if "msgpack" in error_msg.lower() or "unicode" in error_msg.lower() or "rpc" in error_msg.lower():
                self.logger.warning("AirSim msgpack-rpc compatibility issue (Python 3.13+)")
                self.logger.warning("Falling back to synthetic obstacles")
            else:
                self.logger.warning(f"AirSim connection failed: {error_msg}")
            
            return False
    
    def _get_airsim_voxel_grid(self) -> Optional[np.ndarray]:
        """Get voxel grid from AirSim using comprehensive LiDAR scans."""
        if not self.airsim_available or self.client is None:
            return None
        
        try:
            self.logger.info("Building grid from AirSim using LiDAR scans...")
            self.logger.info("Scanning environment with 3D grid pattern...")
            
            grid = np.zeros(self.grid_dims, dtype=np.uint8)
            x_count, y_count, z_count = self.grid_dims
            
            # Create comprehensive 3D scan positions (5x5x10 = 250 scans)
            scan_positions = []
            x_steps, y_steps, z_steps = 5, 5, 10
            
            for xi in range(x_steps):
                x_pos = self.config.origin[0] - self.config.size[0]/2 + \
                       (xi + 0.5) * (self.config.size[0] / x_steps)
                
                for yi in range(y_steps):
                    y_pos = self.config.origin[1] - self.config.size[1]/2 + \
                           (yi + 0.5) * (self.config.size[1] / y_steps)
                    
                    for zi in range(z_steps):
                        z_pos = self.config.origin[2] + \
                               (zi + 0.5) * (self.config.size[2] / z_steps)
                        
                        scan_positions.append((x_pos, y_pos, z_pos))
            
            total_scans = len(scan_positions)
            self.logger.info(f"Will perform {total_scans} LiDAR scans ({x_steps}x{y_steps}x{z_steps})")
            
            point_count = 0
            
            for idx, (scan_x, scan_y, scan_z) in enumerate(scan_positions):
                try:
                    # AirSim API expects meters in NED frame (Z is down)
                    scan_pose = airsim.Pose(
                        airsim.Vector3r(scan_x, scan_y, -scan_z),
                        airsim.Quaternionr(0, 0, 0, 1)
                    )
                    
                    self.client.simSetVehiclePose(scan_pose, True)
                    
                    time.sleep(0.05)
                    
                    lidar_data = self.client.getLidarData(lidar_name="Lidar1")
                    
                    if hasattr(lidar_data, 'point_cloud') and len(lidar_data.point_cloud) > 0:
                        points = np.array(lidar_data.point_cloud, dtype=np.float32)
                        
                        if points.size >= 3:
                            points = points.reshape((-1, 3))
                            point_count += len(points)
                            
                            for point in points:
                                # LiDAR returns meters in NED frame
                                x_world = point[0]
                                y_world = point[1]
                                z_world = -point[2]  # Convert NED (down) to Z-up
                                
                                # Convert to grid indices
                                x_idx = int((x_world - (self.config.origin[0] - self.config.size[0]/2)) / self.config.cell_size)
                                y_idx = int((y_world - (self.config.origin[1] - self.config.size[1]/2)) / self.config.cell_size)
                                z_idx = int((z_world - self.config.origin[2]) / self.config.cell_size)
                                
                                # Mark cell and neighbors as occupied (inflation for safety)
                                if 0 <= x_idx < x_count and 0 <= y_idx < y_count and 0 <= z_idx < z_count:
                                    for dx in range(-1, 2):
                                        for dy in range(-1, 2):
                                            for dz in range(-1, 2):
                                                nx, ny, nz = x_idx + dx, y_idx + dy, z_idx + dz
                                                if 0 <= nx < x_count and 0 <= ny < y_count and 0 <= nz < z_count:
                                                    grid[nx, ny, nz] = 1
                
                except Exception as e:
                    self.logger.debug(f"Scan {idx+1} failed: {e}")
                    continue
                
                # Progress logging every 50 scans
                if (idx + 1) % 50 == 0 or idx == total_scans - 1:
                    progress = ((idx + 1) / total_scans) * 100
                    occupied = int(np.sum(grid))
                    self.logger.info(f"Progress: {progress:.1f}% ({idx+1}/{total_scans}) - {point_count} points, {occupied} occupied cells")
            
            occupied_count = int(np.sum(grid))
            
            if occupied_count > 100:
                self.logger.info(f"✓ LiDAR scan complete: {point_count} total points → {occupied_count} occupied cells")
                return grid
            else:
                self.logger.warning(f"⚠ Insufficient LiDAR data: only {occupied_count} occupied cells detected")
                self.logger.warning("Possible causes:")
                self.logger.warning("  1. LiDAR range too short (increase in settings.json)")
                self.logger.warning("  2. Environment has few obstacles")
                self.logger.warning("  3. Scan positions outside environment bounds")
                return None
                
        except AttributeError as e:
            self.logger.warning(f"LiDAR not available: {e}")
            return None
        except (UnicodeDecodeError, ConnectionError) as e:
            self.logger.warning(f"AirSim connection error during scan: {e}")
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get voxel grid from AirSim: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _get_cell_center(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        x = self.config.origin[0] - self.config.size[0]/2 + (i + 0.5) * self.config.cell_size
        y = self.config.origin[1] - self.config.size[1]/2 + (j + 0.5) * self.config.cell_size
        z = self.config.origin[2] + (k + 0.5) * self.config.cell_size
        return (x, y, z)

    def _in_flyable_region(self, x: float, y: float, z: float) -> bool:
        half_x = self.config.size[0] / 2
        half_y = self.config.size[1] / 2
        
        if abs(x - self.config.origin[0]) > half_x:
            return False
        if abs(y - self.config.origin[1]) > half_y:
            return False
        if z < self.config.origin[2] or z > self.config.origin[2] + self.config.size[2]:
            return False
        
        return True

    def _has_obstacle(self, x: float, y: float, z: float) -> bool:
        if not self.airsim_available or self.client is None:
            return self._has_obstacle_synthetic(x, y, z)
        
        try:
            # AirSim API expects meters in NED frame (Z is down)
            pos_ue = airsim.Vector3r(x, y, -z)
            
            try:
                collision_info = self.client.simGetCollisionInfo()
                if collision_info.has_collided:
                    collision_pos = collision_info.position  # Returns meters in NED
                    
                    # Both positions are in meters now - no conversion needed
                    distance = np.linalg.norm([
                        pos_ue.x_val - collision_pos.x_val,
                        pos_ue.y_val - collision_pos.y_val,
                        pos_ue.z_val - collision_pos.z_val
                    ])
                    
                    # Compare distance in meters
                    if distance < self.config.drone_radius:
                        return True
            except (UnicodeDecodeError, Exception):
                self.airsim_available = False
                return self._has_obstacle_synthetic(x, y, z)
            
            try:
                # Ray casting in 6 directions to check for nearby obstacles
                for direction in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    offset = self.config.cell_size / 2
                    
                    # Both positions in meters (no * 100)
                    check_pos = airsim.Vector3r(
                        x + direction[0] * offset,
                        y + direction[1] * offset,
                        -(z + direction[2] * offset)
                    )
                    
                    ray_hit = self.client.simTestLineOfSightBetweenPoints(pos_ue, check_pos)
                    if ray_hit:
                        return True
            except (UnicodeDecodeError, Exception):
                self.airsim_available = False
                return self._has_obstacle_synthetic(x, y, z)
            
            return False
            
        except (UnicodeDecodeError, ConnectionError, Exception):
            if self.airsim_available:
                self.logger.warning("AirSim connection lost (msgpack-rpc error), switching to synthetic")
                self.airsim_available = False
            
            return self._has_obstacle_synthetic(x, y, z)

    def _has_obstacle_synthetic(self, x: float, y: float, z: float) -> bool:
        """
        Synthetic obstacle model: floors, walls, and stairwells.
        Most of the space should be FREE for drone navigation.
        """
        floor_height = 3.0
        floor_thickness = 0.15
        
        for floor_num in range(0, 11):
            floor_z = self.config.origin[2] + floor_num * floor_height
            if abs(z - floor_z) < floor_thickness:
                if not self._is_in_stairwell(x, y):
                    return True
        
        wall_thickness = 0.15
        half_x = self.config.size[0] / 2
        half_y = self.config.size[1] / 2
        
        x_rel = abs(x - self.config.origin[0])
        y_rel = abs(y - self.config.origin[1])
        
        if x_rel > half_x - wall_thickness:
            return True
        if y_rel > half_y - wall_thickness:
            return True
        
        return False

    def _is_in_stairwell(self, x: float, y: float) -> bool:
        stairwell_positions = [
            (self.config.origin[0] - 7, self.config.origin[1] - 7, 2.0, 2.0),
            (self.config.origin[0] + 7, self.config.origin[1] - 7, 2.0, 2.0),
            (self.config.origin[0] - 7, self.config.origin[1] + 7, 2.0, 2.0),
            (self.config.origin[0] + 7, self.config.origin[1] + 7, 2.0, 2.0)
        ]
        
        for sx, sy, sw, sh in stairwell_positions:
            if abs(x - sx) < sw/2 and abs(y - sy) < sh/2:
                return True
        return False
    
    def _save_checkpoint(self, grid: np.ndarray, layer: int):
        if not self.config.checkpoint_enabled:
            return
        
        checkpoint_dir = self.project_root / "temp" / "grid_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(checkpoint_dir / "grid_partial.npy", grid)
        with open(checkpoint_dir / "layer.txt", "w") as f:
            f.write(str(layer))
        
        self.logger.debug(f"Checkpoint saved at layer {layer}")
    
    def _load_checkpoint(self) -> Tuple[Optional[np.ndarray], int]:
        if not self.config.checkpoint_enabled:
            return None, 0
        
        checkpoint_dir = self.project_root / "temp" / "grid_checkpoints"
        grid_path = checkpoint_dir / "grid_partial.npy"
        layer_path = checkpoint_dir / "layer.txt"
        
        if grid_path.exists() and layer_path.exists():
            try:
                grid = np.load(grid_path)
                with open(layer_path, "r") as f:
                    layer = int(f.read().strip())
                self.logger.info(f"Resuming from checkpoint at layer {layer}")
                return grid, layer
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        return None, 0
    
    def _clear_checkpoint(self):
        checkpoint_dir = self.project_root / "temp" / "grid_checkpoints"
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)
            self.logger.debug("Checkpoint cleared")

    def build_grid(self) -> np.ndarray:
        self.logger.info("Building 3D occupancy grid...")
        
        x_count, y_count, z_count = self.grid_dims
        total_cells = x_count * y_count * z_count
        
        if self.airsim_available and self.config.use_airsim_voxel:
            self.logger.info("Attempting to get voxel grid from AirSim...")
            airsim_grid = self._get_airsim_voxel_grid()
            
            if airsim_grid is not None:
                self.grid = airsim_grid
                occupied_count = int(np.sum(self.grid))
                free_count = int(np.prod(self.grid.shape) - occupied_count)
                
                self.logger.info(f"Grid from AirSim: {free_count} free, {occupied_count} occupied")
                self.logger.info(f"Free space ratio: {free_count/total_cells*100:.1f}%")
                return self.grid
            else:
                self.logger.warning("Failed to get voxel grid, falling back to cell-by-cell query")
        
        checkpoint_grid, start_layer = self._load_checkpoint()
        
        if checkpoint_grid is not None:
            grid = checkpoint_grid
            self.logger.info(f"Resuming from layer {start_layer}/{z_count}")
        else:
            grid = np.zeros((x_count, y_count, z_count), dtype=np.uint8)
            start_layer = 0
        
        occupied_count = int(np.sum(grid[:, :, :start_layer]))
        free_count = int(np.prod(grid[:, :, :start_layer].shape) - occupied_count)
        
        airsim_failed_count = 0
        max_airsim_fails = 5
        
        for k in range(start_layer, z_count):
            for j in range(y_count):
                for i in range(x_count):
                    x, y, z = self._get_cell_center(i, j, k)
                    
                    if not self._in_flyable_region(x, y, z):
                        grid[i, j, k] = 1
                        occupied_count += 1
                    elif self._has_obstacle(x, y, z):
                        grid[i, j, k] = 1
                        occupied_count += 1
                    else:
                        grid[i, j, k] = 0
                        free_count += 1
                    
                    if not self.airsim_available and self.config.use_airsim_voxel:
                        airsim_failed_count += 1
                        if airsim_failed_count >= max_airsim_fails:
                            self.logger.warning("AirSim connection lost, continuing with synthetic obstacles")
                            self.config.use_airsim_voxel = False
                            airsim_failed_count = 0
            
            if (k + 1) % 10 == 0:
                progress = ((k + 1) / z_count) * 100
                self.logger.info(f"Progress: {progress:.1f}% ({k+1}/{z_count} layers)")
                self._save_checkpoint(grid, k + 1)
        
        self.grid = grid
        self._clear_checkpoint()
        
        self.logger.info(f"Grid complete: {free_count} free, {occupied_count} occupied")
        self.logger.info(f"Free space ratio: {free_count/total_cells*100:.1f}%")
        
        return grid

    def save_grid(self, output_dir: Optional[Path] = None) -> Path:
        if self.grid is None:
            raise ValueError("Grid not built yet. Call build_grid() first.")
        
        if output_dir is None:
            output_dir = self.project_root / "data" / "maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        npy_path = output_dir / "occupancy_grid.npy"
        np.save(npy_path, self.grid)
        self.logger.info(f"Saved grid to {npy_path}")
        
        metadata = {
            "origin": self.config.origin,
            "size": self.config.size,
            "cell_size": self.config.cell_size,
            "dimensions": self.grid_dims,
            "shape": self.grid.shape,
            "dtype": str(self.grid.dtype),
            "occupied_cells": int(np.sum(self.grid)),
            "free_cells": int(np.prod(self.grid.shape) - np.sum(self.grid))
        }
        
        json_path = output_dir / "occupancy_grid_metadata.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved metadata to {json_path}")
        
        return npy_path

    def visualize_slice(self, z_index: int, save_path: Optional[Path] = None):
        if self.grid is None:
            raise ValueError("Grid not built yet.")
        
        try:
            import matplotlib.pyplot as plt
            
            slice_data = self.grid[:, :, z_index]
            
            plt.figure(figsize=(10, 10))
            plt.imshow(slice_data.T, origin='lower', cmap='gray_r', interpolation='nearest')
            plt.colorbar(label='Occupancy (0=free, 1=occupied)')
            plt.title(f'Occupancy Grid Slice at Z={z_index} (height={self.config.origin[2] + z_index*self.config.cell_size:.1f}m)')
            plt.xlabel('X cells')
            plt.ylabel('Y cells')
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved visualization to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualization")

    def build_complete(self, output_dir: Optional[Path] = None, visualize: bool = True) -> bool:
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting fly zone construction")
            self.logger.info("=" * 60)
            
            if self.config.use_airsim_voxel:
                self.logger.info("Attempting AirSim connection...")
                connected = self.connect_airsim()
                
                if not connected:
                    self.logger.warning("AirSim not available, using synthetic obstacles")
                    self.logger.info("To use AirSim: 1) Start UE with AirSim, 2) Add --use-airsim flag")
            else:
                self.logger.info("Using synthetic obstacles (--use-airsim not specified)")
            
            self.logger.info("-" * 60)
            self.build_grid()
            
            self.logger.info("-" * 60)
            self.save_grid(output_dir)
            
            if visualize:
                self.logger.info("-" * 60)
                self.logger.info("Generating visualizations...")
                vis_dir = self.project_root / "results" / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                for z_idx in [0, self.grid_dims[2]//4, self.grid_dims[2]//2, 3*self.grid_dims[2]//4]:
                    if z_idx < self.grid_dims[2]:
                        vis_path = vis_dir / f"grid_slice_z{z_idx}.png"
                        self.visualize_slice(z_idx, vis_path)
            
            self.logger.info("=" * 60)
            self.logger.info("Fly zone construction completed successfully")
            self.logger.info("=" * 60)
            return True
            
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error(f"Fly zone construction failed: {e}")
            self.logger.error("=" * 60)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build 3D occupancy grid for drone navigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_fly_zone.py
  python build_fly_zone.py --use-airsim
  python build_fly_zone.py --origin 0 0 20 --size 30 30 40 --cell-size 0.25
        """
    )
    parser.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 15.0],
                        help="Grid origin (x, y, z)")
    parser.add_argument("--size", nargs=3, type=float, default=[20.0, 20.0, 30.0],
                        help="Grid size (x, y, z)")
    parser.add_argument("--cell-size", type=float, default=0.5,
                        help="Cell size in meters")
    parser.add_argument("--drone-radius", type=float, default=0.3,
                        help="Drone safety radius in meters")
    parser.add_argument("--use-airsim", action="store_true",
                        help="Use AirSim voxel API for obstacle detection")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpoint/resume")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization")
    
    args = parser.parse_args()
    
    try:
        config = GridConfig(
            origin=tuple(args.origin),
            size=tuple(args.size),
            cell_size=args.cell_size,
            drone_radius=args.drone_radius,
            use_airsim_voxel=args.use_airsim,
            checkpoint_enabled=not args.no_checkpoint
        )
    except ValueError as e:
        print(f"Invalid configuration: {e}")
        sys.exit(1)
    
    builder = FlyZoneBuilder(config)
    
    output_dir = Path(args.output) if args.output else None
    success = builder.build_complete(output_dir, visualize=not args.no_viz)
    
    if success:
        print("\n" + "=" * 60)
        print("Fly zone built successfully!")
        print("=" * 60)
        print(f"Grid: {builder.project_root / 'data' / 'maps' / 'occupancy_grid.npy'}")
        print(f"Metadata: {builder.project_root / 'data' / 'maps' / 'occupancy_grid_metadata.json'}")
        print(f"Visualizations: {builder.project_root / 'results' / 'visualizations'}")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("Fly zone construction failed")
        print("=" * 60)
        print("Check build_fly_zone.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
