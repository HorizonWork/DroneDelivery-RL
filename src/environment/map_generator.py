import sys
import json
import time
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print("Warning: airsim package not installed. Only synthetic mode available.")

dataclass
class GridConfig:

    origin: Tuple[float, float, float] = (0.0, 0.0, 15.0)
    size: Tuple[float, float, float] = (20.0, 20.0, 30.0)
    cell_size: float = 0.5
    drone_radius: float = 0.3
    use_airsim: bool = False
    checkpoint_enabled: bool = True
    lidar_range: float = 50.0
    world_offset: Tuple[float, float, float] = (60.0, -30.0, 1.5)

    def __post_init__(self):

        if self.cell_size = 0:
            raise ValueError(f"cell_size must be  0, got {self.cell_size}")
        if any(s = 0 for s in self.size):
            raise ValueError(f"size dimensions must be  0, got {self.size}")
        if self.cell_size  min(self.size) / 10:
            raise ValueError(f"cell_size {self.cell_size} too large for size {self.size}")
        if self.drone_radius  0:
            raise ValueError(f"drone_radius must be = 0, got {self.drone_radius}")

class FlyZoneBuilder:

    def __init__(self, config: GridConfig):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        self.config = config
        self.project_root = Path(__file__).parent.parent.parent

        self.grid_dims = self._calculate_grid_dimensions()
        self.grid = None

        self.client = None
        self.airsim_connected = False

        self.inflation_cells = max(1, int(np.ceil(config.drone_radius / config.cell_size)))

        self.logger.info("FlyZone initialized:")
        self.logger.info(f"  Origin (grid): {config.origin}")
        self.logger.info(f"  World offset: {config.world_offset} m")
        self.logger.info(f"  Size: {config.size} m")
        self.logger.info(f"  Cell size: {config.cell_size} m")
        self.logger.info(f"  Grid dimensions: {self.grid_dims}")
        self.logger.info(f"  Inflation radius: {self.inflation_cells} cells ({config.drone_radius} m)")

    def _setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter('(asctime)s - (levelname)s - (message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler("build_fly_zone.log", encoding="utf-8")
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def _calculate_grid_dimensions(self) - Tuple[int, int, int]:
        x_cells = int(self.config.size[0] / self.config.cell_size)
        y_cells = int(self.config.size[1] / self.config.cell_size)
        z_cells = int(self.config.size[2] / self.config.cell_size)
        return (x_cells, y_cells, z_cells)

    def connect_airsim(self) - bool:

        if not self.config.use_airsim:
            self.logger.info("AirSim integration disabled by config")
            return False

        if not AIRSIM_AVAILABLE:
            self.logger.error("airsim package not installed. Install with: pip install airsim")
            return False

        try:
            self.logger.info("Connecting to AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()

            pose = self.client.simGetVehiclePose()
            x_ned, y_ned, z_ned = pose.position.x_val, pose.position.y_val, pose.position.z_val
            x_enu, y_enu, z_enu = self._ned_to_enu(x_ned, y_ned, z_ned)
            self.logger.info("AirSim connected!")
            self.logger.info(f"  Vehicle NED (world): ({x_ned:.2f}, {y_ned:.2f}, {z_ned:.2f}) m")
            self.logger.info(f"  Vehicle ENU (grid):  ({x_enu:.2f}, {y_enu:.2f}, {z_enu:.2f}) m")

            self.client.enableApiControl(True)

            self.airsim_connected = True
            return True

        except ConnectionRefusedError:
            self.logger.error("AirSim connection refused. Is AirSim/Unreal running?")
            self.logger.error("Start Unreal Engine with AirSim plugin first.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to AirSim: {type(e).__name__}: {e}")
            return False

    def _ned_to_enu(self, x_ned: float, y_ned: float, z_ned: float) - Tuple[float, float, float]:

        x_ned_origin = x_ned - self.config.world_offset[0]
        y_ned_origin = y_ned - self.config.world_offset[1]
        z_ned_origin = z_ned - self.config.world_offset[2]

        x_enu = y_ned_origin
        y_enu = x_ned_origin
        z_enu = -z_ned_origin

        return x_enu, y_enu, z_enu

    def _enu_to_ned(self, x_enu: float, y_enu: float, z_enu: float) - Tuple[float, float, float]:

        x_ned_origin = y_enu
        y_ned_origin = x_enu
        z_ned_origin = -z_enu

        x_ned = x_ned_origin + self.config.world_offset[0]
        y_ned = y_ned_origin + self.config.world_offset[1]
        z_ned = z_ned_origin + self.config.world_offset[2]

        return x_ned, y_ned, z_ned

    def _world_to_grid(self, x: float, y: float, z: float) - Tuple[int, int, int]:

        x_idx = int((x - (self.config.origin[0] - self.config.size[0]/2)) / self.config.cell_size)
        y_idx = int((y - (self.config.origin[1] - self.config.size[1]/2)) / self.config.cell_size)
        z_idx = int((z - self.config.origin[2]) / self.config.cell_size)
        return x_idx, y_idx, z_idx

    def _inflate_obstacles(self, grid: np.ndarray) - np.ndarray:

        if self.inflation_cells = 1:
            return grid

        self.logger.info(f"Inflating obstacles by {self.inflation_cells} cells...")

        from scipy.ndimage import binary_dilation

        r = self.inflation_cells
        y, x, z = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        sphere = x2 + y2 + z2 = r2

        inflated = binary_dilation(grid, structure=sphere).astype(np.uint8)

        added_cells = np.sum(inflated) - np.sum(grid)
        self.logger.info(f"Added {added_cells} cells for safety margin")

        return inflated

    def _get_airsim_lidar_grid(self) - Optional[np.ndarray]:

        if not self.airsim_connected or self.client is None:
            return None

        try:
            self.logger.info("="  60)
            self.logger.info("Building grid from AirSim LiDAR...")
            self.logger.info("="  60)

            grid = np.zeros(self.grid_dims, dtype=np.uint8)
            x_count, y_count, z_count = self.grid_dims

            x_steps, y_steps, z_steps = 5, 5, 10
            scan_positions_enu = []

            for xi in range(x_steps):
                x_enu = self.config.origin[0] - self.config.size[0]/2 + (xi + 0.5)  (self.config.size[0] / x_steps)
                for yi in range(y_steps):
                    y_enu = self.config.origin[1] - self.config.size[1]/2 + (yi + 0.5)  (self.config.size[1] / y_steps)
                    for zi in range(z_steps):
                        z_enu = self.config.origin[2] + (zi + 0.5)  (self.config.size[2] / z_steps)
                        scan_positions_enu.append((x_enu, y_enu, z_enu))

            total_scans = len(scan_positions_enu)
            self.logger.info(f"Scanning {total_scans} positions ({x_steps}{y_steps}{z_steps})...")

            total_points = 0
            occupied_points = []

            for idx, (x_enu, y_enu, z_enu) in enumerate(scan_positions_enu):
                try:
                    x_ned, y_ned, z_ned = self._enu_to_ned(x_enu, y_enu, z_enu)

                    scan_pose = airsim.Pose(
                        airsim.Vector3r(x_ned, y_ned, z_ned),
                        airsim.Quaternionr(0, 0, 0, 1)
                    )
                    self.client.simSetVehiclePose(scan_pose, ignore_collision=True)
                    time.sleep(0.05)

                    lidar_data = self.client.getLidarData(lidar_name="LidarSensor1")

                    if len(lidar_data.point_cloud)  3:
                        continue

                    points_ned_relative = np.array(lidar_data.point_cloud, dtype=np.float32).reshape((-1, 3))

                    distances = np.linalg.norm(points_ned_relative, axis=1)
                    valid_mask = distances  self.config.lidar_range
                    points_ned_relative = points_ned_relative[valid_mask]

                    if len(points_ned_relative) == 0:
                        continue

                    points_ned_world = points_ned_relative + np.array([x_ned, y_ned, z_ned])

                    points_enu = np.array([self._ned_to_enu(p[0], p[1], p[2]) for p in points_ned_world])

                    grid_indices = np.array([self._world_to_grid(p[0], p[1], p[2]) for p in points_enu])

                    valid = (
                        (grid_indices[:, 0] = 0)  (grid_indices[:, 0]  x_count)
                        (grid_indices[:, 1] = 0)  (grid_indices[:, 1]  y_count)
                        (grid_indices[:, 2] = 0)  (grid_indices[:, 2]  z_count)
                    )
                    valid_indices = grid_indices[valid]

                    occupied_points.extend(valid_indices.tolist())
                    total_points += len(points_ned_relative)

                    if (idx + 1)  50 == 0 or idx == total_scans - 1:
                        progress = ((idx + 1) / total_scans)  100
                        self.logger.info(f"  [{progress:5.1f}] Scan {idx+1}/{total_scans}  Points: {total_points}  Occupied cells: {len(occupied_points)}")

                except Exception as e:
                    self.logger.debug(f"Scan {idx+1} failed: {type(e).__name__}: {e}")
                    continue

            if occupied_points:
                self.logger.info("Processing occupied cells...")
                occupied_array = np.array(occupied_points, dtype=int)
                grid[occupied_array[:, 0], occupied_array[:, 1], occupied_array[:, 2]] = 1

            occupied_count = int(np.sum(grid))
            free_count = int(np.prod(grid.shape) - occupied_count)

            if occupied_count  100:
                self.logger.info("="  60)
                self.logger.info(f" LiDAR scan complete!")
                self.logger.info(f"  Total points: {total_points}")
                self.logger.info(f"  Occupied cells: {occupied_count}")
                self.logger.info(f"  Free cells: {free_count}")
                self.logger.info(f"  Free space: {free_count/np.prod(grid.shape)100:.1f}")
                self.logger.info("="  60)

                grid = self._inflate_obstacles(grid)

                return grid
            else:
                self.logger.warning(f" Insufficient LiDAR data: only {occupied_count} occupied cells")
                self.logger.warning("Troubleshooting:")
                self.logger.warning("  1. Check LiDAR sensor in settings.json (range, FOV)")
                self.logger.warning("  2. Verify environment has obstacles in scan volume")
                self.logger.warning("  3. Check grid origin/size matches environment")
                return None

        except Exception as e:
            self.logger.error(f"LiDAR scanning failed: {type(e).__name__}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _get_cell_center(self, i: int, j: int, k: int) - Tuple[float, float, float]:
        x = self.config.origin[0] - self.config.size[0]/2 + (i + 0.5)  self.config.cell_size
        y = self.config.origin[1] - self.config.size[1]/2 + (j + 0.5)  self.config.cell_size
        z = self.config.origin[2] + (k + 0.5)  self.config.cell_size
        return (x, y, z)

    def _in_flyable_region(self, x: float, y: float, z: float) - bool:
        half_x = self.config.size[0] / 2
        half_y = self.config.size[1] / 2

        if abs(x - self.config.origin[0])  half_x:
            return False
        if abs(y - self.config.origin[1])  half_y:
            return False
        if z  self.config.origin[2] or z  self.config.origin[2] + self.config.size[2]:
            return False

        return True

    def _has_obstacle_collision_check(self, x: float, y: float, z: float) - bool:

        if not self.airsim_connected or self.client is None:
            return self._has_obstacle_synthetic(x, y, z)

        try:
            x_ned, y_ned, z_ned = self._enu_to_ned(x, y, z)

            pose = airsim.Pose(airsim.Vector3r(x_ned, y_ned, z_ned), airsim.Quaternionr(0, 0, 0, 1))
            self.client.simSetVehiclePose(pose, ignore_collision=True)
            time.sleep(0.01)

            collision_info = self.client.simGetCollisionInfo()
            return collision_info.has_collided

        except Exception as e:
            self.logger.debug(f"Collision check failed: {e}")
            return self._has_obstacle_synthetic(x, y, z)

    def _has_obstacle_synthetic(self, x: float, y: float, z: float) - bool:

        floor_height = 3.0
        floor_thickness = 0.15

        for floor_num in range(0, 11):
            floor_z = self.config.origin[2] + floor_num  floor_height
            if abs(z - floor_z)  floor_thickness:
                if not self._is_in_stairwell(x, y):
                    return True

        wall_thickness = 0.15
        half_x = self.config.size[0] / 2
        half_y = self.config.size[1] / 2

        x_rel = abs(x - self.config.origin[0])
        y_rel = abs(y - self.config.origin[1])

        if x_rel  half_x - wall_thickness:
            return True
        if y_rel  half_y - wall_thickness:
            return True

        return False

    def _is_in_stairwell(self, x: float, y: float) - bool:
        stairwell_positions = [
            (self.config.origin[0] - 7, self.config.origin[1] - 7, 2.0, 2.0),
            (self.config.origin[0] + 7, self.config.origin[1] - 7, 2.0, 2.0),
            (self.config.origin[0] - 7, self.config.origin[1] + 7, 2.0, 2.0),
            (self.config.origin[0] + 7, self.config.origin[1] + 7, 2.0, 2.0)
        ]

        for sx, sy, sw, sh in stairwell_positions:
            if abs(x - sx)  sw/2 and abs(y - sy)  sh/2:
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

    def _load_checkpoint(self) - Tuple[Optional[np.ndarray], int]:
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

    def build_grid(self) - np.ndarray:

        self.logger.info("="  60)
        self.logger.info("Building 3D occupancy grid...")
        self.logger.info("="  60)

        x_count, y_count, z_count = self.grid_dims
        total_cells = x_count  y_count  z_count

        if self.airsim_connected and self.config.use_airsim:
            self.logger.info("Using AirSim LiDAR scanning method...")
            airsim_grid = self._get_airsim_lidar_grid()

            if airsim_grid is not None:
                self.grid = airsim_grid
                return self.grid
            else:
                self.logger.warning("LiDAR scanning failed, falling back to synthetic obstacles")

        self.logger.info("="  60)
        self.logger.info("Building synthetic obstacle grid...")
        self.logger.info("="  60)

        checkpoint_grid, start_layer = self._load_checkpoint()

        if checkpoint_grid is not None:
            grid = checkpoint_grid
            self.logger.info(f"Resuming from checkpoint at layer {start_layer}/{z_count}")
        else:
            grid = np.zeros((x_count, y_count, z_count), dtype=np.uint8)
            start_layer = 0

        for k in range(start_layer, z_count):
            for j in range(y_count):
                for i in range(x_count):
                    x, y, z = self._get_cell_center(i, j, k)

                    if not self._in_flyable_region(x, y, z):
                        grid[i, j, k] = 1
                    elif self._has_obstacle_synthetic(x, y, z):
                        grid[i, j, k] = 1

            if (k + 1)  10 == 0 or k == z_count - 1:
                progress = ((k + 1) / z_count)  100
                occupied = int(np.sum(grid[:, :, :k+1]))
                free = int(np.prod(grid[:, :, :k+1].shape) - occupied)
                self.logger.info(f"  [{progress:5.1f}] Layer {k+1}/{z_count}  Free: {free}  Occupied: {occupied}")
                self._save_checkpoint(grid, k + 1)

        self.grid = grid
        self._clear_checkpoint()

        occupied_count = int(np.sum(grid))
        free_count = int(np.prod(grid.shape) - occupied_count)

        self.logger.info("="  60)
        self.logger.info(" Grid construction complete!")
        self.logger.info(f"  Free cells: {free_count}")
        self.logger.info(f"  Occupied cells: {occupied_count}")
        self.logger.info(f"  Free space: {free_count/total_cells100:.1f}")
        self.logger.info("="  60)

        return grid

    def save_grid(self, output_dir: Optional[Path] = None) - Path:
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
            "world_offset": self.config.world_offset,
            "dimensions": self.grid_dims,
            "shape": self.grid.shape,
            "dtype": str(self.grid.dtype),
            "occupied_cells": int(np.sum(self.grid)),
            "free_cells": int(np.prod(self.grid.shape) - np.sum(self.grid)),
            "coordinate_systems": {
                "grid": "ENU (X=East, Y=North, Z=Up)",
                "airsim": "NED (X=North, Y=East, Z=Down)",
                "world_offset_note": "Unreal world was rebased by {60, -30, 1.5}m from PlayerStart"
            }
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
            plt.title(f'Occupancy Grid Slice at Z={z_index} (height={self.config.origin[2] + z_indexself.config.cell_size:.1f}m)')
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

    def build_complete(self, output_dir: Optional[Path] = None, visualize: bool = True) - bool:

        try:
            self.logger.info("="  60)
            self.logger.info("STARTING FLY ZONE CONSTRUCTION")
            self.logger.info("="  60)

            if self.config.use_airsim:
                self.logger.info("Connecting to AirSim...")
                connected = self.connect_airsim()

                if not connected:
                    self.logger.warning("AirSim not available, falling back to synthetic obstacles")
                    self.logger.warning("To use AirSim:")
                    self.logger.warning("  1. Start Unreal Engine with AirSim plugin")
                    self.logger.warning("  2. Ensure drone is spawned")
                    self.logger.warning("  3. Re-run with --use-airsim flag")
            else:
                self.logger.info("Using SYNTHETIC obstacles (--use-airsim not specified)")

            self.build_grid()

            self.logger.info("\nSaving grid...")
            self.save_grid(output_dir)

            if visualize:
                self.logger.info("\nGenerating visualizations...")
                vis_dir = self.project_root / "results" / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)

                z_slices = [0, self.grid_dims[2]
                for z_idx in z_slices:
                    if z_idx  self.grid_dims[2]:
                        vis_path = vis_dir / f"grid_slice_z{z_idx:03d}.png"
                        self.visualize_slice(z_idx, vis_path)

            if self.airsim_connected and self.client is not None:
                try:
                    self.client.enableApiControl(False)
                    self.logger.info("Disconnected from AirSim")
                except Exception as e:
                    self.logger.debug(f"Error disconnecting: {e}")

            self.logger.info("\n" + "="  60)
            self.logger.info(" FLY ZONE CONSTRUCTION COMPLETED SUCCESSFULLY!")
            self.logger.info("="  60)
            return True

        except KeyboardInterrupt:
            self.logger.warning("\n" + "="  60)
            self.logger.warning("Construction interrupted by user (Ctrl+C)")
            self.logger.warning("Checkpoint saved - you can resume later")
            self.logger.warning("="  60)
            return False

        except Exception as e:
            self.logger.error("\n" + "="  60)
            self.logger.error(f" CONSTRUCTION FAILED: {type(e).__name__}")
            self.logger.error(f"Error: {e}")
            self.logger.error("="  60)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Build 3D occupancy grid for drone navigation using AirSim LiDAR or synthetic obstacles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
    )
    parser.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 15.0],
                        help="Grid origin (x, y, z) in meters. Default: 0 0 15")
    parser.add_argument("--size", nargs=3, type=float, default=[20.0, 20.0, 30.0],
                        help="Grid size (x, y, z) in meters. Default: 20 20 30")
    parser.add_argument("--cell-size", type=float, default=0.5,
                        help="Cell size in meters. Default: 0.5")
    parser.add_argument("--drone-radius", type=float, default=0.3,
                        help="Drone safety radius for obstacle inflation (meters). Default: 0.3")
    parser.add_argument("--use-airsim", action="store_true",
                        help="Use AirSim LiDAR scanning (requires AirSim running)")
    parser.add_argument("--lidar-range", type=float, default=50.0,
                        help="Maximum LiDAR range in meters. Default: 50")
    parser.add_argument("--world-offset", nargs=3, type=float, default=[60.0, -30.0, 1.5],
                        help="Unreal world offset in meters (x, y, z). Default: 60 -30 1.5 (from PlayerStart rebase)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable checkpoint/resume functionality")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/maps)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization generation")

    args = parser.parse_args()

    try:
        config = GridConfig(
            origin=tuple(args.origin),
            size=tuple(args.size),
            cell_size=args.cell_size,
            drone_radius=args.drone_radius,
            use_airsim=args.use_airsim,
            lidar_range=args.lidar_range,
            world_offset=tuple(args.world_offset),
            checkpoint_enabled=not args.no_checkpoint
        )
    except ValueError as e:
        print(f" Invalid configuration: {e}")
        sys.exit(1)

    if args.use_airsim and not AIRSIM_AVAILABLE:
        print(" Error: --use-airsim specified but airsim package not installed")
        print("Install with: pip install airsim")
        sys.exit(1)

    builder = FlyZoneBuilder(config)
    output_dir = Path(args.output) if args.output else None
    success = builder.build_complete(output_dir, visualize=not args.no_viz)

    if success:
        print("\n" + "="  60)
        print(" FLY ZONE BUILT SUCCESSFULLY!")
        print("="  60)
        print(f" Grid:          {builder.project_root / 'data' / 'maps' / 'occupancy_grid.npy'}")
        print(f" Metadata:      {builder.project_root / 'data' / 'maps' / 'occupancy_grid_metadata.json'}")
        print(f" Visualizations: {builder.project_root / 'results' / 'visualizations'}")
        print("="  60)
        print("\n Next steps:")
        print("  1. Check visualizations: results/visualizations/grid_slice_.png")
        print("  2. Load grid in code: np.load('data/maps/occupancy_grid.npy')")
        print("  3. Train RL agent: python scripts/training/train_full_curriculum.py")
        sys.exit(0)
    else:
        print("\n" + "="  60)
        print(" FLY ZONE CONSTRUCTION FAILED")
        print("="  60)
        print("Check build_fly_zone.log for details")
        print("\n Troubleshooting:")
        print("  - If using --use-airsim: Ensure AirSim/Unreal is running")
        print("  - Check grid parameters (origin, size) match your environment")
        print("  - Try without --use-airsim for synthetic obstacles")
        sys.exit(1)

if __name__ == "__main__":
    main()
