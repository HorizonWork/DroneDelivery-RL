import airsim
import numpy as np
import time
import json
import os
from datetime import datetime

class BuildingMapper:

    def __init__(self, client, resolution=0.5):
        self.client = client
        self.resolution = resolution
        self.building_bounds = None
        self.floor_heights = []
        self.static_obstacles = []

    def scan_environment(self, scan_heights=None, scan_radius=100):

        if scan_heights is None:
            scan_heights = [3, 9, 15, 21, 27]

        scan_positions_per_floor = [
            (-60, 30),
        ]

        print("  Phase 1: Floor-by-Floor Scanning (TELEPORT MODE)")
        print(f"    Strategy: {len(scan_heights)} floors  8 directions")
        print(f"    Mode: TELEPORT (instant + safe)")
        print(f"    Coverage: {len(scan_heights)  8} total scans")

        all_depth_data = []
        total_scans = 0

        for floor_idx, scan_height in enumerate(scan_heights):
            print(f"\n{'='60}")
            print(f" FLOOR {floor_idx+1}/{len(scan_heights)} (Height: {scan_height}m)")
            print(f"{'='60}")

            for pos_idx, (x, y) in enumerate(scan_positions_per_floor):
                print(f"    Position: ({x:.0f}, {y:.0f}, {scan_height}m)")

                angles = np.linspace(0, 360, 8, endpoint=False)

                for angle_idx, angle in enumerate(angles):
                    pose = airsim.Pose()
                    pose.position.x_val = float(x)
                    pose.position.y_val = float(y)
                    pose.position.z_val = float(-scan_height)

                    import math
                    yaw_rad = math.radians(angle)
                    pose.orientation = airsim.to_quaternion(0, 0, yaw_rad)

                    self.client.simSetVehiclePose(pose, ignore_collision=True)
                    time.sleep(0.15)

                    responses = self.client.simGetImages(
                        [
                            airsim.ImageRequest(
                                "0",
                                airsim.ImageType.DepthPerspective,
                                pixels_as_float=True,
                                compress=False,
                            )
                        ]
                    )

                    if responses:
                        depth_img = airsim.get_pfm_array(responses[0])
                        all_depth_data.append((angle, depth_img, scan_height, x, y))
                        total_scans += 1

                print(f"       Captured 8 angles from position ({x:.0f}, {y:.0f})")

        print(f"\n{'='60}")
        print(f" SCAN COMPLETE: {total_scans} depth images captured")
        print(f"    Breakdown: {len(scan_heights)} floors  8 angles")
        print(f"{'='60}")

        self._process_depth_scans(all_depth_data, scan_radius)

        self._detect_floors()

        print(f" Found {len(self.floor_heights)} floors")
        print(f" Detected {len(self.static_obstacles)} static obstacles")

        return self.static_obstacles, self.floor_heights

    def _process_depth_scans(self, depth_data, scan_radius):

        print(f"\n Processing {len(depth_data)} depth scans into 3D point cloud...")
        points_added = 0

        for scan_data in depth_data:
            if len(scan_data) == 3:
                angle, depth_img, scan_height = scan_data
                scan_x, scan_y = -60, 30
            else:
                angle, depth_img, scan_height, scan_x, scan_y = scan_data

            if depth_img is None or depth_img.size == 0:
                continue

            h, w = depth_img.shape

            fx = fy = w / (2  np.tan(np.deg2rad(90) / 2))
            cx, cy = w / 2, h / 2

            step = 5
            for i in range(0, h, step):
                for j in range(0, w, step):
                    depth = depth_img[i, j]

                    if depth = 0 or depth  scan_radius:
                        continue

                    x_cam = (j - cx)  depth / fx
                    y_cam = (i - cy)  depth / fy
                    z_cam = depth

                    angle_rad = np.deg2rad(angle)
                    x_world = scan_x + z_cam  np.cos(angle_rad) - x_cam  np.sin(angle_rad)
                    y_world = scan_y + z_cam  np.sin(angle_rad) + x_cam  np.cos(angle_rad)
                    z_world = -scan_height - y_cam

                    self.static_obstacles.append([x_world, y_world, z_world])
                    points_added += 1

        print(f" Extracted {points_added:,} 3D points from depth scans")

    def _detect_floors(self):

        if not self.static_obstacles:
            return

        obstacles = np.array(self.static_obstacles)
        z_coords = obstacles[:, 2]

        hist, bin_edges = np.histogram(z_coords, bins=50)

        threshold = np.mean(hist) + np.std(hist)
        for i, count in enumerate(hist):
            if count  threshold:
                floor_height = (bin_edges[i] + bin_edges[i + 1]) / 2
                self.floor_heights.append(floor_height)

        self.floor_heights.sort()

class OccupancyGrid3D:

    def __init__(self, bounds, resolution=0.5):

        self.resolution = resolution
        self.bounds = np.array(bounds)

        self.dims = ((self.bounds[:, 1] - self.bounds[:, 0]) / resolution).astype(int)

        self.grid = np.full(self.dims, -1, dtype=np.int8)

        self.occupied_cells = set()

        print(
            f" Created occupancy grid: {self.dims[0]}{self.dims[1]}{self.dims[2]} cells"
        )

    def world_to_grid(self, pos):

        grid_pos = ((np.array(pos) - self.bounds[:, 0]) / self.resolution).astype(int)
        return np.clip(grid_pos, 0, self.dims - 1)

    def grid_to_world(self, grid_pos):

        return grid_pos  self.resolution + self.bounds[:, 0]

    def add_obstacles(self, obstacles):

        print(f" Adding {len(obstacles)} obstacles to grid...")

        for obs in obstacles:
            grid_pos = tuple(self.world_to_grid(obs))

            if all(0 = grid_pos[i]  self.dims[i] for i in range(3)):
                self.grid[grid_pos] = 1
                self.occupied_cells.add(grid_pos)

        print(f" Grid populated with {len(self.occupied_cells)} occupied cells")

    def is_collision_free(self, pos, safety_radius=0.5):

        grid_pos = self.world_to_grid(pos)

        radius_cells = int(safety_radius / self.resolution)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    check_pos = tuple(grid_pos + np.array([dx, dy, dz]))

                    if check_pos in self.occupied_cells:
                        return False

        return True

    def raycast(self, start, end):

        start_grid = self.world_to_grid(start)
        end_grid = self.world_to_grid(end)

        points = self._bresenham_3d(start_grid, end_grid)

        for point in points:
            if tuple(point) in self.occupied_cells:
                return False

        return True

    def _bresenham_3d(self, start, end):

        points = []
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])

        xs = 1 if end[0]  start[0] else -1
        ys = 1 if end[1]  start[1] else -1
        zs = 1 if end[2]  start[2] else -1

        x, y, z = start

        if dx = dy and dx = dz:
            p1 = 2  dy - dx
            p2 = 2  dz - dx
            while x != end[0]:
                points.append(np.array([x, y, z]))
                x += xs
                if p1 = 0:
                    y += ys
                    p1 -= 2  dx
                if p2 = 0:
                    z += zs
                    p2 -= 2  dx
                p1 += 2  dy
                p2 += 2  dz
        elif dy = dx and dy = dz:
            p1 = 2  dx - dy
            p2 = 2  dz - dy
            while y != end[1]:
                points.append(np.array([x, y, z]))
                y += ys
                if p1 = 0:
                    x += xs
                    p1 -= 2  dy
                if p2 = 0:
                    z += zs
                    p2 -= 2  dy
                p1 += 2  dx
                p2 += 2  dz
        else:
            p1 = 2  dy - dz
            p2 = 2  dx - dz
            while z != end[2]:
                points.append(np.array([x, y, z]))
                z += zs
                if p1 = 0:
                    y += ys
                    p1 -= 2  dz
                if p2 = 0:
                    x += xs
                    p2 -= 2  dz
                p1 += 2  dy
                p2 += 2  dx

        points.append(end)
        return points

    def export_map(self, output_dir="data/maps", map_name=None):

        os.makedirs(output_dir, exist_ok=True)

        if map_name is None:
            timestamp = datetime.now().strftime("Ymd_HMS")
            map_name = f"airsim_map_{timestamp}"

        grid_file = os.path.join(output_dir, f"{map_name}_grid.npy")
        obstacles_file = os.path.join(output_dir, f"{map_name}_obstacles.npy")
        metadata_file = os.path.join(output_dir, f"{map_name}_metadata.json")

        np.save(grid_file, self.grid)
        print(f" Saved occupancy grid to: {grid_file}")

        obstacles_array = np.array(list(self.occupied_cells))
        if len(obstacles_array)  0:
            np.save(obstacles_file, obstacles_array)
            print(f" Saved obstacles to: {obstacles_file}")

        metadata = {
            "map_name": map_name,
            "timestamp": datetime.now().isoformat(),
            "resolution": float(self.resolution),
            "bounds": {
                "x_min": float(self.bounds[0, 0]),
                "x_max": float(self.bounds[0, 1]),
                "y_min": float(self.bounds[1, 0]),
                "y_max": float(self.bounds[1, 1]),
                "z_min": float(self.bounds[2, 0]),
                "z_max": float(self.bounds[2, 1])
            },
            "dimensions": {
                "x": int(self.dims[0]),
                "y": int(self.dims[1]),
                "z": int(self.dims[2])
            },
            "total_cells": int(np.prod(self.dims)),
            "occupied_cells": len(self.occupied_cells),
            "occupancy_rate": len(self.occupied_cells) / np.prod(self.dims),
            "files": {
                "grid": grid_file,
                "obstacles": obstacles_file,
                "metadata": metadata_file
            }
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f" Saved metadata to: {metadata_file}")

        print("\n" + "="60)
        print(" Map Export Summary")
        print("="60)
        print(f"Map name: {map_name}")
        print(f"Resolution: {self.resolution} m/cell")
        print(f"Bounds: X[{self.bounds[0,0]:.1f}, {self.bounds[0,1]:.1f}] "
              f"Y[{self.bounds[1,0]:.1f}, {self.bounds[1,1]:.1f}] "
              f"Z[{self.bounds[2,0]:.1f}, {self.bounds[2,1]:.1f}] m")
        print(f"Grid size: {self.dims[0]}  {self.dims[1]}  {self.dims[2]} cells")
        print(f"Total cells: {np.prod(self.dims):,}")
        print(f"Occupied cells: {len(self.occupied_cells):,} ({metadata['occupancy_rate']100:.2f})")
        print("="60)

        return metadata

class MapGenerator:

    def __init__(self, resolution=0.5):

        self.resolution = resolution
        self.client = None
        self.mapper = None
        self.grid = None

    def connect_airsim(self):

        print(" Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print(" Connected to AirSim")

    def generate_map(self, scan_heights=None, scan_radius=80):

        if scan_heights is None:
            scan_heights = [3, 9, 15, 21, 27]
        if self.client is None:
            self.connect_airsim()

        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print("\n Taking off for environment scan...")
        self.client.takeoffAsync().join()
        time.sleep(2)

        self.mapper = BuildingMapper(self.client, resolution=self.resolution)
        static_obstacles, floor_heights = self.mapper.scan_environment(
            scan_heights=scan_heights,
            scan_radius=scan_radius
        )

        print("\n  Building occupancy grid...")
        if static_obstacles:
            obs_array = np.array(static_obstacles)
            bounds = np.array([
                [obs_array[:, 0].min() - 5, obs_array[:, 0].max() + 5],
                [obs_array[:, 1].min() - 5, obs_array[:, 1].max() + 5],
                [obs_array[:, 2].min() - 2, obs_array[:, 2].max() + 2]
            ])
        else:
            bounds = np.array([[-50, 50], [-50, 50], [-20, 5]])

        self.grid = OccupancyGrid3D(bounds, resolution=self.resolution)
        self.grid.add_obstacles(static_obstacles)

        print("\n Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

        return self.grid

    def export_map(self, output_dir="data/maps", map_name=None):

        if self.grid is None:
            raise ValueError("No map generated yet. Call generate_map() first.")

        return self.grid.export_map(output_dir=output_dir, map_name=map_name)

    staticmethod
    def load_map(metadata_file):

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        bounds = np.array([
            [metadata['bounds']['x_min'], metadata['bounds']['x_max']],
            [metadata['bounds']['y_min'], metadata['bounds']['y_max']],
            [metadata['bounds']['z_min'], metadata['bounds']['z_max']]
        ])

        grid = OccupancyGrid3D(bounds, resolution=metadata['resolution'])

        grid_file = metadata['files']['grid']
        if os.path.exists(grid_file):
            grid.grid = np.load(grid_file)

        obstacles_file = metadata['files']['obstacles']
        if os.path.exists(obstacles_file):
            obstacles_array = np.load(obstacles_file)
            grid.occupied_cells = set(map(tuple, obstacles_array))

        print(f" Loaded map: {metadata['map_name']}")
        print(f"   Grid size: {grid.dims[0]}  {grid.dims[1]}  {grid.dims[2]}")
        print(f"   Occupied cells: {len(grid.occupied_cells):,}")

        return grid, metadata

def main():

    print("="  60)
    print("  AirSim ULTRA-DENSE Map Generator")
    print("    TELEPORT MODE: Safe + Instant")
    print("    STRATEGY: Dense grid per floor")
    print("="  60)

    generator = MapGenerator(resolution=1.0)

    try:
        print("\n Scanning Configuration:")
        print("   Floors: 5 (3m, 9m, 15m, 21m, 27m)")
        print("   Position: Building center (-60, 30)")
        print("   Angles/floor: 8 (45 intervals)")
        print("   Total scans: 40")
        print("   Expected map quality: OPTIMIZED\n")

        _grid = generator.generate_map(
            scan_heights=[3, 9, 15, 21, 27],
            scan_radius=60
        )

        metadata = generator.export_map(
            output_dir="data/maps",
            map_name="building_5floors"
        )

        print("\n Map generation completed successfully!")
        print(f"Map files saved to: {metadata['files']['metadata']}")

        total_cells = metadata['total_cells']
        print("\n A Performance Estimate:")
        print(f"   Total cells: {total_cells:,}")
        if total_cells  5_000_000:
            print("   Expected A planning time: 1-10s  FAST")
        elif total_cells  15_000_000:
            print("   Expected A planning time: 10-30s  MODERATE")
        else:
            print("   Expected A planning time:  1 minute  SLOW")
            print("    Recommendation: Increase resolution to 1.5m or 2.0m")

    except Exception as e:
        print(f"\n Map generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
