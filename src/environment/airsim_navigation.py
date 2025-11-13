"""
AirSim-Unreal Engine Map Generator
Generates 3D occupancy grid map from AirSim environment for baseline algorithms.
Exports map data for A*, RRT, and other planning algorithms.
"""

import airsim
import numpy as np
import time
import json
import os
from datetime import datetime


class BuildingMapper:
    """Phase 1: Load and process building information from Unreal Engine"""

    def __init__(self, client, resolution=0.5):
        self.client = client
        self.resolution = resolution  # meters per grid cell
        self.building_bounds = None
        self.floor_heights = []
        self.static_obstacles = []

    def scan_environment(self, scan_heights=None, scan_radius=100):
        """
        Scan environment using depth camera from multiple heights.
        Scans at 5 different Z positions for better coverage.
        
        Args:
            scan_heights: List of Z heights to scan from (meters, NED: negative=up)
                         Default: [3, 9, 15, 21, 27] meters above ground
            scan_radius: Radius of scanning area (meters)
        """
        if scan_heights is None:
            # Default: 5 scan positions at different heights
            scan_heights = [3, 9, 15, 21, 27]  # meters
        
        print("🏗️  Phase 1: Scanning building environment...")
        print(f"   Scan strategy: {len(scan_heights)} positions at Z = {scan_heights} m")
        
        # Collect depth data from all scan positions
        all_depth_data = []
        
        for scan_height in scan_heights:
            print(f"\n📍 Scanning from height: {scan_height}m (position {scan_heights.index(scan_height)+1}/{len(scan_heights)})")
            
            # Move drone to scanning position (x=-60, y=30, z=-height)
            # Scan center at building location, then scan at different heights
            self.client.moveToPositionAsync(-60, 30, -scan_height, 5).join()
            time.sleep(1)

            # Get depth images from multiple angles at this height
            angles = np.linspace(0, 360, 8, endpoint=False)

            for angle in angles:
                # Rotate drone
                self.client.rotateToYawAsync(angle, 2).join()
                time.sleep(0.5)

                # Capture depth image
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
                    # Store with scan height info
                    all_depth_data.append((angle, depth_img, scan_height))
        
        print(f"\n✅ Captured {len(all_depth_data)} depth images from {len(scan_heights)} heights")

        # Process all depth data to extract obstacles
        self._process_depth_scans(all_depth_data, scan_radius)

        # Detect floor levels
        self._detect_floors()

        print(f"✅ Found {len(self.floor_heights)} floors")
        print(f"✅ Detected {len(self.static_obstacles)} static obstacles")

        return self.static_obstacles, self.floor_heights

    def _process_depth_scans(self, depth_data, scan_radius):
        """
        Convert depth images to 3D point cloud.
        
        Args:
            depth_data: List of (angle, depth_img, scan_height) tuples
            scan_radius: Maximum depth range to consider
        """
        for angle, depth_img, scan_height in depth_data:
            if depth_img is None or depth_img.size == 0:
                continue

            h, w = depth_img.shape

            # Camera intrinsics (approximate for AirSim default camera)
            fx = fy = w / (2 * np.tan(np.deg2rad(90) / 2))
            cx, cy = w / 2, h / 2

            # Sample points (use every Nth pixel to reduce computation)
            step = 10
            for i in range(0, h, step):
                for j in range(0, w, step):
                    depth = depth_img[i, j]

                    # Filter out invalid depths
                    if depth <= 0 or depth > scan_radius:
                        continue

                    # Convert to 3D point in camera frame
                    x_cam = (j - cx) * depth / fx
                    y_cam = (i - cy) * depth / fy
                    z_cam = depth

                    # Rotate to world frame based on drone orientation
                    angle_rad = np.deg2rad(angle)
                    # Coordinates relative to scan center at (-60, 30)
                    x_world = -60 + z_cam * np.cos(angle_rad) - x_cam * np.sin(angle_rad)
                    y_world = 30 + z_cam * np.sin(angle_rad) + x_cam * np.cos(angle_rad)
                    z_world = -scan_height - y_cam

                    self.static_obstacles.append([x_world, y_world, z_world])

    def _detect_floors(self):
        """Detect floor levels from obstacle point cloud"""
        if not self.static_obstacles:
            return

        obstacles = np.array(self.static_obstacles)
        z_coords = obstacles[:, 2]

        # Histogram-based floor detection
        hist, bin_edges = np.histogram(z_coords, bins=50)

        # Find peaks in histogram (potential floors)
        threshold = np.mean(hist) + np.std(hist)
        for i, count in enumerate(hist):
            if count > threshold:
                floor_height = (bin_edges[i] + bin_edges[i + 1]) / 2
                self.floor_heights.append(floor_height)

        self.floor_heights.sort()


class OccupancyGrid3D:
    """Phase 2: Build 3D occupancy grid from UE data"""

    def __init__(self, bounds, resolution=0.5):
        """
        Args:
            bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            resolution: grid cell size in meters
        """
        self.resolution = resolution
        self.bounds = np.array(bounds)

        # Calculate grid dimensions
        self.dims = ((self.bounds[:, 1] - self.bounds[:, 0]) / resolution).astype(int)

        # Initialize occupancy grid (0=free, 1=occupied, -1=unknown)
        self.grid = np.full(self.dims, -1, dtype=np.int8)

        # For faster collision checking
        self.occupied_cells = set()

        print(
            f"📐 Created occupancy grid: {self.dims[0]}×{self.dims[1]}×{self.dims[2]} cells"
        )

    def world_to_grid(self, pos):
        """Convert world coordinates to grid indices"""
        grid_pos = ((np.array(pos) - self.bounds[:, 0]) / self.resolution).astype(int)
        return np.clip(grid_pos, 0, self.dims - 1)

    def grid_to_world(self, grid_pos):
        """Convert grid indices to world coordinates"""
        return grid_pos * self.resolution + self.bounds[:, 0]

    def add_obstacles(self, obstacles):
        """Add obstacles to occupancy grid"""
        print(f"🔨 Adding {len(obstacles)} obstacles to grid...")

        for obs in obstacles:
            grid_pos = tuple(self.world_to_grid(obs))

            if all(0 <= grid_pos[i] < self.dims[i] for i in range(3)):
                self.grid[grid_pos] = 1
                self.occupied_cells.add(grid_pos)

        print(f"✅ Grid populated with {len(self.occupied_cells)} occupied cells")

    def is_collision_free(self, pos, safety_radius=0.5):
        """Check if position is collision-free with safety margin"""
        grid_pos = self.world_to_grid(pos)

        # Check surrounding cells within safety radius
        radius_cells = int(safety_radius / self.resolution)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for dz in range(-radius_cells, radius_cells + 1):
                    check_pos = tuple(grid_pos + np.array([dx, dy, dz]))

                    if check_pos in self.occupied_cells:
                        return False

        return True

    def raycast(self, start, end):
        """Bresenham's 3D line algorithm for collision checking"""
        start_grid = self.world_to_grid(start)
        end_grid = self.world_to_grid(end)

        # Get all points along the line
        points = self._bresenham_3d(start_grid, end_grid)

        for point in points:
            if tuple(point) in self.occupied_cells:
                return False

        return True

    def _bresenham_3d(self, start, end):
        """3D Bresenham line algorithm"""
        points = []
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        dz = abs(end[2] - start[2])

        xs = 1 if end[0] > start[0] else -1
        ys = 1 if end[1] > start[1] else -1
        zs = 1 if end[2] > start[2] else -1

        x, y, z = start

        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x != end[0]:
                points.append(np.array([x, y, z]))
                x += xs
                if p1 >= 0:
                    y += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y != end[1]:
                points.append(np.array([x, y, z]))
                y += ys
                if p1 >= 0:
                    x += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z != end[2]:
                points.append(np.array([x, y, z]))
                z += zs
                if p1 >= 0:
                    y += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx

        points.append(end)
        return points

    def export_map(self, output_dir="data/maps", map_name=None):
        """
        Export occupancy grid and metadata for baseline algorithms.
        
        Args:
            output_dir: Directory to save map files
            map_name: Name for this map (default: timestamp)
            
        Returns:
            dict: Map metadata including file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate map name if not provided
        if map_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_name = f"airsim_map_{timestamp}"
        
        # File paths
        grid_file = os.path.join(output_dir, f"{map_name}_grid.npy")
        obstacles_file = os.path.join(output_dir, f"{map_name}_obstacles.npy")
        metadata_file = os.path.join(output_dir, f"{map_name}_metadata.json")
        
        # Save occupancy grid (numpy array)
        np.save(grid_file, self.grid)
        print(f"✅ Saved occupancy grid to: {grid_file}")
        
        # Save obstacles list (for visualization)
        obstacles_array = np.array(list(self.occupied_cells))
        if len(obstacles_array) > 0:
            np.save(obstacles_file, obstacles_array)
            print(f"✅ Saved obstacles to: {obstacles_file}")
        
        # Create metadata
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
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved metadata to: {metadata_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Map Export Summary")
        print("="*60)
        print(f"Map name: {map_name}")
        print(f"Resolution: {self.resolution} m/cell")
        print(f"Bounds: X[{self.bounds[0,0]:.1f}, {self.bounds[0,1]:.1f}] " 
              f"Y[{self.bounds[1,0]:.1f}, {self.bounds[1,1]:.1f}] "
              f"Z[{self.bounds[2,0]:.1f}, {self.bounds[2,1]:.1f}] m")
        print(f"Grid size: {self.dims[0]} × {self.dims[1]} × {self.dims[2]} cells")
        print(f"Total cells: {np.prod(self.dims):,}")
        print(f"Occupied cells: {len(self.occupied_cells):,} ({metadata['occupancy_rate']*100:.2f}%)")
        print("="*60)
        
        return metadata


class MapGenerator:
    """
    Main interface for generating maps from AirSim environment.
    Used by baseline algorithms (A*, RRT, etc.)
    """
    
    def __init__(self, resolution=0.5):
        """
        Args:
            resolution: Grid cell size in meters
        """
        self.resolution = resolution
        self.client = None
        self.mapper = None
        self.grid = None
        
    def connect_airsim(self):
        """Connect to AirSim and initialize drone"""
        print("🔗 Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ Connected to AirSim")
        
    def generate_map(self, scan_heights=None, scan_radius=80):
        """
        Generate 3D occupancy grid from AirSim environment.
        
        Args:
            scan_heights: List of heights to scan from (meters)
                         Default: [3, 9, 15, 21, 27] - scans at 5 different levels
            scan_radius: Scanning radius (meters)
            
        Returns:
            OccupancyGrid3D: Generated occupancy grid
        """
        if scan_heights is None:
            scan_heights = [3, 9, 15, 21, 27]
        if self.client is None:
            self.connect_airsim()
        
        # Enable API control
        self.client.enableApiControl(True) #type: ignore
        self.client.armDisarm(True)  # type: ignore

        # Takeoff
        print("\n🛫 Taking off for environment scan...")
        self.client.takeoffAsync().join()  # type: ignore
        time.sleep(2)
        
        # Scan environment from multiple heights
        self.mapper = BuildingMapper(self.client, resolution=self.resolution)
        static_obstacles, floor_heights = self.mapper.scan_environment(
            scan_heights=scan_heights,
            scan_radius=scan_radius
        )
        
        # Build occupancy grid
        print("\n🏗️  Building occupancy grid...")
        if static_obstacles:
            obs_array = np.array(static_obstacles)
            bounds = np.array([
                [obs_array[:, 0].min() - 5, obs_array[:, 0].max() + 5],
                [obs_array[:, 1].min() - 5, obs_array[:, 1].max() + 5],
                [obs_array[:, 2].min() - 2, obs_array[:, 2].max() + 2]
            ])
        else:
            # Fallback bounds if no obstacles detected
            bounds = np.array([[-50, 50], [-50, 50], [-20, 5]])
        
        self.grid = OccupancyGrid3D(bounds, resolution=self.resolution)
        self.grid.add_obstacles(static_obstacles)
        
        # Land
        print("\n🛬 Landing...")
        self.client.landAsync().join() # type: ignore
        self.client.armDisarm(False)  # type: ignore
        self.client.enableApiControl(False)  # type: ignore
        
        return self.grid
    
    def export_map(self, output_dir="data/maps", map_name=None):
        """
        Export generated map to files.
        
        Args:
            output_dir: Directory to save map files
            map_name: Name for this map
            
        Returns:
            dict: Map metadata
        """
        if self.grid is None:
            raise ValueError("No map generated yet. Call generate_map() first.")
        
        return self.grid.export_map(output_dir=output_dir, map_name=map_name)
    
    @staticmethod
    def load_map(metadata_file):
        """
        Load previously generated map from files.
        
        Args:
            metadata_file: Path to map metadata JSON file
            
        Returns:
            OccupancyGrid3D: Loaded occupancy grid
        """
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct bounds
        bounds = np.array([
            [metadata['bounds']['x_min'], metadata['bounds']['x_max']],
            [metadata['bounds']['y_min'], metadata['bounds']['y_max']],
            [metadata['bounds']['z_min'], metadata['bounds']['z_max']]
        ])
        
        # Create grid
        grid = OccupancyGrid3D(bounds, resolution=metadata['resolution'])
        
        # Load occupancy data
        grid_file = metadata['files']['grid']
        if os.path.exists(grid_file):
            grid.grid = np.load(grid_file)
        
        # Load obstacles
        obstacles_file = metadata['files']['obstacles']
        if os.path.exists(obstacles_file):
            obstacles_array = np.load(obstacles_file)
            grid.occupied_cells = set(map(tuple, obstacles_array))
        
        print(f"✅ Loaded map: {metadata['map_name']}")
        print(f"   Grid size: {grid.dims[0]} × {grid.dims[1]} × {grid.dims[2]}")
        print(f"   Occupied cells: {len(grid.occupied_cells):,}")
        
        return grid, metadata


def main():
    """Generate map from AirSim environment"""
    
    print("=" * 60)
    print("🗺️  AirSim Map Generator for Baseline Algorithms")
    print("=" * 60)
    
    # Create map generator
    # OPTIMIZED: Use 1.0m resolution for faster A* planning
    # This reduces grid size by 8x (0.5m → 1.0m = 2^3)
    # Trade-off: Slightly less detailed map, but A* is 20-50x faster
    generator = MapGenerator(resolution=1.0)
    
    try:
        # Generate map from AirSim with multi-height scanning
        # Scan at 5 different heights: 3m, 9m, 15m, 21m, 27m
        # OPTIMIZED: Reduced scan_radius to 60m for smaller grid
        _grid = generator.generate_map(
            scan_heights=[3, 9, 15, 21, 27],
            scan_radius=60  # Reduced from 80m for faster processing
        )
        
        # Export map
        metadata = generator.export_map(
            output_dir="data/maps",
            map_name="building_5floors"
        )
        
        print("\n✅ Map generation completed successfully!")
        print(f"Map files saved to: {metadata['files']['metadata']}")
        
        # Performance estimate for A* planning
        total_cells = metadata['total_cells']
        print("\n📊 A* Performance Estimate:")
        print(f"   Total cells: {total_cells:,}")
        if total_cells < 5_000_000:
            print("   Expected A* planning time: 1-10s ✅ FAST")
        elif total_cells < 15_000_000:
            print("   Expected A* planning time: 10-30s ⚠️ MODERATE")
        else:
            print("   Expected A* planning time: > 1 minute ❌ SLOW")
            print("   💡 Recommendation: Increase resolution to 1.5m or 2.0m")
        
    except Exception as e:
        print(f"\n❌ Map generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
