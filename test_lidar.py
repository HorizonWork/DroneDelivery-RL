import airsim
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim")

client.enableApiControl(True)
client.armDisarm(True)

print("\nMoving to scan position...")
client.moveToPositionAsync(0, 0, -20, 5).join()
print("Position: (0, 0, 20m above ground)")

try:
    lidar_data = client.getLidarData("Lidar1")

    if len(lidar_data.point_cloud)  3:
        print("\n ERROR: LiDAR không thu được điểm nào!")
        print("   Kiểm tra:")
        print("   1. Enabled = true trong settings.json?")
        print("   2. Drone có ở giữa tòa nhà không?")
        print("   3. Range = 50m đủ lớn không?")
    else:
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)
        print(f"\n LiDAR hoạt động!")
        print(f"   Số điểm: {len(points)}")
        print(f"   Point cloud range:")
        print(f"   - X: [{points[:,0].min():.1f}, {points[:,0].max():.1f}] cm")
        print(f"   - Y: [{points[:,1].min():.1f}, {points[:,1].max():.1f}] cm")
        print(f"   - Z: [{points[:,2].min():.1f}, {points[:,2].max():.1f}] cm")

        points_world = points / 100
        points_world[:, 2] = -points_world[:, 2]

        print(f"\n   World coordinates (meters):")
        print(f"   - X: [{points_world[:,0].min():.2f}, {points_world[:,0].max():.2f}]")
        print(f"   - Y: [{points_world[:,1].min():.2f}, {points_world[:,1].max():.2f}]")
        print(f"   - Z: [{points_world[:,2].min():.2f}, {points_world[:,2].max():.2f}]")

except Exception as e:
    print(f"\n ERROR: {e}")
    print("   LiDAR sensor có tên 'Lidar1' trong settings.json không?")

print("\nLanding...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
