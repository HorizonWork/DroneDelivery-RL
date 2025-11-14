Script build_fly_zone.py có 2 chế độ lấy map:

1. Synthetic Obstacles (mặc định) - Giả định cứng về cấu trúc tòa nhà
2. AirSim Real Map (--use-airsim) - Lấy map thực tế từ UE environment

Mở file /Documents/AirSim/settings.json (Windows: C:\Users\YourName\Documents\AirSim\settings.json):

json
{
  "SeeDocsAt": "https:
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",

  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": -15,

      "Sensors": {
        "Lidar1": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "PointsPerSecond": 100000,
          "Range": 50,
          "RotationsPerSecond": 10,
          "VerticalFOVUpper": 15,
          "VerticalFOVLower": -25,
          "HorizontalFOVStart": 0,
          "HorizontalFOVEnd": 359,
          "DrawDebugPoints": false,
          "DataFrame": "VehicleInertialFrame"
        }
      }
    }
  }
}

Sau khi save settings.json, restart Unreal Engine environment.

bash
conda activate drone-delivery-rl
python src/utils/build_fly_zone.py --use-airsim

Script sẽ:
1. Connect AirSim
2. Di chuyển drone qua nhiều vị trí trong không gian
3. Quét LiDAR tại mỗi vị trí
4. Kết hợp point clouds thành occupancy grid

Ưu điểm:
-  Lấy map thực tế từ UE
-  Phát hiện tất cả obstacles (walls, furniture, ceilings...)
-  Không bị msgpack-rpc error (chỉ query LiDAR data)

Nhược điểm:
-  Mất 5-10 phút cho full scan
-  Cần LiDAR sensor configured

---

Nếu AirSim có lỗi msgpack-rpc, script tự động chuyển sang synthetic:

bash
python src/utils/build_fly_zone.py --use-airsim

Output:

2025-11-13 03:00:00 - INFO - Attempting AirSim connection...
2025-11-13 03:00:00 - INFO - Connected to AirSim successfully
2025-11-13 03:00:00 - INFO - Building grid from AirSim using LiDAR scans...
2025-11-13 03:00:05 - WARNING - LiDAR not available: 'MultirotorClient' has no attribute 'getLidarData'
2025-11-13 03:00:05 - WARNING - Failed to get voxel grid, falling back to cell-by-cell query
2025-11-13 03:00:10 - WARNING - AirSim connection lost (msgpack-rpc error), switching to synthetic

---

Nguyên nhân: Chưa configure LiDAR sensor trong settings.json

Giải pháp:
bash
cat /Documents/AirSim/settings.json

Nguyên nhân: LiDAR range quá ngắn hoặc drone ở vị trí không có obstacles

Giải pháp:
json
{
  "Sensors": {
    "Lidar1": {
      "Range": 100,
      "NumberOfChannels": 32,
      "PointsPerSecond": 200000
    }
  }
}

Giải pháp 1: Giảm resolution
bash
python src/utils/build_fly_zone.py --use-airsim --cell-size 1.0

Giải pháp 2: Giảm số scan points
python
scan_step = max(1, min(10, z_count

---

 Method  Thời gian  Độ chính xác  Phụ thuộc UE  Stable
-------------------------------------------------------
 Synthetic  1s   Generic   No   100
 LiDAR Scan  5-10 min   Thực tế   Yes   90
 Cell-by-cell  5 min   Thực tế   Yes   0 (msgpack)

---

bash
python src/utils/build_fly_zone.py

bash
python src/utils/build_fly_zone.py --use-airsim

bash
python src/utils/build_fly_zone.py --use-airsim --origin 0 0 20 --size 50 50 40

python src/utils/build_fly_zone.py --use-airsim --output data/maps/my_building

---

Sau khi build grid, kiểm tra visualizations:

bash
ls results/visualizations/grid_slice_.png

python -c "
import numpy as np
import matplotlib.pyplot as plt

grid = np.load('data/maps/occupancy_grid.npy')
print(f'Shape: {grid.shape}')
print(f'Occupied: {np.sum(grid)} / {grid.size}')
print(f'Free: {grid.size - np.sum(grid)}')

plt.imshow(grid[:, :, grid.shape[2]
plt.title('Middle Z slice')
plt.show()
"

---

1. LiDAR scan tốt nhất cho environment phức tạp (nhiều furniture, cấu trúc chi tiết)
2. Synthetic tốt cho testing nhanh (structure đơn giản, reproducible)
3. Grid được cache  Scan một lần, dùng nhiều lần
4. Checkpoint system  Resume nếu bị interrupt

Sau khi có grid thực tế:
bash
python scripts/training/train_phase.py --grid data/maps/occupancy_grid.npy

python scripts/evaluation/benchmark_baselines.py --grid data/maps/occupancy_grid.npy

