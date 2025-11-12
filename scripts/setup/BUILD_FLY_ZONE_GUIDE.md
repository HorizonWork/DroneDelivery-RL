# Build Fly Zone - Hướng Dẫn Sử Dụng

## Tổng Quan

Script `build_fly_zone.py` tạo 3D occupancy grid cho drone navigation trong môi trường AirSim/UE. Grid này là nền tảng cho A* và S-RRT planners.

## Cải Tiến So Với Phiên Bản Cũ

### ✅ Điểm Mới Được Bổ Sung

1. **Validation Input Config**
   - Kiểm tra `cell_size > 0`
   - Kiểm tra `size` dimensions hợp lệ
   - Cảnh báo nếu `cell_size` quá lớn

2. **AirSim Integration Thực Tế**
   - Sử dụng `simGetCollisionInfo()` để detect collisions
   - Ray casting với `simTestLineOfSightBetweenPoints()`
   - Coordinate conversion UE (cm) ↔ meters

3. **Checkpoint/Resume Mechanism**
   - Lưu progress mỗi 10 layers
   - Resume từ checkpoint khi bị interrupt
   - Clear checkpoint khi hoàn thành

4. **Better Error Handling**
   - Fallback từ AirSim → synthetic khi API fail
   - Debug logging cho từng cell check
   - Exception handling đầy đủ

## Cách Sử Dụng

### 1. Basic Usage (Synthetic Obstacles)

```bash
conda activate drone_delivery_rl
python scripts/setup/build_fly_zone.py
```

Output:
- `data/maps/occupancy_grid.npy` - Grid data
- `data/maps/occupancy_grid_metadata.json` - Metadata
- `results/visualizations/grid_slice_z*.png` - Visualizations

### 2. Với AirSim Connection

```bash
# Bật UE + AirSim trước
python scripts/setup/build_fly_zone.py --use-airsim
```

### 3. Custom Configuration

```bash
python scripts/setup/build_fly_zone.py \
    --origin 0 0 20 \
    --size 30 30 40 \
    --cell-size 0.25 \
    --drone-radius 0.4 \
    --use-airsim
```

### 4. Resume From Checkpoint

```bash
# Nếu bị interrupt, chạy lại lệnh cũ
python scripts/setup/build_fly_zone.py
# Sẽ tự động resume từ checkpoint
```

### 5. Disable Checkpoint (RAM-limited)

```bash
python scripts/setup/build_fly_zone.py --no-checkpoint
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--origin` | `0 0 15` | Grid origin (x, y, z) in meters |
| `--size` | `20 20 30` | Grid size (x, y, z) in meters |
| `--cell-size` | `0.5` | Cell resolution in meters |
| `--drone-radius` | `0.3` | Safety radius in meters |
| `--use-airsim` | `False` | Use AirSim API for collision detection |
| `--no-checkpoint` | `False` | Disable checkpoint/resume |
| `--output` | `data/maps` | Output directory |
| `--no-viz` | `False` | Skip visualization generation |

## Grid Convention & Indexing

### Coordinate System

```
Grid Index:     World Coordinates:
i → X axis      x (meters)
j → Y axis      y (meters) 
k → Z axis      z (meters, up positive)

UE Coordinates (AirSim API):
X (cm) = x * 100
Y (cm) = y * 100
Z (cm) = -z * 100  ⚠️ Z inverted
```

### Cell Status

- `grid[i, j, k] = 0` → **Free** (drone có thể bay qua)
- `grid[i, j, k] = 1` → **Occupied** (có obstacle hoặc ngoài boundary)

### Grid-to-World Conversion

```python
# Cell center
x = origin[0] - size[0]/2 + (i + 0.5) * cell_size
y = origin[1] - size[1]/2 + (j + 0.5) * cell_size
z = origin[2] + (k + 0.5) * cell_size
```

## Synthetic Obstacles Logic

### 1. Floors (Horizontal Slabs)
```python
floor_height = 3.0m
floor_thickness = 0.3m
# Occupied nếu: |z - floor_z| < 0.3m
# Exception: Stairwell positions
```

### 2. Walls (Vertical Boundaries)
```python
wall_thickness = 0.2m
# Occupied nếu gần boundary của grid
```

### 3. Stairwells (4 vị trí)
```python
# Ở 4 góc, kích thước 2m × 2m
# Free cells xuyên qua các tầng
```

## AirSim API Integration

### Collision Detection Flow

```python
1. Convert (x,y,z) meters → UE coordinates (cm)
2. Check simGetCollisionInfo() distance
3. Ray cast 6 directions (±X, ±Y, ±Z)
4. If any ray blocked → Occupied
5. Fallback to synthetic if API fails
```

### API Functions Used

- `simGetCollisionInfo()` - Current collision state
- `simTestLineOfSightBetweenPoints(pos1, pos2)` - Ray casting
- `airsim.Vector3r(x, y, z)` - Position vector

## Output Files

### 1. `occupancy_grid.npy`
```python
import numpy as np
grid = np.load("data/maps/occupancy_grid.npy")
print(grid.shape)  # (40, 40, 60) với default config
```

### 2. `occupancy_grid_metadata.json`
```json
{
  "origin": [0.0, 0.0, 15.0],
  "size": [20.0, 20.0, 30.0],
  "cell_size": 0.5,
  "dimensions": [40, 40, 60],
  "occupied_cells": 15234,
  "free_cells": 80766
}
```

### 3. Visualization Slices
- `grid_slice_z0.png` - Ground level
- `grid_slice_z15.png` - Floor 2
- `grid_slice_z30.png` - Floor 3
- `grid_slice_z45.png` - Floor 4

## Integration với Planners

### A* Planner
```python
from src.planning import AStarPlanner
import numpy as np

grid = np.load("data/maps/occupancy_grid.npy")
planner = AStarPlanner(config)
planner.occupancy_grid.update_grid(grid)

# Plan path
path = planner.plan(start_pos, goal_pos)
```

### S-RRT Planner
```python
from src.planning import SRRTPlanner
import numpy as np

grid = np.load("data/maps/occupancy_grid.npy")
planner = SRRTPlanner(config)
planner.set_occupancy_grid(grid)

# Plan with clearance cost
path = planner.plan_safe_path(start, goal, obstacles)
```

## Performance Tips

### 1. Cell Size Selection
- **0.5m** → Balance speed/accuracy (default, như report)
- **0.25m** → Higher accuracy, 8x cells, slower
- **1.0m** → Fast prototyping, coarse grid

### 2. Memory Usage
```python
Memory = x_cells × y_cells × z_cells × 1 byte
Default (40×40×60) = 96 KB ✓
Fine (80×80×120) = 768 KB ✓
Very fine (160×160×240) = 6 MB ⚠️
```

### 3. Checkpoint Frequency
```python
# Trong build_grid()
if (k + 1) % 10 == 0:  # Mỗi 10 layers
    self._save_checkpoint(grid, k + 1)
```
- Mỗi 10 layers ~ 5-10s
- Trade-off: frequency vs disk I/O

## Troubleshooting

### Issue 1: "AirSim not connected"
```bash
# Check AirSim running
# Fallback to synthetic automatically
# Hoặc thêm --use-airsim nếu AirSim ready
```

### Issue 2: "cell_size too large"
```bash
# Giảm cell_size:
python build_fly_zone.py --cell-size 0.25
```

### Issue 3: Memory Error
```bash
# Giảm grid size hoặc tăng cell_size:
python build_fly_zone.py --size 15 15 20 --cell-size 0.5
```

### Issue 4: Resume không hoạt động
```bash
# Xóa checkpoint và build lại:
rm -rf temp/grid_checkpoints
python build_fly_zone.py
```

## Validation Checklist

Sau khi build grid, kiểm tra:

- [ ] File `occupancy_grid.npy` tồn tại
- [ ] Metadata có `free_cells > 0`
- [ ] Visualizations hiển thị floors/walls rõ ràng
- [ ] Stairwells xuất hiện trong slices
- [ ] Free space ratio ~ 80-85% (hợp lý)

## Next Steps

1. **Train planners với grid:**
   ```bash
   python scripts/training/train_phase.py --config config/phase1_simple.yaml
   ```

2. **Evaluate baselines:**
   ```bash
   python scripts/evaluation/benchmark_baselines.py
   ```

3. **Tích hợp vào environment:**
   ```python
   from src.environment import DroneDeliveryEnv
   env = DroneDeliveryEnv(config)
   env.load_occupancy_grid("data/maps/occupancy_grid.npy")
   ```

## References

- AirSim API: https://microsoft.github.io/AirSim/api_docs/html/
- Occupancy Grid Mapping: Thrun et al., Probabilistic Robotics
- 3D Navigation Grid: Report Section 3.2.1, Table 1
