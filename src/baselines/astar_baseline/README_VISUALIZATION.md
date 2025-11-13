# A* Path Visualization Guide

## 📖 Tổng Quan

Guide này hướng dẫn bạn sử dụng baseline A* để tạo visualization đường bay cho drone trong môi trường Unreal Engine của bạn.

**Script chính:** `visualize_astar_path.py`

**Chức năng:**
- ✅ Load DroneSpawn và Landing_XXX actors từ UE
- ✅ Plan A* path trên 3D occupancy grid
- ✅ Execute path với PID controller
- ✅ **Visualize real-time trong UE** (đường màu đỏ = planned, màu xanh = actual)

---

## 🚀 Cách Sử Dụng

### **Bước 1: Generate Map từ UE Environment**

Trước tiên, bạn cần generate occupancy grid map từ môi trường UE của mình:

```bash
# Đảm bảo UE + AirSim đang chạy
python src/environment/airsim_navigation.py
```

**Kết quả:**
- Map file: `data/maps/building_5floors_metadata.json`
- Grid file: `data/maps/building_5floors_grid.npy`
- Obstacles file: `data/maps/building_5floors_obstacles.npy`

**⚠️ QUAN TRỌNG:** Script này sẽ scan environment ở 5 độ cao khác nhau (3m, 9m, 15m, 21m, 27m) để build 3D occupancy grid. Quá trình này mất ~2-3 phút.

---

### **Bước 2: Setup UE Scene với Actors**

Trong Unreal Engine scene của bạn, đảm bảo có các actors sau:

#### **DroneSpawn Actor**
- **Name:** `DroneSpawn`
- **Type:** Empty Actor hoặc Target Point
- **Location:** Vị trí spawn của drone (ví dụ: X=0, Y=0, Z=0)

#### **Landing Target Actors**
Tạo 30 landing targets với naming convention:
```
Landing_101, Landing_102, Landing_103, Landing_104, Landing_105, Landing_106  (Floor 1)
Landing_201, Landing_202, Landing_203, Landing_204, Landing_205, Landing_206  (Floor 2)
Landing_301, Landing_302, Landing_303, Landing_304, Landing_305, Landing_306  (Floor 3)
Landing_401, Landing_402, Landing_403, Landing_404, Landing_405, Landing_406  (Floor 4)
Landing_501, Landing_502, Landing_503, Landing_504, Landing_505, Landing_506  (Floor 5)
```

**💡 Tips:**
- Đặt actors ở các vị trí khác nhau trên mỗi tầng
- Đảm bảo Z-coordinate tăng dần theo floor (Floor 1 < Floor 2 < ... < Floor 5)
- Actor names phải chính xác (case-sensitive!)

**Cách tạo nhanh trong UE:**
1. Tạo 1 TargetPoint actor
2. Đặt tên `Landing_101`
3. Duplicate (Ctrl+W) và rename thành `Landing_102`, `Landing_103`, etc.
4. Di chuyển đến vị trí mong muốn

---

### **Bước 3: Run Visualization**

```bash
# Start Unreal Engine + AirSim
# Sau đó:

python src/baselines/astar_baseline/visualize_astar_path.py
```

**Script sẽ:**
1. Connect to AirSim
2. Load tất cả actor positions từ UE
3. Takeoff
4. Run 3 missions đến random targets (có thể thay đổi số lượng)
5. Mỗi mission:
   - Plan A* path
   - **Vẽ planned path (màu đỏ) trong UE**
   - Execute với PID
   - **Vẽ actual trajectory (màu xanh) trong UE**
   - Print metrics
6. Land

---

## ⚙️ Customization

### **Mission đến target cụ thể:**

Mở file `visualize_astar_path.py`, tìm `main()` function, uncomment và chỉnh:

```python
# OPTION 1: Single mission to specific target
result = visualizer.run_mission(
    start_name="DroneSpawn",
    target_name="Landing_301",  # ← Chỉnh target ở đây!
    visualize=True
)
```

### **Thay đổi số lượng missions:**

```python
# OPTION 2: Multiple missions (random targets)
results = visualizer.run_multiple_missions(
    num_missions=5,  # ← Chỉnh số missions ở đây!
    visualize=True
)
```

### **Tuning PID Controller:**

Nếu drone bay không smooth, chỉnh PID gains trong `main()`:

```python
config = {
    # Position PID (X, Y, Z control)
    'position_kp': 2.0,   # ↑ Tăng = responsive hơn, giảm = smooth hơn
    'position_ki': 0.1,   # Integral (xử lý steady-state error)
    'position_kd': 0.5,   # Derivative (damping)
    
    # Yaw PID
    'yaw_kp': 1.5,
    'yaw_ki': 0.05,
    'yaw_kd': 0.3,
    
    # Limits
    'max_velocity': 5.0,      # m/s
    'max_yaw_rate': 1.0,      # rad/s
    'waypoint_tolerance': 1.0, # meters
}
```

**Troubleshooting PID:**
- **Drone oscillates:** Giảm `kp`, tăng `kd`
- **Slow to reach waypoint:** Tăng `kp`
- **Overshoots target:** Giảm `kp`, tăng `kd`
- **Drifts over time:** Tăng `ki` (nhưng cẩn thận với windup!)

### **Thay đổi A* floor penalty:**

```python
config = {
    'floor_penalty': 5.0,  # ← Cost thêm khi chuyển floor
}
```

- **Tăng:** A* sẽ ưu tiên đi ngang trong cùng 1 floor
- **Giảm:** A* dễ dàng chuyển floor hơn

---

## 🎨 Visualization Colors trong UE

Khi script chạy, bạn sẽ thấy:

| Color | Meaning |
|-------|---------|
| 🔴 **Red Line** | Planned A* path |
| 🟢 **Green Line** | Actual trajectory (what drone flew) |
| 🟡 **Yellow Spheres** | Start & Goal positions |

**💡 Tips:**
- Nếu red và green lines gần nhau → PID tracking tốt
- Nếu green line lệch nhiều khỏi red → PID cần tuning
- Nếu không thấy lines → Check console warnings

---

## 📊 Output Metrics

Sau mỗi mission, bạn sẽ thấy:

```
📈 MISSION RESULTS
=====================================================
✓ Success: YES
⏱️  Planning time: 0.142s
⏱️  Execution time: 28.45s
⏱️  Total time: 28.59s
🗺️  Path waypoints: 87
📏 Path length: 42.31 m
⚡ Energy consumed: 3.24 kJ
🎯 ATE error: 0.087 m
📍 Distance to goal: 0.31 m
=====================================================
```

**Metrics giải thích:**
- **Success:** Đạt goal trong tolerance (0.5m)
- **Planning time:** Thời gian A* tìm path
- **Execution time:** Thời gian bay thực tế
- **Path waypoints:** Số waypoints A* generate
- **Path length:** Tổng độ dài path
- **Energy:** Energy tiêu thụ (kinetic + acceleration)
- **ATE (Average Trajectory Error):** Độ lệch trung bình so với planned path
- **Distance to goal:** Khoảng cách cuối đến goal

---

## 🔍 Troubleshooting

### **Problem: "Map file not found"**
```
❌ Map file not found: data/maps/building_5floors_metadata.json
```

**Solution:** Generate map trước:
```bash
python src/environment/airsim_navigation.py
```

---

### **Problem: "No landing targets found"**
```
❌ ERROR: No landing targets found!
```

**Reasons:**
1. Actor names sai (phải là `Landing_101`, không phải `landing_101` hay `LandingPad_101`)
2. Actors chưa được tạo trong UE scene
3. AirSim không thấy actors

**Solution:**
1. Mở UE, check World Outliner
2. Tìm actors bắt đầu bằng `Landing_`
3. Rename nếu cần
4. Save UE scene
5. Restart AirSim

---

### **Problem: "A* planning failed"**
```
❌ A* planning failed! No path found.
```

**Reasons:**
1. Start hoặc goal position nằm trong obstacle
2. Không có path khả thi (bị block hoàn toàn)
3. Map bounds không bao quát được start/goal

**Solution:**
1. Check actor positions trong UE (phải nằm trong map bounds)
2. Re-generate map với larger `scan_radius`:
   ```python
   # In airsim_navigation.py, change:
   grid = generator.generate_map(scan_radius=100)  # Tăng từ 80 lên 100
   ```
3. Check occupancy grid có đúng không:
   ```python
   import numpy as np
   grid = np.load('data/maps/building_5floors_grid.npy')
   print(f"Occupied cells: {np.sum(grid == 1)}")
   ```

---

### **Problem: Drone oscillates/crashes**

**Reasons:** PID gains không phù hợp

**Solution:** Tune PID trong config (xem phần Customization ở trên)

**Quick fixes:**
- **Oscillates:** Giảm `position_kp` xuống 1.0, tăng `position_kd` lên 1.0
- **Too slow:** Tăng `position_kp` lên 3.0
- **Crashes:** Giảm `max_velocity` xuống 3.0

---

### **Problem: No visualization lines in UE**

**Reasons:**
1. AirSim plotting API không được support
2. Visualization bị disable

**Solution:**
Script vẫn chạy bình thường, chỉ không có visual feedback. Metrics vẫn được log.

Để debug:
```python
# In visualize_astar_path.py
try:
    self.client.simPlotLineStrip(...)
except Exception as e:
    print(f"Visualization error: {e}")  # Xem error message
```

---

## 🎯 Expected Results

**Typical results cho well-configured system:**

| Metric | Expected Value |
|--------|----------------|
| Success Rate | > 90% |
| Planning Time | 0.1 - 0.5s |
| Execution Time | 20 - 60s (depends on distance) |
| ATE Error | < 0.5m |
| Energy | 2 - 10 kJ (depends on path length) |

**⚠️ Nếu results khác nhiều:**
- Success rate < 70% → Check actor positions, re-tune PID
- Planning time > 1s → Map resolution quá cao, giảm xuống
- ATE > 1m → PID tracking kém, cần tune
- Frequent collisions → Map không chính xác, re-scan environment

---

## 📝 Files Structure

```
src/baselines/astar_baseline/
├── astar_controller.py       # A* algorithm
├── pid_controller.py          # PID control
├── visualize_astar_path.py   # ← Script chính (mới)
├── run_airsim_evaluation.py  # Full evaluation (random scenarios)
└── README_VISUALIZATION.md   # ← Guide này

src/environment/
└── airsim_navigation.py       # Map generation

data/maps/
├── building_5floors_metadata.json  # Map info
├── building_5floors_grid.npy       # Occupancy grid
└── building_5floors_obstacles.npy  # Obstacle list
```

---

## 💡 Next Steps

Sau khi visualization chạy thành công:

1. **Collect baseline data:** Run full evaluation
   ```bash
   python src/baselines/astar_baseline/run_airsim_evaluation.py
   ```

2. **Compare với RL agent:** Train RL và compare với A* baseline

3. **Tune environment:** Thêm dynamic obstacles, human agents, etc.

4. **Custom scenarios:** Tạo test cases cụ thể trong UE

---

## 🤝 Support

**Issues thường gặp đã được giải quyết trong Troubleshooting section.**

Nếu vẫn gặp vấn đề:
1. Check console output (có detailed error messages)
2. Verify UE scene setup (actors + naming)
3. Re-generate map
4. Check AirSim connection

---

## 📚 References

- A* algorithm: Section 4.2 of project report
- PID control: Classic 3-term controller
- AirSim API: https://github.com/Microsoft/AirSim/blob/main/docs/apis.md
- Visualization API: `simPlotLineStrip`, `simPlotPoints`

---

**✅ Happy Flying!** 🚁
