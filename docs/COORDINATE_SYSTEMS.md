# 🧭 COORDINATE SYSTEMS & TRANSFORMATIONS

**Project:** DroneDelivery-RL  
**Last Updated:** November 13, 2025

---

## 🌍 WORLD REBASE ISSUE

### Unreal Engine World Translation
```
WORLD TRANSLATION: {0, 0, 0} → {6000, -3000, 150} cm
                               = {60, -30, 1.5} m
```

**Nguyên nhân:** PlayerStart trong Unreal bị rebase từ origin → coordinates bị shift.

**Ảnh hưởng:** Tất cả tọa độ AirSim API trả về đã **BỊ OFFSET** bởi `{60, -30, 1.5}` mét!

---

## 📐 HỆ TỌA ĐỘ TRONG DỰ ÁN

### 1. **AirSim/Unreal Frame (NED)**
```
X-axis: North (forward)
Y-axis: East (right)
Z-axis: Down (negative altitude)
Origin: {60, -30, 1.5} m from world origin
```

**Example:**
```python
pose = client.simGetVehiclePose()
# Returns NED with offset: (65.5, -28.3, -12.0) m
```

### 2. **Grid Frame (ENU)**
```
X-axis: East (horizontal)
Y-axis: North (horizontal)
Z-axis: Up (altitude)
Origin: (0, 0, 15) m - center of grid at 15m height
```

**Example:**
```python
grid_origin = (0.0, 0.0, 15.0)  # Grid center
grid[20, 20, 30] = 1  # Occupied cell
```

### 3. **Occupancy Grid Indices**
```
grid[i, j, k]:
  i: X-axis index (0 to width-1)
  j: Y-axis index (0 to height-1)
  k: Z-axis index (0 to depth-1)
```

---

## 🔄 COORDINATE TRANSFORMATIONS

### **AirSim NED → Grid ENU**

```python
def _ned_to_enu(x_ned, y_ned, z_ned):
    """
    Step 1: Remove world offset
    """
    x_ned_origin = x_ned - 60.0
    y_ned_origin = y_ned - (-30.0)
    z_ned_origin = z_ned - 1.5
    
    """
    Step 2: Rotate NED → ENU
    """
    x_enu = y_ned_origin   # East = NED Y
    y_enu = x_ned_origin   # North = NED X
    z_enu = -z_ned_origin  # Up = -Down
    
    return x_enu, y_enu, z_enu
```

**Example:**
```python
# AirSim returns (with offset):
airsim_pos = (65.5, -28.3, -12.0)

# Remove offset:
origin_pos = (5.5, 1.7, -13.5)

# Rotate NED → ENU:
grid_pos = (1.7, 5.5, 13.5)  # ENU
```

### **Grid ENU → AirSim NED**

```python
def _enu_to_ned(x_enu, y_enu, z_enu):
    """
    Step 1: Rotate ENU → NED
    """
    x_ned_origin = y_enu     # North = ENU Y
    y_ned_origin = x_enu     # East = ENU X
    z_ned_origin = -z_enu    # Down = -Up
    
    """
    Step 2: Add world offset
    """
    x_ned = x_ned_origin + 60.0
    y_ned = y_ned_origin + (-30.0)
    z_ned = z_ned_origin + 1.5
    
    return x_ned, y_ned, z_ned
```

**Example:**
```python
# Want to move drone to grid position:
target_grid = (5.0, 10.0, 20.0)  # ENU

# Rotate ENU → NED origin:
target_origin = (10.0, 5.0, -20.0)

# Add offset:
airsim_cmd = (70.0, -25.0, -18.5)  # Send to AirSim
```

### **Grid Indices → World ENU**

```python
def grid_to_world(i, j, k, origin, size, cell_size):
    """Convert grid index to world coordinates"""
    x = origin[0] - size[0]/2 + (i + 0.5) * cell_size
    y = origin[1] - size[1]/2 + (j + 0.5) * cell_size
    z = origin[2] + (k + 0.5) * cell_size
    return (x, y, z)
```

**Example:**
```python
# Grid parameters:
origin = (0, 0, 15)
size = (20, 20, 30)
cell_size = 0.5

# Grid index:
i, j, k = 20, 20, 30

# World ENU:
x = 0 - 10 + 10.25 = 0.25
y = 0 - 10 + 10.25 = 0.25
z = 15 + 15.25 = 30.25

# Result: (0.25, 0.25, 30.25) m in ENU
```

### **World ENU → Grid Indices**

```python
def world_to_grid(x, y, z, origin, size, cell_size):
    """Convert world coordinates to grid index"""
    i = int((x - (origin[0] - size[0]/2)) / cell_size)
    j = int((y - (origin[1] - size[1]/2)) / cell_size)
    k = int((z - origin[2]) / cell_size)
    return (i, j, k)
```

---

## 📊 TRANSFORMATION PIPELINE

### **Full Pipeline: AirSim → Grid → Planning → AirSim**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AirSim Returns Pose (NED with offset)                   │
│    pose = client.simGetVehiclePose()                       │
│    → (65.5, -28.3, -12.0) m                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Convert NED → ENU (remove offset + rotate)              │
│    _ned_to_enu(65.5, -28.3, -12.0)                         │
│    → (1.7, 5.5, 13.5) m                                    │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Convert ENU → Grid Indices                              │
│    _world_to_grid(1.7, 5.5, 13.5)                          │
│    → grid[23, 31, 27]                                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. A* Planning on Grid                                      │
│    path_grid = astar.plan(start_idx, goal_idx)             │
│    → [(23,31,27), (24,31,28), ..., (45,50,35)]            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Convert Grid → ENU Waypoints                            │
│    waypoints_enu = [grid_to_world(idx) for idx in path]    │
│    → [(1.7,5.5,13.5), (2.25,5.5,14.0), ..., (12.75,15,17.5)]│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Convert ENU → NED for AirSim Commands                   │
│    ned_cmds = [_enu_to_ned(wp) for wp in waypoints_enu]    │
│    → [(65.5,-28.3,-12.0), (65.5,-27.75,-12.5), ...]       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Send to AirSim                                           │
│    client.moveToPositionAsync(x_ned, y_ned, z_ned, 5)      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ USAGE EXAMPLES

### **Example 1: LiDAR Scan Position**

```python
# Want to scan at grid position (5, 5, 20) ENU
scan_enu = (5.0, 5.0, 20.0)

# Convert to AirSim NED
scan_ned = _enu_to_ned(5.0, 5.0, 20.0)
# → (65.0, -25.0, -18.5)

# Move drone
pose = airsim.Pose(
    airsim.Vector3r(scan_ned[0], scan_ned[1], scan_ned[2]),
    airsim.Quaternionr(0, 0, 0, 1)
)
client.simSetVehiclePose(pose, ignore_collision=True)
```

### **Example 2: Process LiDAR Points**

```python
# Get LiDAR data (returns NED relative to drone)
lidar_data = client.getLidarData()
points_ned_relative = np.array(lidar_data.point_cloud).reshape(-1, 3)

# Drone position in NED (with offset)
drone_ned = (65.5, -28.3, -12.0)

# Convert points to world NED
points_ned_world = points_ned_relative + drone_ned

# Convert to ENU
points_enu = [_ned_to_enu(p[0], p[1], p[2]) for p in points_ned_world]

# Convert to grid indices
grid_indices = [_world_to_grid(p[0], p[1], p[2]) for p in points_enu]

# Mark occupied in grid
for i, j, k in grid_indices:
    if 0 <= i < width and 0 <= j < height and 0 <= k < depth:
        grid[i, j, k] = 1
```

### **Example 3: Plan and Execute Path**

```python
# Current position from AirSim
pose = client.simGetVehiclePose()
curr_ned = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
curr_enu = _ned_to_enu(*curr_ned)
curr_grid = _world_to_grid(*curr_enu)

# Goal position (ENU)
goal_enu = (10.0, 15.0, 25.0)
goal_grid = _world_to_grid(*goal_enu)

# A* planning
path_grid = astar.plan(curr_grid, goal_grid)

# Convert to waypoints
waypoints_enu = [grid_to_world(*idx) for idx in path_grid]

# Execute
for wp_enu in waypoints_enu:
    wp_ned = _enu_to_ned(*wp_enu)
    client.moveToPositionAsync(wp_ned[0], wp_ned[1], wp_ned[2], 5).join()
```

---

## ⚠️ COMMON PITFALLS

### ❌ **Mistake 1: Forgetting World Offset**
```python
# WRONG: Direct rotation without offset removal
x_enu = y_ned  # Missing offset subtraction!
```

### ❌ **Mistake 2: Wrong Rotation Order**
```python
# WRONG: Rotating before removing offset
x_enu = y_ned - offset[1]  # Should remove offset FIRST!
```

### ❌ **Mistake 3: Mixing Units**
```python
# WRONG: AirSim uses meters, not cm!
client.moveToPositionAsync(600, -300, 15, 5)  # Should be (60, -30, 1.5)
```

### ❌ **Mistake 4: Grid Origin Confusion**
```python
# WRONG: Treating grid origin as world origin
grid[0, 0, 0] != world (0, 0, 0)
# grid[0, 0, 0] = world (-10, -10, 15) in ENU
```

---

## ✅ VALIDATION

### **Test Coordinate Conversions**

```python
def test_coordinate_roundtrip():
    """Test NED ↔ ENU conversion"""
    # Original NED (with offset)
    ned_original = (65.5, -28.3, -12.0)
    
    # NED → ENU → NED
    enu = _ned_to_enu(*ned_original)
    ned_back = _enu_to_ned(*enu)
    
    assert np.allclose(ned_original, ned_back, atol=1e-6)
    print(f"✓ Roundtrip test passed: {ned_original} → {enu} → {ned_back}")

def test_grid_conversion():
    """Test Grid ↔ World conversion"""
    # Grid index
    grid_idx = (20, 20, 30)
    
    # Grid → World → Grid
    world_enu = grid_to_world(*grid_idx)
    grid_back = world_to_grid(*world_enu)
    
    assert grid_idx == grid_back
    print(f"✓ Grid test passed: {grid_idx} → {world_enu} → {grid_back}")
```

### **Verify with AirSim**

```python
# Move drone to known position
target_enu = (5.0, 5.0, 20.0)
target_ned = _enu_to_ned(*target_enu)

client.moveToPositionAsync(*target_ned, 5).join()

# Read back position
pose = client.simGetVehiclePose()
actual_ned = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
actual_enu = _ned_to_enu(*actual_ned)

# Should match
print(f"Target ENU: {target_enu}")
print(f"Actual ENU: {actual_enu}")
assert np.allclose(target_enu, actual_enu, atol=0.1)  # 10cm tolerance
```

---

## 📝 CONFIGURATION

### **Update `map_generator.py` Arguments**

```powershell
# Default world offset (from log: {6000, -3000, 150} cm)
python src/environment/map_generator.py --use-airsim \
    --world-offset 60 -30 1.5

# Custom offset if world rebase changes
python src/environment/map_generator.py --use-airsim \
    --world-offset 70 -40 2.0
```

### **Metadata Verification**

Check `data/maps/occupancy_grid_metadata.json`:
```json
{
  "world_offset": [60.0, -30.0, 1.5],
  "coordinate_systems": {
    "grid": "ENU (X=East, Y=North, Z=Up)",
    "airsim": "NED (X=North, Y=East, Z=Down)",
    "world_offset_note": "Unreal world was rebased by {60, -30, 1.5}m from PlayerStart"
  }
}
```

---

## 🎯 SUMMARY

| Transformation | Formula | Example Input | Example Output |
|----------------|---------|---------------|----------------|
| **NED → ENU** | `x_enu = y_ned - offset_y`<br>`y_enu = x_ned - offset_x`<br>`z_enu = -(z_ned - offset_z)` | `(65.5, -28.3, -12.0)` | `(1.7, 5.5, 13.5)` |
| **ENU → NED** | `x_ned = y_enu + offset_x`<br>`y_ned = x_enu + offset_y`<br>`z_ned = -z_enu + offset_z` | `(1.7, 5.5, 13.5)` | `(65.5, -28.3, -12.0)` |
| **Grid → World** | `x = origin_x - size_x/2 + (i+0.5)*cell`<br>`y = origin_y - size_y/2 + (j+0.5)*cell`<br>`z = origin_z + (k+0.5)*cell` | `grid[20,20,30]` | `(0.25, 0.25, 30.25)` |
| **World → Grid** | `i = int((x - (origin_x - size_x/2)) / cell)`<br>`j = int((y - (origin_y - size_y/2)) / cell)`<br>`k = int((z - origin_z) / cell)` | `(0.25, 0.25, 30.25)` | `grid[20,20,30]` |

**World Offset:** `{60, -30, 1.5}` meters (from Unreal rebase)

---

**END OF DOCUMENTATION**
