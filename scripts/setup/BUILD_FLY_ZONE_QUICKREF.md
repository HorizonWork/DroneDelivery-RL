# Build Fly Zone - Quick Reference

## Command Line Examples

```bash
# Default (20x20x30m @ 0.5m cells, synthetic obstacles)
python scripts/setup/build_fly_zone.py

# With AirSim (requires UE running)
python scripts/setup/build_fly_zone.py --use-airsim

# High resolution (0.25m cells)
python scripts/setup/build_fly_zone.py --cell-size 0.25

# Custom environment (30x30x40m)
python scripts/setup/build_fly_zone.py --origin 0 0 20 --size 30 30 40

# No visualization (faster)
python scripts/setup/build_fly_zone.py --no-viz

# Disable checkpoint (for small grids)
python scripts/setup/build_fly_zone.py --no-checkpoint
```

## Python API

```python
from scripts.setup.build_fly_zone import FlyZoneBuilder, GridConfig
import numpy as np

# Create config
config = GridConfig(
    origin=(0.0, 0.0, 15.0),
    size=(20.0, 20.0, 30.0),
    cell_size=0.5,
    drone_radius=0.3,
    use_airsim_voxel=False,
    checkpoint_enabled=True
)

# Build grid
builder = FlyZoneBuilder(config)
grid = builder.build_grid()

# Save
builder.save_grid()

# Visualize specific slice
builder.visualize_slice(z_index=30, save_path="floor3.png")
```

## Load and Use Grid

```python
import numpy as np

# Load grid
grid = np.load("data/maps/occupancy_grid.npy")
print(f"Shape: {grid.shape}")
print(f"Occupied: {np.sum(grid)}")

# Check specific cell
i, j, k = 20, 20, 30
if grid[i, j, k] == 0:
    print("Cell is free")
else:
    print("Cell is occupied")

# Get free cells
free_indices = np.where(grid == 0)
free_count = len(free_indices[0])
```

## Grid Specs (Default Config)

```
Origin:      (0, 0, 15)m
Size:        20 × 20 × 30m
Cell size:   0.5m
Dimensions:  40 × 40 × 60 cells
Total cells: 96,000
Expected:    ~80% free, ~20% occupied
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "cell_size too large" | Use smaller cell_size or larger size |
| "AirSim not connected" | Script will auto-fallback to synthetic |
| Out of memory | Increase cell_size or reduce grid size |
| Slow build | Disable checkpoint, reduce resolution |

## File Locations

```
data/maps/
  ├── occupancy_grid.npy          # Main grid data
  └── occupancy_grid_metadata.json # Config and stats

results/visualizations/
  ├── grid_slice_z0.png           # Ground level
  ├── grid_slice_z15.png          # Mid level
  ├── grid_slice_z30.png          # Upper level
  └── grid_slice_z45.png          # Top level

temp/grid_checkpoints/             # Resume data
  ├── grid_partial.npy
  └── layer.txt
```

## Performance Reference

| Resolution | Time | Memory | Use Case |
|------------|------|--------|----------|
| 1.0m | 2s | 24KB | Fast prototyping |
| 0.5m | 10s | 96KB | **Default (balanced)** |
| 0.25m | 90s | 768KB | High accuracy |
| 0.125m | 15min | 6MB | Research/fine-tuning |
