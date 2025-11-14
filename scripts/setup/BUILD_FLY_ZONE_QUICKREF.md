bash
python scripts/setup/build_fly_zone.py

python scripts/setup/build_fly_zone.py --use-airsim

python scripts/setup/build_fly_zone.py --cell-size 0.25

python scripts/setup/build_fly_zone.py --origin 0 0 20 --size 30 30 40

python scripts/setup/build_fly_zone.py --no-viz

python scripts/setup/build_fly_zone.py --no-checkpoint

python
from scripts.setup.build_fly_zone import FlyZoneBuilder, GridConfig
import numpy as np

config = GridConfig(
    origin=(0.0, 0.0, 15.0),
    size=(20.0, 20.0, 30.0),
    cell_size=0.5,
    drone_radius=0.3,
    use_airsim_voxel=False,
    checkpoint_enabled=True
)

builder = FlyZoneBuilder(config)
grid = builder.build_grid()

builder.save_grid()

builder.visualize_slice(z_index=30, save_path="floor3.png")

python
import numpy as np

grid = np.load("data/maps/occupancy_grid.npy")
print(f"Shape: {grid.shape}")
print(f"Occupied: {np.sum(grid)}")

i, j, k = 20, 20, 30
if grid[i, j, k] == 0:
    print("Cell is free")
else:
    print("Cell is occupied")

free_indices = np.where(grid == 0)
free_count = len(free_indices[0])

Origin:      (0, 0, 15)m
Size:        20  20  30m
Cell size:   0.5m
Dimensions:  40  40  60 cells
Total cells: 96,000
Expected:    80 free, 20 occupied

 Issue  Solution
-----------------
 "cell_size too large"  Use smaller cell_size or larger size
 "AirSim not connected"  Script will auto-fallback to synthetic
 Out of memory  Increase cell_size or reduce grid size
 Slow build  Disable checkpoint, reduce resolution

data/maps/
   occupancy_grid.npy
   occupancy_grid_metadata.json

results/visualizations/
   grid_slice_z0.png
   grid_slice_z15.png
   grid_slice_z30.png
   grid_slice_z45.png

temp/grid_checkpoints/
   grid_partial.npy
   layer.txt

 Resolution  Time  Memory  Use Case
------------------------------------
 1.0m  2s  24KB  Fast prototyping
 0.5m  10s  96KB  Default (balanced)
 0.25m  90s  768KB  High accuracy
 0.125m  15min  6MB  Research/fine-tuning
