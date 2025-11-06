"""
35D observation implementation matching Table 1
- Pose (7): 3D position + quaternion
- Velocity (4): Body-frame velocities + yaw rate
- Goal vector (3): 3D vector to target
- Battery (1): Remaining battery fraction
- Occupancy (24): 24-sector histogram
- Localization error (1): ATE estimate
""",
import numpy as np

class ObservationSpace:
    def __init__(self):
        self.dimensions = 35  # 7 + 4 + 3 + 1 + 24 + 1
        self.observation = np.zeros(self.dimensions)
        
    def get_observation(self):
        return self.observation
