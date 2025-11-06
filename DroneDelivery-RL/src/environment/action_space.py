"""
4D continuous action space [vx,vy,vz,ω]
""",
import numpy as np

class ActionSpace:
    def __init__(self):
        self.size = 4  # [vx, vy, vz, ω]
        self.action = np.zeros(self.size)
        
    def sample(self):
        # Return random action in valid range
        return np.random.uniform(-1, 1, self.size)
