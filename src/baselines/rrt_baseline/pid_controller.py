"""
PID Controller for RRT* baseline (same as A* baseline).
"""

# Import the same PID controller from A* baseline to avoid duplication
from ..astar_baseline.pid_controller import PIDController

# Re-export for this module
__all__ = ['PIDController']