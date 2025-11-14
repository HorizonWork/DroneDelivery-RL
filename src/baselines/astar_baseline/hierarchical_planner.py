import numpy as np
from typing import List, Tuple
import time

class HierarchicalPlanner:

    def __init__(self, astar_controller):

        self.astar = astar_controller

        self.hub_xy = np.array([-60.0, 30.0])

        self.floor_heights_positive = [3, 9, 15, 21, 27]
        self.floor_z_ned = [-h for h in self.floor_heights_positive]

    def plan_path(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float]
    ) - List[Tuple[float, float, float]]:

        plan_start = time.time()

        horizontal_dist = np.linalg.norm(np.array(start_pos[:2]) - np.array(goal_pos[:2]))
        vertical_dist = abs(start_pos[2] - goal_pos[2])

        use_hierarchical = (vertical_dist  3.0) or (horizontal_dist  30.0)

        if use_hierarchical:
            print(f"    HIERARCHICAL planning (hub routing)")
            path = self._plan_hierarchical(start_pos, goal_pos, plan_start)
        else:
            print(f"    DIRECT A (same floor)")
            path = self._plan_direct(start_pos, goal_pos, plan_start)

        return path if path else []

    def _plan_hierarchical(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        plan_start: float
    ) - List[Tuple[float, float, float]]:

        start_floor_z = self._nearest_floor(start_pos[2])
        goal_floor_z = self._nearest_floor(goal_pos[2])

        hub_start = (self.hub_xy[0], self.hub_xy[1], start_floor_z)
        hub_goal = (self.hub_xy[0], self.hub_xy[1], goal_floor_z)

        print(f"      Phase 1: Start  Hubfloor{self._floor_name(start_floor_z)}")
        print(f"      Phase 2: Vertical {self._floor_name(start_floor_z)}  {self._floor_name(goal_floor_z)}")
        print(f"      Phase 3: Hubfloor{self._floor_name(goal_floor_z)}  Goal")

        path1 = self.astar.plan_path(start_pos, hub_start)
        if not path1:
            print(f"        Phase 1 failed  fallback to direct")
            return self._plan_direct(start_pos, goal_pos, plan_start)

        path2 = []
        if abs(start_floor_z - goal_floor_z)  1.0:
            num_steps = max(3, int(abs(goal_floor_z - start_floor_z) / 2.0))
            z_waypoints = np.linspace(start_floor_z, goal_floor_z, num_steps)
            path2 = [(self.hub_xy[0], self.hub_xy[1], z) for z in z_waypoints]

        path3 = self.astar.plan_path(hub_goal, goal_pos)
        if not path3:
            print(f"        Phase 3 failed  fallback to direct")
            return self._plan_direct(start_pos, goal_pos, plan_start)

        combined = path1.copy()
        if path2:
            combined.extend(path2[1:])
        combined.extend(path3[1:])

        elapsed = time.time() - plan_start
        print(f"    Hierarchical: {len(path1)}+{len(path2)}+{len(path3)} = {len(combined)} waypoints in {elapsed:.1f}s")

        return combined

    def _plan_direct(
        self,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        plan_start: float
    ) - List[Tuple[float, float, float]]:

        path = self.astar.plan_path(start_pos, goal_pos)
        elapsed = time.time() - plan_start
        if path:
            print(f"    Direct A: {len(path)} waypoints in {elapsed:.1f}s")
        return path

    def _nearest_floor(self, z_pos: float) - float:

        distances = [abs(z_pos - z) for z in self.floor_z_ned]
        nearest_idx = np.argmin(distances)
        return self.floor_z_ned[nearest_idx]

    def _floor_name(self, z_ned: float) - str:

        try:
            idx = self.floor_z_ned.index(z_ned)
            return f"{idx+1} ({self.floor_heights_positive[idx]}m)"
        except ValueError:
            return f"({z_ned:.0f}m)"
