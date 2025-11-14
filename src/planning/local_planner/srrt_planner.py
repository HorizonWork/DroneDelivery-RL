import numpy as np
import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from src.planning.local_planner.cost_functions import SRRTCostFunction
from src.planning.local_planner.dynamic_obstacles import DynamicObstacleTracker
from src.planning.local_planner.safety_checker import SafetyChecker

dataclass
class SRRTNode:

    position: np.ndarray
    parent: Optional["SRRTNode"] = None
    children: List["SRRTNode"] = field(default_factory=list)
    cost_from_start: float = float("inf")
    clearance: float = float("inf")

    def __lt__(self, other):
        return self.cost_from_start  other.cost_from_start

dataclass
class SRRTResult:

    path: List[Tuple[float, float, float]]
    total_cost: float
    planning_time: float
    nodes_generated: int
    success: bool
    safety_clearance: float

class SRRTPlanner:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.max_iterations = config.get("max_iterations", 5000)
        self.max_planning_time = config.get("max_planning_time", 1.0)
        self.step_size = config.get("step_size", 0.5)
        self.goal_bias = config.get("goal_bias", 0.1)
        self.rewiring_radius = config.get("rewiring_radius", 2.0)

        self.lambda_c = config.get("lambda_c", 1.0)
        self.lambda_kappa = config.get("lambda_kappa", 2.0)

        self.min_clearance = config.get("min_clearance", 0.5)
        self.safety_margin = config.get("safety_margin", 0.3)
        self.prediction_horizon = config.get("prediction_horizon", 2.0)

        self.search_radius = config.get(
            "search_radius", 10.0
        )

        self.cost_function = SRRTCostFunction(config.get("cost_function", {}))
        self.obstacle_tracker = DynamicObstacleTracker(
            config.get("obstacle_tracker", {})
        )
        self.safety_checker = SafetyChecker(config.get("safety_checker", {}))

        self.tree_nodes: List[SRRTNode] = []
        self.node_index: Dict[Tuple[float, float, float], int] = {}

        self.planning_stats = {
            "total_plans": 0,
            "successful_plans": 0,
            "average_planning_time": 0.0,
            "average_nodes_generated": 0.0,
            "average_cost": 0.0,
        }

        self.logger.info("S-RRT Local Planner initialized")
        self.logger.info(
            f"Cost function: ℓ + {self.lambda_c}(1/dmin)² + {self.lambda_kappa}κ²"
        )
        self.logger.info(
            f"Max iterations: {self.max_iterations}, Step size: {self.step_size}m"
        )
        self.logger.info(f"Safety clearance: {self.min_clearance}m")

    def plan_safe_path(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        obstacles: List[Tuple[float, float, float]],
        global_path_hint: Optional[List[Tuple[float, float, float]]] = None,
    ) - Dict[str, Any]:

        planning_start = time.time()

        self.logger.info(f"S-RRT planning: {start}  {goal}")
        self.logger.info(f"Dynamic obstacles: {len(obstacles)}")

        self.obstacle_tracker.update_obstacles(obstacles)

        self._initialize_tree(start)

        search_region = self._define_search_region(start, goal)

        nodes_generated = 0
        best_goal_node = None

        for iteration in range(self.max_iterations):
            if time.time() - planning_start  self.max_planning_time:
                self.logger.warning("S-RRT planning timed out")
                break

            if random.random()  self.goal_bias:
                sample_point = np.array(goal)
            else:
                sample_point = self._sample_in_region(search_region)

            nearest_node = self._find_nearest_node(sample_point)

            new_position = self._steer(nearest_node.position, sample_point)

            if not self._is_safe_motion(nearest_node.position, new_position, obstacles):
                continue

            new_node = SRRTNode(position=new_position)
            new_node.cost_from_start = self._calculate_cost(
                nearest_node, new_node, obstacles
            )
            new_node.parent = nearest_node
            nearest_node.children.append(new_node)

            self.tree_nodes.append(new_node)
            nodes_generated += 1

            self._rewire_tree(new_node, obstacles)

            goal_distance = np.linalg.norm(new_node.position - np.array(goal))
            if goal_distance = self.step_size:
                if self._is_safe_motion(new_node.position, np.array(goal), obstacles):
                    goal_node = SRRTNode(position=np.array(goal))
                    goal_node.cost_from_start = self._calculate_cost(
                        new_node, goal_node, obstacles
                    )
                    goal_node.parent = new_node

                    if (
                        best_goal_node is None
                        or goal_node.cost_from_start  best_goal_node.cost_from_start
                    ):
                        best_goal_node = goal_node

        planning_time = time.time() - planning_start

        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            total_cost = best_goal_node.cost_from_start
            success = True

            safety_clearance = self._calculate_path_clearance(path, obstacles)

            self.logger.info(
                f"S-RRT successful: {len(path)} waypoints, cost: {total_cost:.2f}"
            )
        else:
            path = []
            total_cost = float("inf")
            success = False
            safety_clearance = 0.0

            self.logger.warning("S-RRT failed to find safe path")

        self._update_statistics(planning_time, nodes_generated, total_cost, success)

        result = SRRTResult(
            path=(
                [(float(p[0]), float(p[1]), float(p[2])) for p in path]
                if success
                else []
            ),
            total_cost=total_cost,
            planning_time=planning_time,
            nodes_generated=nodes_generated,
            success=success,
            safety_clearance=safety_clearance,
        )

        return {
            "success": result.success,
            "path": result.path,
            "cost": result.total_cost,
            "planning_time": result.planning_time,
            "nodes_generated": result.nodes_generated,
            "safety_clearance": result.safety_clearance,
        }

    def _initialize_tree(self, start: Tuple[float, float, float]):

        self.tree_nodes.clear()
        self.node_index.clear()

        start_node = SRRTNode(position=np.array(start))
        start_node.cost_from_start = 0.0

        self.tree_nodes.append(start_node)
        self.node_index[start] = 0

    def _define_search_region(
        self, start: Tuple[float, float, float], goal: Tuple[float, float, float]
    ) - Dict[str, float]:

        start_array = np.array(start)
        goal_array = np.array(goal)

        region_center = (start_array + goal_array) / 2
        region_size = np.linalg.norm(goal_array - start_array) + 2  self.search_radius

        return {
            "center": region_center,
            "radius": region_size / 2,
            "bounds": {
                "x_min": region_center[0] - region_size / 2,
                "x_max": region_center[0] + region_size / 2,
                "y_min": region_center[1] - region_size / 2,
                "y_max": region_center[1] + region_size / 2,
                "z_min": region_center[2] - region_size / 2,
                "z_max": region_center[2] + region_size / 2,
            },
        }

    def _sample_in_region(self, search_region: Dict[str, float]) - np.ndarray:

        bounds = search_region["bounds"]

        sample = np.array(
            [
                random.uniform(bounds["x_min"], bounds["x_max"]),
                random.uniform(bounds["y_min"], bounds["y_max"]),
                random.uniform(bounds["z_min"], bounds["z_max"]),
            ]
        )

        return sample

    def _find_nearest_node(self, sample_point: np.ndarray) - SRRTNode:

        min_distance = float("inf")
        nearest_node = self.tree_nodes[0]

        for node in self.tree_nodes:
            distance = np.linalg.norm(node.position - sample_point)
            if distance  min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) - np.ndarray:

        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance = self.step_size:
            return to_pos
        else:
            unit_direction = direction / distance
            return from_pos + unit_direction  self.step_size

    def _is_safe_motion(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        obstacles: List[Tuple[float, float, float]],
    ) - bool:

        return self.safety_checker.check_path_safety(
            from_pos, to_pos, obstacles, self.prediction_horizon
        )

    def _calculate_cost(
        self,
        parent_node: SRRTNode,
        child_node: SRRTNode,
        obstacles: List[Tuple[float, float, float]],
    ) - float:

        return (
            self.cost_function.calculate_edge_cost(parent_node, child_node, obstacles)
            + parent_node.cost_from_start
        )

    def _rewire_tree(
        self, new_node: SRRTNode, obstacles: List[Tuple[float, float, float]]
    ):

        nearby_nodes = []

        for node in self.tree_nodes[:-1]:
            distance = np.linalg.norm(node.position - new_node.position)
            if distance = self.rewiring_radius:
                nearby_nodes.append(node)

        for nearby_node in nearby_nodes:
            if self._is_safe_motion(new_node.position, nearby_node.position, obstacles):
                new_cost = self._calculate_cost(new_node, nearby_node, obstacles)

                if new_cost  nearby_node.cost_from_start:
                    if nearby_node.parent:
                        nearby_node.parent.children.remove(nearby_node)

                    nearby_node.parent = new_node
                    nearby_node.cost_from_start = new_cost
                    new_node.children.append(nearby_node)

                    self._propagate_cost_updates(nearby_node, obstacles)

    def _propagate_cost_updates(
        self, node: SRRTNode, obstacles: List[Tuple[float, float, float]]
    ):

        for child in node.children:
            old_cost = child.cost_from_start
            new_cost = self._calculate_cost(node, child, obstacles)

            if new_cost  old_cost:
                child.cost_from_start = new_cost
                self._propagate_cost_updates(child, obstacles)

    def _extract_path(self, goal_node: SRRTNode) - List[np.ndarray]:

        path = []
        current = goal_node

        while current is not None:
            path.append(current.position.copy())
            current = current.parent

        path.reverse()
        return path

    def _calculate_path_clearance(
        self, path: List[np.ndarray], obstacles: List[Tuple[float, float, float]]
    ) - float:

        if not path or not obstacles:
            return float("inf")

        min_clearance = float("inf")
        obstacle_positions = [np.array(obs) for obs in obstacles]

        for waypoint in path:
            for obstacle_pos in obstacle_positions:
                distance = np.linalg.norm(waypoint - obstacle_pos)
                min_clearance = min(min_clearance, distance)

        return min_clearance

    def _update_statistics(
        self,
        planning_time: float,
        nodes_generated: int,
        total_cost: float,
        success: bool,
    ):

        self.planning_stats["total_plans"] += 1

        if success:
            self.planning_stats["successful_plans"] += 1

            n = self.planning_stats["successful_plans"]

            self.planning_stats["average_planning_time"] = (
                (n - 1)  self.planning_stats["average_planning_time"] + planning_time
            ) / n

            self.planning_stats["average_nodes_generated"] = (
                (n - 1)  self.planning_stats["average_nodes_generated"]
                + nodes_generated
            ) / n

            self.planning_stats["average_cost"] = (
                (n - 1)  self.planning_stats["average_cost"] + total_cost
            ) / n

    def get_statistics(self) - Dict[str, Any]:

        stats = self.planning_stats.copy()

        if stats["total_plans"]  0:
            stats["success_rate"] = stats["successful_plans"] / stats["total_plans"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset(self):

        self.tree_nodes.clear()
        self.node_index.clear()
        self.obstacle_tracker.reset()

        self.logger.debug("S-RRT planner reset")

    def visualize_tree(self) - Dict[str, Any]:

        if not self.tree_nodes:
            return {"nodes": [], "edges": []}

        nodes = [node.position.tolist() for node in self.tree_nodes]
        edges = []

        for node in self.tree_nodes:
            if node.parent is not None:
                edges.append([node.parent.position.tolist(), node.position.tolist()])

        return {
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "tree_depth": self._calculate_tree_depth(),
        }

    def _calculate_tree_depth(self) - int:

        if not self.tree_nodes:
            return 0

        max_depth = 0

        for node in self.tree_nodes:
            depth = 0
            current = node

            while current.parent is not None:
                depth += 1
                current = current.parent

            max_depth = max(max_depth, depth)

        return max_depth
