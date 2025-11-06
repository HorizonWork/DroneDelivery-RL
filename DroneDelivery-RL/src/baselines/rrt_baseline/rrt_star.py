"""
RRT* (Rapidly-exploring Random Tree Star) Path Planner
Implements classical RRT* for baseline comparison.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import random
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

@dataclass
class RRTNode:
    """Node in the RRT* tree."""
    position: Tuple[float, float, float]
    parent: Optional['RRTNode'] = None
    cost: float = 0.0
    children: List['RRTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class RRTStarController:
    """
    RRT* path planner for 5-floor building navigation.
    Implements asymptotically optimal sampling-based planning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Environment configuration
        self.floors = config.get('floors', 5)
        self.floor_length = config.get('floor_length', 20.0)  # meters
        self.floor_width = config.get('floor_width', 40.0)    # meters
        self.floor_height = config.get('floor_height', 3.0)   # meters
        
        # RRT* parameters
        self.max_iterations = config.get('max_iterations', 5000)
        self.step_size = config.get('step_size', 1.0)         # meters
        self.goal_bias = config.get('goal_bias', 0.1)         # 10% goal bias
        self.rewire_radius = config.get('rewire_radius', 3.0) # meters
        self.goal_tolerance = config.get('goal_tolerance', 0.5) # meters
        
        # Obstacle checking
        self.collision_check_resolution = config.get('collision_resolution', 0.2)  # meters
        
        # Environment bounds
        self.bounds = {
            'x_min': 0.0, 'x_max': self.floor_length,
            'y_min': 0.0, 'y_max': self.floor_width,
            'z_min': 0.0, 'z_max': self.floors * self.floor_height
        }
        
        # Obstacles (updated dynamically)
        self.obstacles: List[Tuple[float, float, float, float]] = []  # (x, y, z, radius)
        
        # Tree storage
        self.nodes: List[RRTNode] = []
        self.goal_node: Optional[RRTNode] = None
        
        # Current path
        self.current_path: List[Tuple[float, float, float]] = []
        self.path_index = 0
        
        # Random seed for reproducibility
        random.seed(config.get('seed', 42))
        np.random.seed(config.get('seed', 42))
    
    def update_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """Update obstacle positions."""
        # Convert point obstacles to spheres with radius
        obstacle_radius = 0.5  # meters
        self.obstacles = [(x, y, z, obstacle_radius) for x, y, z in obstacles]
    
    def sample_random_point(self, goal_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Sample random point in configuration space with goal bias."""
        if random.random() < self.goal_bias:
            return goal_pos
        
        # Uniform random sampling in 3D space
        x = random.uniform(self.bounds['x_min'], self.bounds['x_max'])
        y = random.uniform(self.bounds['y_min'], self.bounds['y_max'])
        z = random.uniform(self.bounds['z_min'], self.bounds['z_max'])
        
        return (x, y, z)
    
    def find_nearest_node(self, point: Tuple[float, float, float]) -> RRTNode:
        """Find nearest node in tree to given point."""
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            distance = euclidean(node.position, point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def steer(self, from_pos: Tuple[float, float, float], 
              to_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Steer from one position towards another with step size limit."""
        direction = np.array(to_pos) - np.array(from_pos)
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_pos
        
        # Normalize and scale by step size
        unit_direction = direction / distance
        new_pos = np.array(from_pos) + unit_direction * self.step_size
        
        return tuple(new_pos)
    
    def is_collision_free(self, pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> bool:
        """Check if path between two positions is collision-free."""
        # Check bounds
        for pos in [pos1, pos2]:
            if not (self.bounds['x_min'] <= pos[0] <= self.bounds['x_max'] and
                   self.bounds['y_min'] <= pos[1] <= self.bounds['y_max'] and
                   self.bounds['z_min'] <= pos[2] <= self.bounds['z_max']):
                return False
        
        # Discretize path and check collisions
        distance = euclidean(pos1, pos2)
        num_checks = int(np.ceil(distance / self.collision_check_resolution))
        
        if num_checks <= 1:
            return self._is_point_collision_free(pos2)
        
        for i in range(num_checks + 1):
            t = i / num_checks
            intermediate_pos = (
                pos1[0] + t * (pos2[0] - pos1[0]),
                pos1[1] + t * (pos2[1] - pos1[1]),
                pos1[2] + t * (pos2[2] - pos1[2])
            )
            
            if not self._is_point_collision_free(intermediate_pos):
                return False
        
        return True
    
    def _is_point_collision_free(self, pos: Tuple[float, float, float]) -> bool:
        """Check if a single point is collision-free."""
        for obs_x, obs_y, obs_z, radius in self.obstacles:
            distance = euclidean(pos, (obs_x, obs_y, obs_z))
            if distance <= radius:
                return False
        return True
    
    def get_nearby_nodes(self, position: Tuple[float, float, float], radius: float) -> List[RRTNode]:
        """Get all nodes within radius of given position."""
        nearby_nodes = []
        for node in self.nodes:
            if euclidean(node.position, position) <= radius:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def compute_path_cost(self, node: RRTNode) -> float:
        """Compute total cost from root to node."""
        cost = 0.0
        current = node
        
        while current.parent is not None:
            cost += euclidean(current.position, current.parent.position)
            current = current.parent
        
        return cost
    
    def rewire_tree(self, new_node: RRTNode):
        """Rewire tree to maintain optimality (RRT* key feature)."""
        nearby_nodes = self.get_nearby_nodes(new_node.position, self.rewire_radius)
        
        for nearby_node in nearby_nodes:
            if nearby_node == new_node or nearby_node == new_node.parent:
                continue
            
            # Check if routing through new_node is better
            potential_cost = new_node.cost + euclidean(new_node.position, nearby_node.position)
            
            if (potential_cost < nearby_node.cost and 
                self.is_collision_free(new_node.position, nearby_node.position)):
                
                # Remove nearby_node from its current parent
                if nearby_node.parent:
                    nearby_node.parent.children.remove(nearby_node)
                
                # Set new parent
                nearby_node.parent = new_node
                new_node.children.append(nearby_node)
                
                # Update cost
                nearby_node.cost = potential_cost
                
                # Recursively update costs of descendants
                self._update_descendant_costs(nearby_node)
    
    def _update_descendant_costs(self, node: RRTNode):
        """Recursively update costs of all descendants."""
        for child in node.children:
            child.cost = node.cost + euclidean(node.position, child.position)
            self._update_descendant_costs(child)
    
    def plan_path(self, start_pos: Tuple[float, float, float], 
                  goal_pos: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        Plan path using RRT* algorithm.
        Returns path as list of world coordinates.
        """
        # Initialize tree with start node
        start_node = RRTNode(position=start_pos, cost=0.0)
        self.nodes = [start_node]
        self.goal_node = None
        
        # RRT* main loop
        for iteration in range(self.max_iterations):
            # Sample random point
            random_point = self.sample_random_point(goal_pos)
            
            # Find nearest node
            nearest_node = self.find_nearest_node(random_point)
            
            # Steer towards random point
            new_position = self.steer(nearest_node.position, random_point)
            
            # Check collision
            if not self.is_collision_free(nearest_node.position, new_position):
                continue
            
            # Create new node
            new_node = RRTNode(
                position=new_position,
                parent=nearest_node,
                cost=nearest_node.cost + euclidean(nearest_node.position, new_position)
            )
            
            # Find best parent among nearby nodes
            nearby_nodes = self.get_nearby_nodes(new_position, self.rewire_radius)
            best_parent = nearest_node
            best_cost = new_node.cost
            
            for nearby_node in nearby_nodes:
                potential_cost = nearby_node.cost + euclidean(nearby_node.position, new_position)
                
                if (potential_cost < best_cost and 
                    self.is_collision_free(nearby_node.position, new_position)):
                    best_parent = nearby_node
                    best_cost = potential_cost
            
            # Set best parent
            new_node.parent = best_parent
            new_node.cost = best_cost
            best_parent.children.append(new_node)
            
            # Add to tree
            self.nodes.append(new_node)
            
            # Rewire tree
            self.rewire_tree(new_node)
            
            # Check if goal reached
            if euclidean(new_position, goal_pos) <= self.goal_tolerance:
                self.goal_node = new_node
                break
        
        # Extract path
        if self.goal_node:
            return self._extract_path()
        else:
            # Return path to closest node to goal
            closest_node = min(self.nodes, key=lambda n: euclidean(n.position, goal_pos))
            self.goal_node = closest_node
            return self._extract_path()
    
    def _extract_path(self) -> List[Tuple[float, float, float]]:
        """Extract path from goal node to start."""
        if not self.goal_node:
            return []
        
        path = []
        current = self.goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def set_path(self, path: List[Tuple[float, float, float]]):
        """Set current path to follow."""
        self.current_path = path
        self.path_index = 0
    
    def get_next_waypoint(self, current_pos: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Get next waypoint in path based on current position."""
        if not self.current_path or self.path_index >= len(self.current_path):
            return None
        
        # Check if we're close to current waypoint
        current_waypoint = self.current_path[self.path_index]
        distance = euclidean(current_pos, current_waypoint)
        
        if distance < 0.5:  # Within 0.5m tolerance
            self.path_index += 1
            if self.path_index >= len(self.current_path):
                return None
        
        return self.current_path[self.path_index]
    
    def visualize_tree(self, start_pos: Tuple[float, float, float], 
                      goal_pos: Tuple[float, float, float]):
        """Visualize RRT* tree and path (for debugging)."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw tree edges
        for node in self.nodes:
            if node.parent:
                xs = [node.position[0], node.parent.position[0]]
                ys = [node.position[1], node.parent.position[1]]
                zs = [node.position[2], node.parent.position[2]]
                ax.plot(xs, ys, zs, 'b-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        positions = np.array([node.position for node in self.nodes])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='blue', s=10, alpha=0.6)
        
        # Draw start and goal
        ax.scatter(*start_pos, c='green', s=100, marker='o', label='Start')
        ax.scatter(*goal_pos, c='red', s=100, marker='*', label='Goal')
        
        # Draw path
        if self.current_path:
            path_array = np.array(self.current_path)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                   'r-', linewidth=3, label='Path')
        
        # Draw obstacles
        for obs_x, obs_y, obs_z, radius in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius * np.outer(np.cos(u), np.sin(v)) + obs_x
            y = radius * np.outer(np.sin(u), np.sin(v)) + obs_y
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obs_z
            ax.plot_surface(x, y, z, alpha=0.3, color='red')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('RRT* Tree and Path')
        
        plt.show()
