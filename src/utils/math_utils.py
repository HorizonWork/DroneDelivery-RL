import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.interpolate import interp1d, UnivariateSpline
import math

class MathUtils:

    staticmethod
    def normalize_angle(angle: float) - float:

        while angle  np.pi:
            angle -= 2  np.pi
        while angle  -np.pi:
            angle += 2  np.pi
        return angle

    staticmethod
    def wrap_to_2pi(angle: float) - float:

        return angle  (2  np.pi)

    staticmethod
    def degrees_to_radians(degrees: float) - float:

        return degrees  np.pi / 180.0

    staticmethod
    def radians_to_degrees(radians: float) - float:

        return radians  180.0 / np.pi

    staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) - np.ndarray:

        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return np.array(
            [
                w1  x2 + x1  w2 + y1  z2 - z1  y2,
                w1  y2 - x1  z2 + y1  w2 + z1  x2,
                w1  z2 + x1  y2 - y1  x2 + z1  w2,
                w1  w2 - x1  x2 - y1  y2 - z1  z2,
            ]
        )

    staticmethod
    def quaternion_conjugate(q: np.ndarray) - np.ndarray:

        return np.array([-q[0], -q[1], -q[2], q[3]])

    staticmethod
    def quaternion_to_euler(q: np.ndarray) - Tuple[float, float, float]:

        rotation = Rotation.from_quat(q)
        return rotation.as_euler("xyz", degrees=False)

    staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) - np.ndarray:

        rotation = Rotation.from_euler("xyz", [roll, pitch, yaw])
        return rotation.as_quat()

    staticmethod
    def skew_symmetric(v: np.ndarray) - np.ndarray:

        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    staticmethod
    def rotation_matrix_from_vectors(v1: np.ndarray, v2: np.ndarray) - np.ndarray:

        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        cross = np.cross(v1_norm, v2_norm)
        dot = np.dot(v1_norm, v2_norm)

        if np.allclose(cross, 0):
            if dot  0:
                return np.eye(3)
            else:
                if abs(v1_norm[0])  0.9:
                    axis = np.array([1, 0, 0])
                else:
                    axis = np.array([0, 1, 0])
                axis = np.cross(v1_norm, axis)
                axis = axis / np.linalg.norm(axis)
                return 2  np.outer(axis, axis) - np.eye(3)

        cross_norm = np.linalg.norm(cross)
        axis = cross / cross_norm
        angle = np.arcsin(cross_norm)

        K = MathUtils.skew_symmetric(axis)
        R = np.eye(3) + np.sin(angle)  K + (1 - np.cos(angle))  (K  K)

        return R

    staticmethod
    def smooth_step(x: float, edge0: float, edge1: float) - float:

        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t  t  (3.0 - 2.0  t)

    staticmethod
    def lerp(a: float, b: float, t: float) - float:

        return a + t  (b - a)

    staticmethod
    def clamp(value: float, min_val: float, max_val: float) - float:

        return max(min_val, min(max_val, value))

class GeometryUtils:

    staticmethod
    def point_to_line_distance(
        point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray
    ) - float:

        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len_sq = np.dot(line_vec, line_vec)

        if line_len_sq == 0:
            return np.linalg.norm(point_vec)

        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t  line_vec

        return np.linalg.norm(point - projection)

    staticmethod
    def point_in_polygon(point: np.ndarray, polygon: np.ndarray) - bool:

        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i  n]
            if y  min(p1y, p2y):
                if y = max(p1y, p2y):
                    if x = max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y)  (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x = xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    staticmethod
    def sphere_box_collision(
        sphere_center: np.ndarray,
        sphere_radius: float,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) - bool:

        closest_point = np.clip(sphere_center, box_min, box_max)

        distance = np.linalg.norm(sphere_center - closest_point)

        return distance = sphere_radius

    staticmethod
    def line_sphere_intersection(
        line_start: np.ndarray,
        line_end: np.ndarray,
        sphere_center: np.ndarray,
        sphere_radius: float,
    ) - Tuple[bool, float]:

        line_vec = line_end - line_start
        to_sphere = line_start - sphere_center

        a = np.dot(line_vec, line_vec)
        b = 2.0  np.dot(to_sphere, line_vec)
        c = np.dot(to_sphere, to_sphere) - sphere_radius  sphere_radius

        discriminant = b  b - 4  a  c

        if discriminant  0:
            return False, 0.0

        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2  a)
        t2 = (-b + sqrt_discriminant) / (2  a)

        if 0 = t1 = 1:
            return True, t1
        elif 0 = t2 = 1:
            return True, t2

        return False, 0.0

    staticmethod
    def calculate_polygon_area(vertices: np.ndarray) - float:

        if len(vertices)  3:
            return 0.0

        area = 0.0
        n = len(vertices)

        for i in range(n):
            j = (i + 1)  n
            area += vertices[i][0]  vertices[j][1]
            area -= vertices[j][0]  vertices[i][1]

        return abs(area) / 2.0

    staticmethod
    def convex_hull_2d(points: np.ndarray) - np.ndarray:

        def cross_product(o, a, b):
            return (a[0] - o[0])  (b[1] - o[1]) - (a[1] - o[1])  (b[0] - o[0])

        points = sorted(set(map(tuple, points)))

        if len(points) = 1:
            return np.array(points)

        lower = []
        for p in points:
            while len(lower) = 2 and cross_product(lower[-2], lower[-1], p) = 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) = 2 and cross_product(upper[-2], upper[-1], p) = 0:
                upper.pop()
            upper.append(p)

        return np.array(lower[:-1] + upper[:-1])

def normalize_angle(angle: float) - float:

    return MathUtils.normalize_angle(angle)

class TrajectoryUtils:

    staticmethod
    def smooth_trajectory(
        waypoints: np.ndarray, smoothing_factor: float = 0.1
    ) - np.ndarray:

        if len(waypoints)  3:
            return waypoints

        distances = np.zeros(len(waypoints))
        for i in range(1, len(waypoints)):
            distances[i] = distances[i - 1] + np.linalg.norm(
                waypoints[i] - waypoints[i - 1]
            )

        smoothed_points = []

        for dim in range(waypoints.shape[1]):
            spline = UnivariateSpline(distances, waypoints[:, dim], s=smoothing_factor)

            smoothed_dim = spline(distances)
            smoothed_points.append(smoothed_dim)

        return np.column_stack(smoothed_points)

    staticmethod
    def calculate_trajectory_curvature(trajectory: np.ndarray) - np.ndarray:

        if len(trajectory)  3:
            return np.zeros(len(trajectory))

        curvatures = np.zeros(len(trajectory))

        for i in range(1, len(trajectory) - 1):
            p1, p2, p3 = trajectory[i - 1], trajectory[i], trajectory[i + 1]

            v1 = p2 - p1
            v2 = p3 - p2

            cross = np.cross(v1, v2)
            cross_mag = np.linalg.norm(cross)
            v1_mag = np.linalg.norm(v1)

            if v1_mag  1e-6:
                curvatures[i] = cross_mag / (v1_mag3)

        return curvatures

    staticmethod
    def resample_trajectory(trajectory: np.ndarray, target_points: int) - np.ndarray:

        if len(trajectory) = 2:
            return trajectory

        distances = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            distances[i] = distances[i - 1] + np.linalg.norm(
                trajectory[i] - trajectory[i - 1]
            )

        total_distance = distances[-1]
        target_distances = np.linspace(0, total_distance, target_points)

        resampled_points = []
        for dim in range(trajectory.shape[1]):
            interp_func = interp1d(distances, trajectory[:, dim], kind="linear")
            resampled_dim = interp_func(target_distances)
            resampled_points.append(resampled_dim)

        return np.column_stack(resampled_points)
