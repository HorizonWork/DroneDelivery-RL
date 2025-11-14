import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

dataclass
class PoseEstimate:

    position: np.ndarray
    orientation: np.ndarray
    confidence: float
    num_inliers: int
    reprojection_error: float
    timestamp: float

class PoseEstimator:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.K = np.array([
            [config.get('fx', 460.0), 0, config.get('cx', 320.0)],
            [0, config.get('fy', 460.0), config.get('cy', 240.0)],
            [0, 0, 1]
        ])

        self.baseline = config.get('baseline', 0.10)

        self.ransac_threshold = config.get('ransac_threshold', 1.0)
        self.ransac_confidence = config.get('ransac_confidence', 0.999)
        self.ransac_max_iters = config.get('ransac_max_iters', 1000)

        self.min_parallax = config.get('min_parallax', 1.0)
        self.max_reprojection_error = config.get('max_reprojection_error', 2.0)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.match_ratio_threshold = config.get('match_ratio_threshold', 0.8)

        self.logger.info("Pose Estimator initialized")
        self.logger.info(f"Camera matrix: fx={self.K[0,0]}, fy={self.K[1,1]}")
        self.logger.info(f"Baseline: {self.baseline}m")

    def estimate_pose_stereo(self, kp_left: List, desc_left: np.ndarray,
                           kp_right: List, desc_right: np.ndarray,
                           timestamp: float) - Optional[PoseEstimate]:

        try:
            matches = self.matcher.match(desc_left, desc_right)

            if len(matches)  8:
                return None

            pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches])
            pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches])

            disparities = pts_left[:, 0] - pts_right[:, 0]

            valid_disparity = disparities  0.5
            if np.sum(valid_disparity)  8:
                return None

            pts_left = pts_left[valid_disparity]
            pts_right = pts_right[valid_disparity]
            disparities = disparities[valid_disparity]

            points_3d = self._triangulate_stereo_points(pts_left, pts_right)

            pose = self._estimate_motion_pnp(pts_left, points_3d, timestamp)

            return pose

        except Exception as e:
            self.logger.error(f"Stereo pose estimation error: {e}")
            return None

    def _triangulate_stereo_points(self, pts_left: np.ndarray, pts_right: np.ndarray) - np.ndarray:

        points_3d = []

        for i in range(len(pts_left)):
            xl, yl = pts_left[i]
            xr, yr = pts_right[i]

            d = xl - xr

            if d  0.5:
                Z = self.K[0, 0]  self.baseline / d
                X = (xl - self.K[0, 2])  Z / self.K[0, 0]
                Y = (yl - self.K[1, 2])  Z / self.K[1, 1]

                points_3d.append([X, Y, Z])

        return np.array(points_3d) if points_3d else np.array([]).reshape(0, 3)

    def _estimate_motion_pnp(self, image_points: np.ndarray, world_points: np.ndarray,
                           timestamp: float) - Optional[PoseEstimate]:

        if len(image_points) != len(world_points) or len(image_points)  6:
            return None

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                world_points.astype(np.float32),
                image_points.astype(np.float32),
                self.K,
                None,
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_confidence,
                iterationsCount=self.ransac_max_iters
            )

            if not success or inliers is None or len(inliers)  6:
                return None

            R_mat, _ = cv2.Rodrigues(rvec)

            projected_points, _ = cv2.projectPoints(
                world_points[inliers.flatten()], rvec, tvec, self.K, None
            )
            reprojection_errors = np.linalg.norm(
                projected_points.reshape(-1, 2) - image_points[inliers.flatten()], axis=1
            )
            mean_reprojection_error = np.mean(reprojection_errors)

            confidence = len(inliers) / len(image_points)

            pose_estimate = PoseEstimate(
                position=tvec.flatten(),
                orientation=R_mat,
                confidence=confidence,
                num_inliers=len(inliers),
                reprojection_error=mean_reprojection_error,
                timestamp=timestamp
            )

            return pose_estimate

        except Exception as e:
            self.logger.error(f"PnP estimation error: {e}")
            return None

    def ned_to_enu(self, ned_pos: Tuple[float, float, float]) - Tuple[float, float, float]:

        n, e, d = ned_pos
        return (e, n, -d)

    def enu_to_ned(self, enu_pos: Tuple[float, float, float]) - Tuple[float, float, float]:

        e, n, u = enu_pos
        return (n, e, -u)

    def quaternion_ned_to_enu(self, quat_ned: Tuple[float, float, float, float]) - Tuple[float, float, float, float]:

        w, x, y, z = quat_ned

        rotation = R.from_quat([x, y, z, w])

        ned_to_enu_rot = R.from_euler('zx', [np.pi/2, np.pi])
        enu_rotation = ned_to_enu_rot  rotation

        enu_quat = enu_rotation.as_quat()
        return (float(enu_quat[3]), float(enu_quat[0]), float(enu_quat[1]), float(enu_quat[2]))
