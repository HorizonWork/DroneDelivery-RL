import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import time

ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

    from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
    from sensor_msgs.msg import Image, Imu, PointCloud2, CompressedImage
    from nav_msgs.msg import Path, OccupancyGrid
    from std_msgs.msg import Header, Float32, Bool
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    from tf2_geometry_msgs import do_transform_pose
    import tf_transformations

    ROS2_AVAILABLE = True
    logging.info("ROS2 packages imported successfully")

except ImportError as e:
    logging.warning(f"ROS2 not available - ROS bridge disabled: {e}")
    Node = object
    PoseStamped = None
    TwistStamped = None
    Vector3Stamped = None
    Image = None
    Imu = None
    PointCloud2 = None
    CompressedImage = None
    Path = None
    OccupancyGrid = None
    Header = None
    Float32 = None
    Bool = None

CV_BRIDGE_AVAILABLE = False
CvBridge = None
try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
    logging.info("cv_bridge imported successfully")
except ImportError as e:
    logging.warning(f"cv_bridge not available - image processing disabled: {e}")

dataclass
class SLAMPose:

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0

dataclass
class ROSTopics:

    stereo_left: str = "/airsim_node/Drone0/front_left/Image"
    stereo_right: str = "/airsim_node/Drone0/front_right/Image"

    imu: str = "/airsim_node/Drone0/Imu"

    slam_pose: str = "/slam/pose"
    slam_map_points: str = "/slam/map_points"
    slam_trajectory: str = "/slam/trajectory"
    slam_ate: str = "/slam/ate"

    velocity_command: str = "/drone/cmd_vel"
    goal_pose: str = "/drone/goal"

    battery: str = "/drone/battery"
    collision: str = "/drone/collision"

class ROSBridge(Node if ROS2_AVAILABLE else object):

    def __init__(self, config: Dict[str, Any]):

        if not ROS2_AVAILABLE:
            logging.warning(
                "ROSBridge initialized but ROS2 not available. "
                "ROS integration features will be disabled."
            )
            self.enabled = False
            self.config = config
            return

        super().__init__("drone_delivery_bridge")
        self.config = config
        self.logger = self.get_logger()
        self.enabled = True

        self.topics = ROSTopics()
        if "topics" in config:
            for key, value in config["topics"].items():
                setattr(self.topics, key, value)

        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.control_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cv_bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None

        self.latest_slam_pose: Optional[SLAMPose] = None
        self.latest_map_points: Optional[np.ndarray] = None
        self.slam_trajectory: List[SLAMPose] = []
        self.latest_ate: float = 0.0

        self.pose_callbacks: List[Callable[[SLAMPose], None]] = []
        self.map_callbacks: List[Callable[[np.ndarray], None]] = []

        self.ros_thread: Optional[threading.Thread] = None
        self.is_running = False

        self._setup_publishers()
        self._setup_subscribers()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.logger.info("ROS2 Bridge initialized successfully")

    def _check_enabled(self) - bool:

        if not self.enabled:
            logging.debug("ROS2 bridge operation skipped - not enabled")
        return self.enabled

    def _setup_publishers(self):

        if not self._check_enabled():
            return

        self.stereo_left_pub = self.create_publisher(
            Image, self.topics.stereo_left, self.sensor_qos
        )
        self.stereo_right_pub = self.create_publisher(
            Image, self.topics.stereo_right, self.sensor_qos
        )

        self.imu_pub = self.create_publisher(Imu, self.topics.imu, self.sensor_qos)

        self.velocity_pub = self.create_publisher(
            TwistStamped, self.topics.velocity_command, self.control_qos
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, self.topics.goal_pose, self.control_qos
        )

        self.logger.info("ROS2 publishers created")

    def _setup_subscribers(self):

        if not self._check_enabled():
            return

        self.slam_pose_sub = self.create_subscription(
            PoseStamped,
            self.topics.slam_pose,
            self._slam_pose_callback,
            self.sensor_qos,
        )

        self.slam_map_sub = self.create_subscription(
            PointCloud2,
            self.topics.slam_map_points,
            self._map_points_callback,
            self.sensor_qos,
        )

        self.slam_trajectory_sub = self.create_subscription(
            Path,
            self.topics.slam_trajectory,
            self._trajectory_callback,
            self.sensor_qos,
        )

        self.ate_sub = self.create_subscription(
            Float32, self.topics.slam_ate, self._ate_callback, self.sensor_qos
        )

        self.logger.info("ROS2 subscribers created")

    def _slam_pose_callback(self, msg):

        if not self._check_enabled():
            return

        pose = SLAMPose(
            position=(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            orientation=(
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ),
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec  1e-9,
        )

        self.latest_slam_pose = pose

        for callback in self.pose_callbacks:
            try:
                callback(pose)
            except Exception as e:
                self.logger.error(f"Pose callback error: {e}")

    def _map_points_callback(self, msg):

        if not self._check_enabled():
            return
        pass

    def _trajectory_callback(self, msg):

        if not self._check_enabled():
            return
        pass

    def _ate_callback(self, msg):

        if not self._check_enabled():
            return
        self.latest_ate = msg.data

    def start(self):

        if not self._check_enabled():
            logging.warning("Cannot start ROS2 bridge - not enabled")
            return

        if self.is_running:
            self.logger.warning("ROS2 bridge already running")
            return

        self.is_running = True
        self.ros_thread = threading.Thread(target=self._run_ros_spin, daemon=True)
        self.ros_thread.start()
        self.logger.info("ROS2 bridge started")

    def _run_ros_spin(self):

        while self.is_running and ROS2_AVAILABLE:
            rclpy.spin_once(self, timeout_sec=0.01)

    def stop(self):

        if not self.enabled:
            return

        self.is_running = False
        if self.ros_thread:
            self.ros_thread.join(timeout=1.0)
        self.logger.info("ROS2 bridge stopped")

    def get_slam_pose(self) - Optional[SLAMPose]:

        return self.latest_slam_pose

    def get_slam_ate(self) - float:

        return self.latest_ate

    def publish_image(self, image: np.ndarray, topic: str = "left"):

        if not self._check_enabled() or not CV_BRIDGE_AVAILABLE:
            return

        try:
            msg = self.cv_bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()

            if topic == "left":
                self.stereo_left_pub.publish(msg)
            elif topic == "right":
                self.stereo_right_pub.publish(msg)
        except Exception as e:
            self.logger.error(f"Image publish error: {e}")

    def __del__(self):

        if self.enabled:
            self.stop()

def create_ros_bridge(config: Dict[str, Any]) - Optional[ROSBridge]:

    if not ROS2_AVAILABLE:
        logging.info("ROS2 not available - skipping bridge creation")
        return None

    try:
        if not rclpy.ok():
            rclpy.init()

        bridge = ROSBridge(config)
        return bridge
    except Exception as e:
        logging.error(f"Failed to create ROS bridge: {e}")
        return None

__all__ = [
    "ROSBridge",
    "SLAMPose",
    "ROSTopics",
    "ROS2_AVAILABLE",
    "CV_BRIDGE_AVAILABLE",
    "create_ros_bridge",
]
