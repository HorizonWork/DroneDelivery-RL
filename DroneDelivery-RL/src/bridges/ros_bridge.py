"""
ROS2 Bridge  
Manages ROS2 integration for SLAM data exchange and system coordination.
Connects with ORB-SLAM3 ROS2 nodes from old report setup.
"""

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
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    import logging
    logging.warning("ROS2 not available - ROS bridge disabled")

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import threading
import time

# CV bridge for image conversion
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    logging.warning("cv_bridge not available - image processing disabled")

@dataclass
class SLAMPose:
    """SLAM pose estimate."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.0

@dataclass
class ROSTopics:
    """ROS topic names configuration."""
    # Camera topics (matching old report: stereo at 30Hz)
    stereo_left: str = "/airsim_node/Drone0/front_left/Image"
    stereo_right: str = "/airsim_node/Drone0/front_right/Image"
    
    # IMU topic (matching old report: 200Hz)
    imu: str = "/airsim_node/Drone0/Imu"
    
    # SLAM output topics
    slam_pose: str = "/slam/pose"
    slam_map_points: str = "/slam/map_points"
    slam_trajectory: str = "/slam/trajectory"
    slam_ate: str = "/slam/ate"
    
    # Control topics
    velocity_command: str = "/drone/cmd_vel"
    goal_pose: str = "/drone/goal"
    
    # Status topics
    battery: str = "/drone/battery"
    collision: str = "/drone/collision"

class ROSBridge:
    """
    ROS2 bridge for SLAM and sensor data integration.
    Connects AirSim data with ORB-SLAM3 ROS2 nodes.
    Falls back to dummy implementation when ROS2 is not available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        if not ROS_AVAILABLE:
            # Set up a dummy implementation when ROS is not available
            self.config = config
            self.logger = None
            self.topics = ROSTopics()
            if 'topics' in config:
                for key, value in config['topics'].items():
                    setattr(self.topics, key, value)
            
            # Initialize with None values for all ROS-related attributes
            self.cv_bridge = None
            self.latest_slam_pose = None
            self.latest_map_points = None
            self.slam_trajectory = []
            self.latest_ate = 0.0
            self.pose_callbacks = []
            self.map_callbacks = []
            self.ros_thread = None
            self.is_running = False
            
            logging.warning("ROS2 Bridge initialized in dummy mode (ROS2 not available)")
            return
        
        rclpy.init()
        # Only import Node when ROS is available
        from rclpy.node import Node
        Node.__init__(self, 'drone_delivery_bridge')
        
        self.config = config
        self.logger = self.get_logger()
        
        # Topic configuration
        self.topics = ROSTopics()
        if 'topics' in config:
            for key, value in config['topics'].items():
                setattr(self.topics, key, value)
        
        # QoS profiles
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.control_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Initialize CV bridge if available
        self.cv_bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        
        # Data storage
        self.latest_slam_pose: Optional[SLAMPose] = None
        self.latest_map_points: Optional[np.ndarray] = None
        self.slam_trajectory: List[SLAMPose] = []
        self.latest_ate: float = 0.0
        
        # Callbacks
        self.pose_callbacks: List[Callable[[SLAMPose], None]] = []
        self.map_callbacks: List[Callable[[np.ndarray], None]] = []
        
        # Threading
        self.ros_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Initialize publishers and subscribers if ROS is available
        if ROS_AVAILABLE:
            self._setup_publishers()
            self._setup_subscribers()
            
            # Transform handling
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_broadcaster = TransformBroadcaster(self)
        
        if ROS_AVAILABLE:
            self.logger.info("ROS2 Bridge initialized")
        else:
            logging.warning("ROS2 Bridge not initialized due to missing ROS2 dependencies")
    
    def _setup_publishers(self):
        """Setup ROS2 publishers."""
        if not ROS_AVAILABLE:
            return
            
        # Camera image publishers (for SLAM input)
        self.stereo_left_pub = self.create_publisher(
            Image, self.topics.stereo_left, self.sensor_qos
        )
        self.stereo_right_pub = self.create_publisher(
            Image, self.topics.stereo_right, self.sensor_qos
        )
        
        # IMU publisher (for SLAM input)
        self.imu_pub = self.create_publisher(
            Imu, self.topics.imu, self.sensor_qos
        )
        
        # Control publishers
        self.velocity_cmd_pub = self.create_publisher(
            TwistStamped, self.topics.velocity_command, self.control_qos
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, self.topics.goal_pose, self.control_qos
        )
        
        # Status publishers
        self.battery_pub = self.create_publisher(
            Float32, self.topics.battery, self.control_qos
        )
        self.collision_pub = self.create_publisher(
            Bool, self.topics.collision, self.control_qos
        )
        
    def _setup_subscribers(self):
        """Setup ROS2 subscribers."""
        if not ROS_AVAILABLE:
            return
            
        # SLAM output subscribers
        self.slam_pose_sub = self.create_subscription(
            PoseStamped, self.topics.slam_pose,
            self._slam_pose_callback, self.sensor_qos
        )
        
        self.slam_points_sub = self.create_subscription(
            PointCloud2, self.topics.slam_map_points,
            self._map_points_callback, self.sensor_qos
        )
        
        self.slam_trajectory_sub = self.create_subscription(
            Path, self.topics.slam_trajectory,
            self._trajectory_callback, self.sensor_qos
        )
        
        self.slam_ate_sub = self.create_subscription(
            Float32, self.topics.slam_ate,
            self._ate_callback, self.sensor_qos
        )
    
    def _slam_pose_callback(self, msg: PoseStamped):
        """Handle SLAM pose updates."""
        if not ROS_AVAILABLE:
            return  # Skip callback if ROS is not available
            
        pose = SLAMPose(
            position=(
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ),
            orientation=(
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ),
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        )
        
        self.latest_slam_pose = pose
        
        # Notify callbacks
        for callback in self.pose_callbacks:
            try:
                callback(pose)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Pose callback error: {e}")
    
    def _map_points_callback(self, msg: PointCloud2):
        """Handle SLAM map points updates."""
        if not ROS_AVAILABLE:
            return  # Skip callback if ROS is not available
            
        try:
            # Convert PointCloud2 to numpy array
            # This is a simplified conversion - full implementation would use sensor_msgs_py
            points = self._pointcloud2_to_array(msg)
            self.latest_map_points = points
            
            # Notify callbacks
            for callback in self.map_callbacks:
                try:
                    callback(points)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Map callback error: {e}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Map points conversion error: {e}")
    
    def _trajectory_callback(self, msg: Path):
        """Handle SLAM trajectory updates."""
        if not ROS_AVAILABLE:
            return  # Skip callback if ROS is not available
            
        trajectory = []
        
        for pose_stamped in msg.poses:
            pose = SLAMPose(
                position=(
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y,
                    pose_stamped.pose.position.z
                ),
                orientation=(
                    pose_stamped.pose.orientation.w,
                    pose_stamped.pose.orientation.x,
                    pose_stamped.pose.orientation.y,
                    pose_stamped.pose.orientation.z
                ),
                timestamp=pose_stamped.header.stamp.sec + pose_stamped.header.stamp.nanosec * 1e-9
            )
            trajectory.append(pose)
        
        self.slam_trajectory = trajectory
    
    def _ate_callback(self, msg: Float32):
        """Handle ATE (Absolute Trajectory Error) updates."""
        if not ROS_AVAILABLE:
            return  # Skip callback if ROS is not available
            
        self.latest_ate = msg.data
    
    def publish_stereo_images(self, left_image: np.ndarray, right_image: np.ndarray,
                             timestamp: Optional[float] = None):
        """
        Publish stereo camera images to ROS2.
        
        Args:
            left_image: Left camera image (numpy array)
            right_image: Right camera image (numpy array)
            timestamp: Image timestamp (uses current time if None)
        """
        if not ROS_AVAILABLE or not self.cv_bridge:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        # Create ROS timestamp
        ros_time = self.get_clock().now().to_msg()
        
        try:
            # Convert images to ROS messages
            left_msg = self.cv_bridge.cv2_to_imgmsg(left_image, encoding='rgb8')
            right_msg = self.cv_bridge.cv2_to_imgmsg(right_image, encoding='rgb8')
            
            # Set headers
            left_msg.header.stamp = ros_time
            left_msg.header.frame_id = 'drone_camera_left'
            
            right_msg.header.stamp = ros_time
            right_msg.header.frame_id = 'drone_camera_right'
            
            # Publish
            self.stereo_left_pub.publish(left_msg)
            self.stereo_right_pub.publish(right_msg)
            
        except Exception as e:
            self.logger.error(f"Error publishing stereo images: {e}")
    
    def publish_imu_data(self, imu_data: Dict[str, Any]):
        """
        Publish IMU data to ROS2.
        
        Args:
            imu_data: IMU data dictionary from AirSim
        """
        if not ROS_AVAILABLE:
            return
            
        try:
            imu_msg = Imu()
            
            # Header
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'drone_imu'
            
            # Orientation
            imu_msg.orientation.w = imu_data['orientation'][0]
            imu_msg.orientation.x = imu_data['orientation'][1]
            imu_msg.orientation.y = imu_data['orientation'][2]
            imu_msg.orientation.z = imu_data['orientation'][3]
            
            # Angular velocity
            imu_msg.angular_velocity.x = imu_data['angular_velocity'][0]
            imu_msg.angular_velocity.y = imu_data['angular_velocity'][1]
            imu_msg.angular_velocity.z = imu_data['angular_velocity'][2]
            
            # Linear acceleration
            imu_msg.linear_acceleration.x = imu_data['linear_acceleration'][0]
            imu_msg.linear_acceleration.y = imu_data['linear_acceleration'][1]
            imu_msg.linear_acceleration.z = imu_data['linear_acceleration'][2]
            
            # Publish
            self.imu_pub.publish(imu_msg)
            
        except Exception as e:
            self.logger.error(f"Error publishing IMU data: {e}")
    
    def publish_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """
        Publish velocity command.
        
        Args:
            vx, vy, vz: Linear velocities (m/s)
            yaw_rate: Angular velocity (rad/s)
        """
        if not ROS_AVAILABLE:
            return
            
        try:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'drone_base_link'
            
            twist_msg.twist.linear.x = vx
            twist_msg.twist.linear.y = vy
            twist_msg.twist.linear.z = vz
            twist_msg.twist.angular.z = yaw_rate
            
            self.velocity_cmd_pub.publish(twist_msg)
            
        except Exception as e:
            self.logger.error(f"Error publishing velocity command: {e}")
    
    def publish_status(self, battery_level: float, collision: bool):
        """
        Publish drone status information.
        
        Args:
            battery_level: Battery level [0, 1]
            collision: Collision detected flag
        """
        if not ROS_AVAILABLE:
            return
            
        try:
            # Battery
            battery_msg = Float32()
            battery_msg.data = battery_level
            self.battery_pub.publish(battery_msg)
            
            # Collision
            collision_msg = Bool()
            collision_msg.data = collision
            self.collision_pub.publish(collision_msg)
            
        except Exception as e:
            self.logger.error(f"Error publishing status: {e}")
    
    def get_slam_pose(self) -> Optional[SLAMPose]:
        """
        Get latest SLAM pose estimate.
        
        Returns:
            Latest SLAM pose or None if not available
        """
        return self.latest_slam_pose
    
    def get_slam_ate(self) -> float:
        """
        Get latest ATE (Absolute Trajectory Error) from SLAM.
        
        Returns:
            ATE in meters
        """
        return self.latest_ate
    
    def add_pose_callback(self, callback: Callable[[SLAMPose], None]):
        """Add callback for SLAM pose updates."""
        self.pose_callbacks.append(callback)
    
    def add_map_callback(self, callback: Callable[[np.ndarray], None]):
        """Add callback for map point updates."""
        self.map_callbacks.append(callback)
    
    def _pointcloud2_to_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        """
        Convert PointCloud2 message to numpy array.
        Simplified implementation - full version would use sensor_msgs_py.
        
        Args:
            cloud_msg: PointCloud2 message
            
        Returns:
            Numpy array of points [N, 3]
        """
        # This is a placeholder implementation
        # In practice, would use sensor_msgs_py.point_cloud2 module
        
        # Extract basic XYZ points (simplified)
        if cloud_msg.width * cloud_msg.height == 0:
            return np.array([]).reshape(0, 3)
        
        # Dummy implementation - return empty array
        # Real implementation would parse the binary data
        return np.array([]).reshape(0, 3)
    
    def start_ros_thread(self):
        """Start ROS2 spinning in separate thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.ros_thread = threading.Thread(target=self._ros_spin_thread)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        
        if self.logger:
            self.logger.info("ROS2 thread started")
    
    def stop_ros_thread(self):
        """Stop ROS2 spinning thread."""
        self.is_running = False
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=1.0)
        
        if self.logger:
            self.logger.info("ROS2 thread stopped")
    
    def _ros_spin_thread(self):
        """ROS2 spinning thread function."""
        while self.is_running and ROS_AVAILABLE:
            try:
                if rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.1)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"ROS spin error: {e}")
                time.sleep(0.1)
    
    def shutdown(self):
        """Shutdown ROS2 bridge."""
        self.stop_ros_thread()
        
        if ROS_AVAILABLE:
        if ROS_AVAILABLE:
            try:
                self.destroy_node()
            except:
                pass  # Node might already be destroyed
            
            if rclpy.ok():
                rclpy.shutdown()
        
        if self.logger:
            self.logger.info("ROS2 Bridge shutdown")
