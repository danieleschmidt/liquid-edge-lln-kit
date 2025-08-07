"""ROS 2 integration for liquid neural networks."""

from typing import Optional, Dict, Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import json
import time
from dataclasses import dataclass
import threading
from queue import Queue, Empty

# ROS 2 imports (optional, only if available)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist, PoseStamped
    from sensor_msgs.msg import LaserScan, Imu, PointCloud2
    from nav_msgs.msg import OccupancyGrid, Odometry
    from std_msgs.msg import Header, Float32MultiArray
    from tf2_ros import Buffer, TransformListener
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS 2 not available. ROS integration disabled.")

from .core import LiquidNN, LiquidConfig
from .profiling import EnergyProfiler, ProfilingConfig


@dataclass
class ROS2Config:
    """Configuration for ROS 2 integration."""
    node_name: str = "liquid_controller"
    control_frequency_hz: float = 50.0
    sensor_timeout_s: float = 0.1
    enable_profiling: bool = True
    enable_diagnostics: bool = True
    tf_frame_id: str = "base_link"
    map_frame_id: str = "map"
    

class LiquidController:
    """Liquid neural network controller for ROS 2."""
    
    def __init__(self, 
                 model: LiquidNN,
                 params: Dict[str, Any],
                 config: LiquidConfig,
                 ros2_config: ROS2Config = None):
        self.model = model
        self.params = params
        self.config = config
        self.ros2_config = ros2_config or ROS2Config()
        
        # State management
        self.hidden_state = jnp.zeros((1, config.hidden_dim))
        self.last_inference_time = 0.0
        self.sensor_data = {}
        self.sensor_timestamps = {}
        
        # Performance tracking
        self.inference_times = Queue(maxsize=100)
        self.energy_estimates = Queue(maxsize=100)
        
        if self.ros2_config.enable_profiling:
            profiler_config = ProfilingConfig(
                device="ros2_simulation",
                voltage=3.3,
                sampling_rate=int(self.ros2_config.control_frequency_hz)
            )
            self.energy_profiler = EnergyProfiler(profiler_config)
        
    @classmethod
    def load(cls, model_path: str, ros2_config: ROS2Config = None) -> 'LiquidController':
        """Load a trained liquid controller from file."""
        model_path = Path(model_path)
        
        # Load model configuration
        config_path = model_path.parent / f"{model_path.stem}_config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = LiquidConfig(**config_data)
        
        # Create model
        model = LiquidNN(config)
        
        # Load parameters (simplified - in practice would use pickle/joblib)
        params_path = model_path.parent / f"{model_path.stem}_params.npy"
        if params_path.exists():
            # Placeholder - actual implementation would load JAX params
            key = jax.random.PRNGKey(42)
            dummy_input = jnp.ones((1, config.input_dim))
            params = model.init(key, dummy_input, training=False)
        else:
            raise FileNotFoundError(f"Model parameters not found: {params_path}")
        
        return cls(model, params, config, ros2_config)
    
    def preprocess_scan(self, scan_msg) -> jnp.ndarray:
        """Preprocess LaserScan message for liquid network input."""
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 not available")
            
        ranges = np.array(scan_msg.ranges)
        
        # Handle inf/nan values
        ranges = np.where(np.isfinite(ranges), ranges, scan_msg.range_max)
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)
        
        # Downsample to fixed number of rays (e.g., 8 directions)
        num_rays = 8
        if len(ranges) > num_rays:
            indices = np.linspace(0, len(ranges) - 1, num_rays, dtype=int)
            downsampled = ranges[indices]
        else:
            downsampled = np.pad(ranges, (0, max(0, num_rays - len(ranges))), 
                               constant_values=scan_msg.range_max)
        
        # Normalize to [0, 1]
        normalized = (downsampled - scan_msg.range_min) / (scan_msg.range_max - scan_msg.range_min)
        
        return jnp.array(normalized)
    
    def preprocess_imu(self, imu_msg) -> jnp.ndarray:
        """Preprocess IMU message for liquid network input."""
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 not available")
            
        # Extract linear acceleration (3D)
        accel = jnp.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y, 
            imu_msg.linear_acceleration.z
        ])
        
        # Extract angular velocity (3D)
        gyro = jnp.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])
        
        # Combine and normalize
        imu_data = jnp.concatenate([accel, gyro])
        
        # Simple normalization (in practice, use proper scaling)
        return jnp.clip(imu_data / 10.0, -1.0, 1.0)
    
    def preprocess_odometry(self, odom_msg) -> jnp.ndarray:
        """Preprocess Odometry message for liquid network input."""
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 not available")
            
        # Extract velocity information
        linear_vel = jnp.array([
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y
        ])
        
        angular_vel = odom_msg.twist.twist.angular.z
        
        # Combine velocities
        vel_data = jnp.concatenate([linear_vel, jnp.array([angular_vel])])
        
        # Normalize velocities (assuming max 2 m/s linear, 2 rad/s angular)
        linear_normalized = linear_vel / 2.0
        angular_normalized = angular_vel / 2.0
        
        return jnp.concatenate([linear_normalized, jnp.array([angular_normalized])])
    
    def infer(self, sensor_input: jnp.ndarray) -> jnp.ndarray:
        """Run inference with liquid neural network."""
        start_time = time.time()
        
        # Run model inference
        output, new_hidden = self.model.apply(
            self.params, sensor_input.reshape(1, -1), self.hidden_state, training=False
        )
        
        # Update hidden state
        self.hidden_state = new_hidden
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000000  # microseconds
        
        if not self.inference_times.full():
            self.inference_times.put(inference_time)
        
        # Estimate energy consumption
        energy_estimate = self.model.energy_estimate()
        if not self.energy_estimates.full():
            self.energy_estimates.put(energy_estimate)
        
        self.last_inference_time = time.time()
        
        return output.flatten()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        stats = {
            'avg_inference_time_us': 0.0,
            'avg_energy_estimate_mw': 0.0,
            'control_frequency_hz': 0.0,
            'hidden_state_norm': float(jnp.linalg.norm(self.hidden_state))
        }
        
        # Calculate averages from queued data
        inference_times = list(self.inference_times.queue)
        energy_estimates = list(self.energy_estimates.queue)
        
        if inference_times:
            stats['avg_inference_time_us'] = np.mean(inference_times)
        
        if energy_estimates:
            stats['avg_energy_estimate_mw'] = np.mean(energy_estimates)
        
        # Estimate actual control frequency
        current_time = time.time()
        if self.last_inference_time > 0:
            dt = current_time - self.last_inference_time
            if dt > 0:
                stats['control_frequency_hz'] = 1.0 / dt
        
        return stats


if ROS2_AVAILABLE:
    class LiquidTurtleBot(Node):
        """ROS 2 TurtleBot controller using liquid neural networks."""
        
        def __init__(self, controller: LiquidController):
            super().__init__(controller.ros2_config.node_name)
            
            self.controller = controller
            self.ros2_config = controller.ros2_config
            
            # QoS profiles
            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            control_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            
            # Publishers
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', control_qos)
            
            if self.ros2_config.enable_diagnostics:
                self.stats_pub = self.create_publisher(Float32MultiArray, '/liquid/stats', control_qos)
            
            # Subscribers
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, sensor_qos)
            
            self.imu_sub = self.create_subscription(
                Imu, '/imu', self.imu_callback, sensor_qos)
                
            self.odom_sub = self.create_subscription(
                Odometry, '/odom', self.odom_callback, sensor_qos)
            
            # Control timer
            control_period = 1.0 / self.ros2_config.control_frequency_hz
            self.control_timer = self.create_timer(control_period, self.control_loop)
            
            # Sensor data storage
            self.sensor_data = {}
            self.sensor_lock = threading.Lock()
            
            self.get_logger().info(f"Liquid TurtleBot controller started at {self.ros2_config.control_frequency_hz}Hz")
        
        def scan_callback(self, msg: LaserScan):
            """Process laser scan data."""
            try:
                processed_scan = self.controller.preprocess_scan(msg)
                
                with self.sensor_lock:
                    self.sensor_data['scan'] = processed_scan
                    self.sensor_data['scan_timestamp'] = time.time()
                    
            except Exception as e:
                self.get_logger().error(f"Error processing scan: {e}")
        
        def imu_callback(self, msg: Imu):
            """Process IMU data."""
            try:
                processed_imu = self.controller.preprocess_imu(msg)
                
                with self.sensor_lock:
                    self.sensor_data['imu'] = processed_imu
                    self.sensor_data['imu_timestamp'] = time.time()
                    
            except Exception as e:
                self.get_logger().error(f"Error processing IMU: {e}")
        
        def odom_callback(self, msg: Odometry):
            """Process odometry data."""
            try:
                processed_odom = self.controller.preprocess_odometry(msg)
                
                with self.sensor_lock:
                    self.sensor_data['odom'] = processed_odom
                    self.sensor_data['odom_timestamp'] = time.time()
                    
            except Exception as e:
                self.get_logger().error(f"Error processing odometry: {e}")
        
        def control_loop(self):
            """Main control loop."""
            try:
                # Collect sensor data
                with self.sensor_lock:
                    current_time = time.time()
                    
                    # Check for fresh sensor data
                    sensor_inputs = []
                    
                    # LaserScan data (required)
                    if 'scan' in self.sensor_data:
                        scan_age = current_time - self.sensor_data.get('scan_timestamp', 0)
                        if scan_age < self.ros2_config.sensor_timeout_s:
                            sensor_inputs.append(self.sensor_data['scan'])
                        else:
                            self.get_logger().warn("Stale scan data")
                            return
                    else:
                        return  # No scan data available
                    
                    # IMU data (optional)
                    if 'imu' in self.sensor_data:
                        imu_age = current_time - self.sensor_data.get('imu_timestamp', 0)
                        if imu_age < self.ros2_config.sensor_timeout_s:
                            sensor_inputs.append(self.sensor_data['imu'])
                    
                    # Odometry data (optional)
                    if 'odom' in self.sensor_data:
                        odom_age = current_time - self.sensor_data.get('odom_timestamp', 0)
                        if odom_age < self.ros2_config.sensor_timeout_s:
                            sensor_inputs.append(self.sensor_data['odom'])
                
                # Combine sensor inputs
                if sensor_inputs:
                    combined_input = jnp.concatenate(sensor_inputs)
                    
                    # Pad or truncate to expected input dimension
                    expected_dim = self.controller.config.input_dim
                    if len(combined_input) > expected_dim:
                        combined_input = combined_input[:expected_dim]
                    elif len(combined_input) < expected_dim:
                        padding = jnp.zeros(expected_dim - len(combined_input))
                        combined_input = jnp.concatenate([combined_input, padding])
                    
                    # Run liquid network inference
                    motor_commands = self.controller.infer(combined_input)
                    
                    # Create and publish Twist message
                    twist = Twist()
                    if len(motor_commands) >= 2:
                        twist.linear.x = float(jnp.clip(motor_commands[0], -1.0, 1.0))
                        twist.angular.z = float(jnp.clip(motor_commands[1], -2.0, 2.0))
                    
                    self.cmd_vel_pub.publish(twist)
                    
                    # Publish diagnostics
                    if self.ros2_config.enable_diagnostics:
                        self.publish_diagnostics()
                        
            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
                
                # Publish stop command for safety
                stop_twist = Twist()
                self.cmd_vel_pub.publish(stop_twist)
        
        def publish_diagnostics(self):
            """Publish performance diagnostics."""
            try:
                stats = self.controller.get_performance_stats()
                
                msg = Float32MultiArray()
                msg.data = [
                    stats['avg_inference_time_us'],
                    stats['avg_energy_estimate_mw'],
                    stats['control_frequency_hz'],
                    stats['hidden_state_norm']
                ]
                
                self.stats_pub.publish(msg)
                
            except Exception as e:
                self.get_logger().error(f"Error publishing diagnostics: {e}")
    
    
    class LiquidNavigationNode(Node):
        """Advanced navigation node with path planning."""
        
        def __init__(self, controller: LiquidController):
            super().__init__('liquid_navigation')
            
            self.controller = controller
            
            # Navigation state
            self.current_goal = None
            self.path = []
            
            # Publishers
            self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.goal_reached_pub = self.create_publisher(Header, '/goal_reached', 10)
            
            # Subscribers
            self.goal_sub = self.create_subscription(
                PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
            
            self.scan_sub = self.create_subscription(
                LaserScan, '/scan', self.scan_callback, 10)
            
            self.odom_sub = self.create_subscription(
                Odometry, '/odom', self.odom_callback, 10)
            
            # TF2 listener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            # Control timer
            self.create_timer(0.02, self.navigation_control_loop)  # 50Hz
            
            self.get_logger().info("Liquid Navigation Node started")
        
        def goal_callback(self, msg: PoseStamped):
            """Set new navigation goal."""
            self.current_goal = msg
            self.get_logger().info(f"New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        
        def scan_callback(self, msg: LaserScan):
            """Store latest scan data."""
            self.latest_scan = msg
        
        def odom_callback(self, msg: Odometry):
            """Store latest odometry."""
            self.latest_odom = msg
        
        def navigation_control_loop(self):
            """Main navigation control loop."""
            if not hasattr(self, 'latest_scan') or not hasattr(self, 'latest_odom'):
                return
            
            if self.current_goal is None:
                return
            
            try:
                # Prepare navigation input for liquid network
                nav_input = self.prepare_navigation_input()
                
                if nav_input is not None:
                    # Run liquid network inference
                    control_output = self.controller.infer(nav_input)
                    
                    # Convert to motion commands
                    twist = Twist()
                    twist.linear.x = float(jnp.clip(control_output[0], 0.0, 0.8))
                    twist.angular.z = float(jnp.clip(control_output[1], -2.0, 2.0))
                    
                    self.cmd_vel_pub.publish(twist)
                    
                    # Check if goal is reached
                    if self.is_goal_reached():
                        self.get_logger().info("Goal reached!")
                        self.current_goal = None
                        
                        # Publish goal reached signal
                        goal_msg = Header()
                        goal_msg.stamp = self.get_clock().now().to_msg()
                        self.goal_reached_pub.publish(goal_msg)
                        
                        # Stop robot
                        stop_twist = Twist()
                        self.cmd_vel_pub.publish(stop_twist)
            
            except Exception as e:
                self.get_logger().error(f"Navigation control error: {e}")
        
        def prepare_navigation_input(self) -> Optional[jnp.ndarray]:
            """Prepare input for liquid network navigation."""
            try:
                # Process scan data
                scan_data = self.controller.preprocess_scan(self.latest_scan)
                
                # Process odometry
                odom_data = self.controller.preprocess_odometry(self.latest_odom)
                
                # Calculate goal direction
                goal_data = self.calculate_goal_direction()
                
                if goal_data is not None:
                    # Combine all inputs
                    nav_input = jnp.concatenate([scan_data, odom_data, goal_data])
                    
                    # Ensure correct dimensionality
                    expected_dim = self.controller.config.input_dim
                    if len(nav_input) > expected_dim:
                        nav_input = nav_input[:expected_dim]
                    elif len(nav_input) < expected_dim:
                        padding = jnp.zeros(expected_dim - len(nav_input))
                        nav_input = jnp.concatenate([nav_input, padding])
                    
                    return nav_input
                
            except Exception as e:
                self.get_logger().error(f"Error preparing navigation input: {e}")
                
            return None
        
        def calculate_goal_direction(self) -> Optional[jnp.ndarray]:
            """Calculate direction to goal in robot frame."""
            if self.current_goal is None:
                return None
            
            try:
                # Get robot pose in map frame
                robot_pose = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                
                # Calculate relative position to goal
                goal_x = self.current_goal.pose.position.x
                goal_y = self.current_goal.pose.position.y
                
                robot_x = robot_pose.transform.translation.x
                robot_y = robot_pose.transform.translation.y
                
                # Relative position
                rel_x = goal_x - robot_x
                rel_y = goal_y - robot_y
                
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Normalize
                if distance > 0:
                    goal_direction = jnp.array([rel_x / distance, rel_y / distance, distance / 10.0])
                else:
                    goal_direction = jnp.zeros(3)
                
                return goal_direction
                
            except Exception as e:
                self.get_logger().warn(f"Could not calculate goal direction: {e}")
                return jnp.zeros(3)
        
        def is_goal_reached(self, tolerance: float = 0.3) -> bool:
            """Check if robot has reached the goal."""
            if self.current_goal is None:
                return False
            
            try:
                robot_pose = self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time())
                
                goal_x = self.current_goal.pose.position.x
                goal_y = self.current_goal.pose.position.y
                
                robot_x = robot_pose.transform.translation.x
                robot_y = robot_pose.transform.translation.y
                
                distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2)
                
                return distance < tolerance
                
            except Exception:
                return False

else:
    # Stub classes when ROS 2 is not available
    class LiquidTurtleBot:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS 2 not available. Install ROS 2 to use this functionality.")
    
    class LiquidNavigationNode:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS 2 not available. Install ROS 2 to use this functionality.")


# Convenience functions
def create_turtlebot_controller(model_path: str, ros2_config: ROS2Config = None) -> 'LiquidTurtleBot':
    """Create a TurtleBot controller with liquid neural network."""
    controller = LiquidController.load(model_path, ros2_config)
    return LiquidTurtleBot(controller)

def create_navigation_controller(model_path: str, ros2_config: ROS2Config = None) -> 'LiquidNavigationNode':
    """Create a navigation controller with liquid neural network."""
    controller = LiquidController.load(model_path, ros2_config)
    return LiquidNavigationNode(controller)
