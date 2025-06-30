# ~/sim_ws/src/pot_follower/pot_follower/pot_follower_node.py

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PotFollower(Node):
    def __init__(self):
        super().__init__('pot_follower_node')
        self.get_logger().info('Pot Follower Node has been started.')

        # Declare parameters for tuning (you can adjust these values after initial testing)
        self.declare_parameter('target_pot_class', 'flower_pot')
        self.declare_parameter('linear_speed', 0.2) # m/s (How fast the robot moves forward when centered)
        self.declare_parameter('angular_speed_gain', 0.5) # Gain for turning (How sharply it turns to center)
        self.declare_parameter('center_tolerance_x', 0.1) # Tolerance for x-axis centering (fraction of image width, e.g., 0.1 means +/- 10% of center)
        self.declare_parameter('max_pot_height_for_stop', 0.7) # Max height of bounding box (fraction of image height) to stop (0.7 means 70% of image height)
        self.declare_parameter('image_width', 640.0) # From your camera config in XACRO/Gazebo plugin
        self.declare_parameter('image_height', 480.0) # From your camera config in XACRO/Gazebo plugin
        self.declare_parameter('detection_timeout_sec', 0.5) # Seconds to wait before assuming pot is lost and stopping

        self.target_pot_class = self.get_parameter('target_pot_class').get_parameter_value().string_value
        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.angular_speed_gain = self.get_parameter('angular_speed_gain').get_parameter_value().double_value
        self.center_tolerance_x = self.get_parameter('center_tolerance_x').get_parameter_value().double_value
        self.max_pot_height_for_stop = self.get_parameter('max_pot_height_for_stop').get_parameter_value().double_value
        self.image_width = self.get_parameter('image_width').get_parameter_value().double_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().double_value
        self.detection_timeout_sec = self.get_parameter('detection_timeout_sec').get_parameter_value().double_value

        # QoS profile for subscriptions/publications (important for reliable communication)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # YOLO might publish fast, best effort is good
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber for YOLO detections
        self.subscription = self.create_subscription(
            Detection2DArray,
            '/yolo/detections', # This is the topic where yolo_ros publishes
            self.detection_callback,
            qos_profile
        )
        self.subscription  # Prevent unused variable warning

        # Publisher for robot velocity commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10) # Publish to the differential drive controller

        # Timer to periodically publish commands (even if no new detections)
        self.timer = self.create_timer(0.1, self.publish_cmd_vel) # Publish commands every 0.1 seconds (10 Hz)

        self.last_detection_time = self.get_clock().now()
        self.detected_pot = None # Stores the best (highest confidence) detected pot

    def detection_callback(self, msg):
        self.last_detection_time = self.get_clock().now()
        best_pot = None
        max_score = 0.0

        for detection in msg.detections:
            for result in detection.results:
                # Check if the detected class is our target flower pot
                # vision_msgs.msg.ObjectHypothesisWithPose has id as string (e.g., 'flower_pot')
                if result.id == self.target_pot_class:
                    if result.score > max_score:
                        max_score = result.score
                        best_pot = detection
        self.detected_pot = best_pot # Update the stored best pot

    def publish_cmd_vel(self):
        twist_msg = Twist() # Initialize a new Twist message

        # Stop if no detections for a while
        time_since_last_detection = (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9
        if time_since_last_detection > self.detection_timeout_sec:
            self.detected_pot = None # Clear the detected pot if timed out

        if self.detected_pot:
            bbox = self.detected_pot.bbox # Get the bounding box information

            # Calculate normalized x-position of the pot's center (-1 to 1, where 0 is center)
            pot_center_x = (bbox.center.position.x - (self.image_width / 2.0)) / (self.image_width / 2.0)

            # Calculate normalized height of the bounding box (0 to 1)
            pot_height_norm = bbox.size_y / self.image_height

            # Autonomy Logic:
            if pot_height_norm > self.max_pot_height_for_stop:
                # Pot is too close, stop the robot
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.get_logger().info("Flower pot too close, stopping.")
            else:
                # If pot is not centered, turn to center it
                if abs(pot_center_x) > self.center_tolerance_x:
                    twist_msg.angular.z = -self.angular_speed_gain * pot_center_x
                    twist_msg.linear.x = 0.0 # Stop linear movement while turning significantly
                    self.get_logger().info(f"Centering pot (x_norm: {pot_center_x:.2f}), angular: {twist_msg.angular.z:.2f}")
                else:
                    # Pot is roughly centered, move forward
                    twist_msg.linear.x = self.linear_speed
                    twist_msg.angular.z = 0.0
                    self.get_logger().info(f"Moving towards pot (height_norm: {pot_height_norm:.2f})")
        else:
            # No pot detected or lost track, stop the robot
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.get_logger().info("No flower pot detected or timeout, stopping movement.")

        # Publish the calculated Twist message
        self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    pot_follower = PotFollower()
    rclpy.spin(pot_follower) # Keeps the node alive and processing callbacks
    pot_follower.destroy_node() # Clean up when shutdown
    rclpy.shutdown()

if __name__ == '__main__':
    main()