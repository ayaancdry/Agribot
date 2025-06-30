# ~/sim_ws/src/pot_follower/scripts/extract_images_from_bag.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage # Added CompressedImage
from cv_bridge import CvBridge
import cv2
import os
from rosbags.highlevel import AnyReader
from pathlib import Path

class BagImageExtractor(Node):
    def __init__(self, bag_path, output_dir, image_topic):
        super().__init__('bag_image_extractor')
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.image_topic = image_topic
        self.bridge = CvBridge()
        self.image_count = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            self.get_logger().info(f"Created output directory: {self.output_dir}")

        self.get_logger().info(f"Extracting images from bag: {self.bag_path}")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info(f"Image topic: {self.image_topic}")

    def extract_images(self):
        try:
            with AnyReader([Path(self.bag_path)]) as reader:
                # Filter connections for the specified image topic
                image_connections = [
                    c for c in reader.connections if c.topic == self.image_topic
                ]

                if not image_connections:
                    self.get_logger().error(f"Image topic '{self.image_topic}' not found in the bag file.")
                    return

                self.get_logger().info(f"Found {len(image_connections)} connections for topic '{self.image_topic}'.")

                for connection, timestamp, rawdata in reader.messages(connections=image_connections):
                    try:
                        msg = reader.deserialize(rawdata, connection.msgtype)

                        cv_image = None
                        if connection.msgtype == 'sensor_msgs/msg/Image':
                            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                        elif connection.msgtype == 'sensor_msgs/msg/CompressedImage':
                            np_arr = cv2.imdecode(msg.data, cv2.IMREAD_COLOR)
                            cv_image = np_arr
                        else:
                            self.get_logger().warn(f"Unsupported image message type: {connection.msgtype}")
                            continue

                        if cv_image is not None:
                            image_filename = os.path.join(self.output_dir, f"image_{self.image_count:06d}.jpg")
                            cv2.imwrite(image_filename, cv_image)
                            self.get_logger().info(f"Saved {image_filename}")
                            self.image_count += 1

                    except Exception as e:
                        self.get_logger().error(f"Error processing image message: {e}")
                        continue

            self.get_logger().info(f"Finished extracting {self.image_count} images.")

        except Exception as e:
            self.get_logger().error(f"Error opening or reading bag file: {e}")

def main_run(self):
    self.extract_images()
    self.destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # USER CONFIGURATION:
    # IMPORTANT: Update these paths for your environment
    bag_folder = os.path.expanduser('~/sim_ws/rosbag_pot_data') # Path to the folder containing your bag file (.db3)
    output_images_folder = os.path.expanduser('~/yolo_pot_dataset/images/train') # Where to save the images
    image_topic = '/front_camera/image_raw' # Your camera topic

    extractor = BagImageExtractor(bag_folder, output_images_folder, image_topic)
    extractor.extract_images()
    extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()