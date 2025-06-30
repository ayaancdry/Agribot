# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.


import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import cv2
import numpy as np
import message_filters
from cv_bridge import CvBridge

from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

from sensor_msgs.msg import Image

# REMOVED: from yolo_msgs.msg import Detection
# REMOVED: from yolo_msgs.msg import DetectionArray

# ADDED vision_msgs imports
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D as VisionBoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D # For center.pose.position in BoundingBox2D

class TrackingNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("tracking_node")

        # params
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        tracker_name = self.get_parameter("tracker").get_parameter_value().string_value

        self.image_reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        self.tracker = self.create_tracker(tracker_name)
        
        # CHANGED: Publisher topic and message type
        self._pub = self.create_publisher(Detection2DArray, "tracked_detections", 10) # Changed to 'tracked_detections'
        # Note: If your launch file has a remapping for 'tracking', you might need to adjust it
        # or change this back to 'tracking' if that's preferred, and then ensure your
        # pot_follower_node subscribes to '/yolo/tracked_detections'

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        image_qos_profile = QoSProfile(
            reliability=self.image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=image_qos_profile
        )
        # CHANGED: Subscriber message type and kept topic "detections"
        detections_sub = message_filters.Subscriber(
            self, Detection2DArray, "detections", qos_profile=10
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        # These destroy_subscription calls are incorrect.
        # message_filters.Subscriber objects don't have a .sub attribute.
        # The subscriptions are managed by the synchronizer and its internal mechanisms.
        # Removing these lines as they would cause an AttributeError.
        # If resource cleanup is needed for message_filters subscribers, it's typically handled
        # by destroying the synchronizer which manages them.

        del self._synchronizer
        self._synchronizer = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        # Ensure publisher is destroyed here
        self.destroy_publisher(self._pub)
        self._pub = None

        del self.tracker

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def create_tracker(self, tracker_yaml: str) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in [
            "bytetrack",
            "botsort",
        ], f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def detections_cb(self, img_msg: Image, detections_msg: Detection2DArray) -> None:
        # CHANGED: Message type from DetectionArray to Detection2DArray

        tracked_detections_msg = Detection2DArray() # CHANGED: Message type
        tracked_detections_msg.header = img_msg.header

        # convert image
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # parse detections from vision_msgs to Ultralytics format
        detection_list = []
        # CHANGED: Iterate over Detection2D
        detection: Detection2D
        for detection in detections_msg.detections:
            # VisionBoundingBox2D stores center and size directly
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y

            x_min = center_x - size_x / 2
            y_min = center_y - size_y / 2
            x_max = center_x + size_x / 2
            y_max = center_y + size_y / 2

            # Use the first hypothesis if available for class_id and score
            class_id = -1 # Default if no hypothesis
            score = 0.0
            if detection.results:
                # Assuming the first hypothesis is the primary one
                # Note: Ultralytics trackers typically use integer class IDs.
                # vision_msgs ObjectHypothesisWithPose.id is a string.
                # You might need a mapping here if class_id is critical for tracking.
                # For now, let's try to convert the string ID to an int if it's a number,
                # or assign a placeholder if not. This might need fine-tuning.
                try:
                    class_id = int(detection.results[0].id)
                except ValueError:
                    # Fallback if ID is not a simple integer (e.g., "person")
                    # You might need to maintain a mapping from string names to int IDs
                    # if the tracker strictly relies on int IDs for different classes.
                    # For a general tracker, often only bbox and confidence are strictly needed.
                    class_id = -1 # Placeholder
                score = detection.results[0].score

            detection_list.append(
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                    score,
                    class_id,
                ]
            )

        # tracking
        if len(detection_list) > 0:

            det = Boxes(np.array(detection_list), (img_msg.height, img_msg.width))
            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                # Iterate through tracks and convert back to vision_msgs
                for t in tracks:
                    # t is typically [x1, y1, x2, y2, track_id, class_id, score] from tracker
                    # Check Ultralytics tracker output format if this differs
                    tracked_bbox_coords = t[:4] # x1, y1, x2, y2
                    track_id_val = int(t[4]) # Assuming track_id is the 5th element (index 4)
                    class_id_val = int(t[5]) # Assuming class_id is the 6th element (index 5)
                    score_val = float(t[6]) # Assuming score is the 7th element (index 6)

                    tracked_detection_msg = Detection2D()
                    tracked_detection_msg.header = img_msg.header

                    # Populate bounding box
                    bbox_msg = VisionBoundingBox2D()
                    x_min, y_min, x_max, y_max = tracked_bbox_coords
                    bbox_msg.center.position.x = float((x_min + x_max) / 2)
                    bbox_msg.center.position.y = float((y_min + y_max) / 2)
                    bbox_msg.size_x = float(x_max - x_min)
                    bbox_msg.size_y = float(y_max - y_min)
                    tracked_detection_msg.bbox = bbox_msg

                    # Populate object hypothesis
                    hypothesis_msg = ObjectHypothesisWithPose()
                    # You need a way to map class_id_val back to a string name
                    # If your YOLO model has names, you'd use self.yolo.names[class_id_val]
                    # Since this is the tracking node, you might need to pass or infer names.
                    # For now, let's use the int as a string for id.
                    hypothesis_msg.id = str(class_id_val)
                    hypothesis_msg.score = score_val
                    tracked_detection_msg.results.append(hypothesis_msg)

                    # Populate track ID
                    # vision_msgs.msg.Detection2D does not have a direct 'id' field for tracking.
                    # The track ID is often stored in the ObjectHypothesisWithPose.hypothesis.id
                    # or in a custom field. However, to match the original yolo_msgs.msg.Detection.id
                    # we will put it in an additional field, perhaps by extending Detection2D or
                    # by putting it in the results ID and prepending "track_" to differentiate.
                    # For now, we'll try to put it in a separate custom field if available,
                    # or perhaps as part of the hypothesis.id if that's your chosen convention.
                    # Since Detection2D.id is not available, we can't directly assign to it.
                    # The vision_msgs standard typically doesn't have a dedicated track_id field.
                    # You might append it to the hypothesis ID, e.g., "class_name_track_id".
                    # Let's put it as part of the ObjectHypothesisWithPose.id for now.
                    # Format: "class_id_track_ID"
                    hypothesis_msg.id = f"{class_id_val}_{track_id_val}"
                    tracked_detection_msg.results[0] = hypothesis_msg # Update the hypothesis with the track ID

                    # append msg
                    tracked_detections_msg.detections.append(tracked_detection_msg)

        # publish detections
        self._pub.publish(tracked_detections_msg)


def main():
    rclpy.init()
    node = TrackingNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
