# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from std_srvs.srv import SetBool
from sensor_msgs.msg import Image

'''
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses
'''

from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import BoundingBox2D as VisionBoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose

class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)

        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld}

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )

        # ros params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )

        # detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(Detection2DArray, "detections", 10)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}")

        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)

        # COMMENT OUT or REMOVE this block as SetClasses is from yolo_msgs
        # if isinstance(self.yolo, YOLOWorld):
        #     self._set_classes_srv = self.create_service(
        #         SetClasses, "set_classes", self.set_classes_cb
        #     )

        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb, self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        # COMMENT OUT or REMOVE this block as _set_classes_srv is from yolo_msgs
        # if isinstance(self.yolo, YOLOWorld):
        #     self.destroy_service(self._set_classes_srv)
        #     self._set_classes_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.enable = request.data
        response.success = True
        return response
    def parse_hypothesis(self, results: Results) -> List[ObjectHypothesisWithPose]:

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = ObjectHypothesisWithPose() # CHANGE to vision_msgs type
                hypothesis.id = self.yolo.names[int(box_data.cls)] # Use class name as ID
                hypothesis.score = float(box_data.conf)
                # No pose info in 2D detection, so leave pose field empty
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = ObjectHypothesisWithPose() # CHANGE to vision_msgs type
                hypothesis.id = self.yolo.names[int(results.obb.cls[i])] # Use class name as ID
                hypothesis.score = float(results.obb.conf[i])
                # No pose info in 2D detection, so leave pose field empty
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[VisionBoundingBox2D]:

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                # Use VisionBoundingBox2D (aliased from vision_msgs)
                msg = VisionBoundingBox2D()

                # get boxes values (vision_msgs uses min/max for bounding box)
                # xywh = [x_center, y_center, width, height]
                # xyxy = [x_min, y_min, x_max, y_max]
                x_center, y_center, width, height = box_data.xywh[0]

                msg.center.position.x = float(x_center)
                msg.center.position.y = float(y_center)
                msg.size_x = float(width)
                msg.size_y = float(height)

                # Append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = VisionBoundingBox2D() # CHANGE to vision_msgs type

                # get boxes values for OBB (xywhr = [x_center, y_center, width, height, rotation])
                x_center, y_center, width, height, rotation = results.obb.xywhr[i]

                msg.center.position.x = float(x_center)
                msg.center.position.y = float(y_center)
                msg.size_x = float(width)
                msg.size_y = float(height)
                # msg.center.theta = float(rotation) # VisionBoundingBox2D does not have theta directly, use orientation.
                # Need to convert yaw to quaternion for orientation if exact orientation is needed, but for 2D bbox this is often ignored.
                # For simplicity, we'll leave orientation as default or set a dummy.

                # Append msg
                boxes_list.append(msg)

        return boxes_list

    # def parse_masks(self, results: Results) -> List[Mask]:

    #     masks_list = []

    #     def create_point2d(x: float, y: float) -> Point2D:
    #         p = Point2D()
    #         p.x = x
    #         p.y = y
    #         return p

    #     mask: Masks
    #     for mask in results.masks:

    #         msg = Mask()

    #         msg.data = [
    #             create_point2d(float(ele[0]), float(ele[1]))
    #             for ele in mask.xy[0].tolist()
    #         ]
    #         msg.height = results.orig_img.shape[0]
    #         msg.width = results.orig_img.shape[1]

    #         masks_list.append(msg)

    #     return masks_list

    # def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

    #     keypoints_list = []

    #     points: Keypoints
    #     for points in results.keypoints:

    #         msg_array = KeyPoint2DArray()

    #         if points.conf is None:
    #             continue

    #         for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

    #             if conf >= self.threshold:
    #                 msg = KeyPoint2D()

    #                 msg.id = kp_id + 1
    #                 msg.point.x = float(p[0])
    #                 msg.point.y = float(p[1])
    #                 msg.score = float(conf)

    #                 msg_array.data.append(msg)

    #         keypoints_list.append(msg_array)

    #     return keypoints_list

    def image_cb(self, msg: Image) -> None:

        if self.enable:

            # convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding=self.yolo_encoding
            )
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                half=self.half,
                max_det=self.max_det,
                augment=self.augment,
                agnostic_nms=self.agnostic_nms,
                retina_masks=self.retina_masks,
                device=self.device,
            )
            results: Results = results[0].cpu()

            # --- START CHANGES HERE ---

            if results.boxes or results.obb:
                # Parse for vision_msgs. The return types of these functions are now vision_msgs types.
                hypotheses = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

                # Create the main Detection2DArray message (vision_msgs)
                detections_msg = Detection2DArray()
                detections_msg.header = msg.header # Use the header from the input image

                for i in range(len(hypotheses)): # Iterate through detections
                    detection = Detection2D() # Create an individual Detection2D message
                    detection.header = msg.header # Each individual detection also gets the header

                    # Assign the bounding box
                    detection.bbox = boxes[i]

                    # Assign the object hypothesis (score and ID)
                    detection.results.append(hypotheses[i]) # vision_msgs expects a list of hypotheses

                    # If you had masks/keypoints and converted them to vision_msgs, you'd add them here:
                    # if results.masks and masks:
                    #     detection.mask = masks[i]
                    # if results.keypoints and keypoints:
                    #     detection.keypoints.append(keypoints[i]) # Keypoints is a list in Detection2D

                    detections_msg.detections.append(detection) # Add the fully populated detection to the array

                # publish detections
                self._pub.publish(detections_msg)
            else:
                # If no detections, publish an empty Detection2DArray to keep the topic alive
                empty_detections_msg = Detection2DArray()
                empty_detections_msg.header = msg.header
                self._pub.publish(empty_detections_msg)

            # --- END CHANGES HERE ---

            del results
            del cv_image

    # def set_classes_cb(
    #     self,
    #     req: SetClasses.Request,
    #     res: SetClasses.Response,
    # ) -> SetClasses.Response:
    #     self.get_logger().info(f"Setting classes: {req.classes}")
    #     self.yolo.set_classes(req.classes)
    #     self.get_logger().info(f"New classes: {self.yolo.names}")
    #     return res



def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
