#!/usr/bin/env python3
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from pose_interfaces.msg import PointConf, TrackedKeypoints3D, TrackedKeypoints3DArray

try:
    import onnxruntime as ort
except Exception:
    ort = None


# BodyPoseNet labels are OpenPose-style 18 keypoints. The output message used by
# the rest of this workspace expects COCO-17 order.
BODYPOSE_TO_COCO17 = [
    0,   # nose
    15,  # left_eye
    14,  # right_eye
    17,  # left_ear
    16,  # right_ear
    5,   # left_shoulder
    2,   # right_shoulder
    6,   # left_elbow
    3,   # right_elbow
    7,   # left_wrist
    4,   # right_wrist
    11,  # left_hip
    8,   # right_hip
    12,  # left_knee
    9,   # right_knee
    13,  # left_ankle
    10,  # right_ankle
]

COCO17_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]

KEYPOINT_COLORS = [
    (0, 255, 255),
    (255, 255, 0),
    (255, 255, 0),
    (255, 128, 0),
    (255, 128, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 192, 255),
    (0, 192, 255),
    (0, 128, 255),
    (0, 128, 255),
    (255, 0, 255),
    (255, 0, 255),
    (192, 0, 255),
    (192, 0, 255),
    (128, 0, 255),
    (128, 0, 255),
]


def _shape_value(value, fallback: Optional[int] = None) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


class BodyPoseNetOnnxNode(Node):
    def __init__(self):
        super().__init__("bodyposenet_onnx_node")

        self.declare_parameter(
            "model_path",
            "./bodyposenet_deployable_onnx_v1.0.1/model.onnx",
        )
        self.declare_parameter("image_topic", "/rgb")
        self.declare_parameter("compressed", False)
        self.declare_parameter("output_topic", "/humans/bodyposenet/keypoints")
        self.declare_parameter("annotated_image_topic", "/human_keypoints")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("providers", ["CPUExecutionProvider"])
        self.declare_parameter("onnx_intra_op_num_threads", 1)
        self.declare_parameter("onnx_inter_op_num_threads", 1)
        self.declare_parameter("max_fps", 10.0)
        self.declare_parameter("publish_annotated_image", True)

        self.declare_parameter("input_width", 448)
        self.declare_parameter("input_height", 320)
        self.declare_parameter("input_scale", 0.00392156862745098)
        self.declare_parameter("input_mean", [0.0, 0.0, 0.0])
        self.declare_parameter("input_std", [1.0, 1.0, 1.0])
        self.declare_parameter("rgb_input", True)

        self.declare_parameter("heatmap_output", "")
        self.declare_parameter("heatmap_layout", "auto")
        self.declare_parameter("num_bodypose_keypoints", 18)
        self.declare_parameter("min_confidence", 0.10)
        self.declare_parameter("apply_sigmoid", False)
        self.declare_parameter("publish_empty", True)

        if ort is None:
            raise RuntimeError(
                "onnxruntime is required to run bodyposenet_onnx_node. "
                "Install it in this ROS Python environment, for example: "
                "python3 -m pip install onnxruntime"
            )

        model_path = str(self.get_parameter("model_path").value)
        providers = list(self.get_parameter("providers").value)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = int(self.get_parameter("onnx_intra_op_num_threads").value)
        session_options.inter_op_num_threads = int(self.get_parameter("onnx_inter_op_num_threads").value)
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers,
        )
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.output_metas = self.session.get_outputs()

        self.input_height, self.input_width = self._resolve_input_size(self.input_meta.shape)
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.min_frame_period_ns = int(1_000_000_000 / self.max_fps) if self.max_fps > 0.0 else 0
        self.next_allowed_frame_ns = 0
        self.publish_annotated_image = bool(self.get_parameter("publish_annotated_image").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            TrackedKeypoints3DArray,
            str(self.get_parameter("output_topic").value),
            10,
        )
        self.pub_annotated = None
        if self.publish_annotated_image:
            self.pub_annotated = self.create_publisher(
                Image,
                str(self.get_parameter("annotated_image_topic").value),
                10,
            )

        image_topic = str(self.get_parameter("image_topic").value)
        if bool(self.get_parameter("compressed").value):
            msg_type = CompressedImage
            sub_topic = image_topic if image_topic.endswith("/compressed") else f"{image_topic}/compressed"
        else:
            msg_type = Image
            sub_topic = image_topic

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(
            msg_type,
            sub_topic,
            self.image_cb,
            image_qos,
        )

        outputs = ", ".join([f"{m.name}{m.shape}" for m in self.output_metas])
        self.get_logger().info(
            f"Loaded BodyPoseNet ONNX: {model_path}\n"
            f" input={self.input_name}{self.input_meta.shape}, resize={self.input_width}x{self.input_height}\n"
            f" outputs={outputs}\n"
            f" providers={providers}, onnx_threads="
            f"{session_options.intra_op_num_threads}/{session_options.inter_op_num_threads}, max_fps={self.max_fps}\n"
            f" subscribed={sub_topic}, publishing={self.get_parameter('output_topic').value}, "
            f"publish_annotated_image={self.publish_annotated_image}"
        )

    def _resolve_input_size(self, shape) -> Tuple[int, int]:
        param_h = int(self.get_parameter("input_height").value)
        param_w = int(self.get_parameter("input_width").value)
        if len(shape) != 4:
            return param_h, param_w

        # Most TAO exported image models use NCHW. If the channel position is
        # dynamic, keep the parameter defaults.
        if shape[1] in (1, 3):
            return _shape_value(shape[2], param_h), _shape_value(shape[3], param_w)
        if shape[3] in (1, 3):
            return _shape_value(shape[1], param_h), _shape_value(shape[2], param_w)
        return param_h, param_w

    def _decode_image(self, msg) -> np.ndarray:
        if isinstance(msg, CompressedImage):
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if bool(self.get_parameter("rgb_input").value):
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        arr = resized.astype(np.float32) * float(self.get_parameter("input_scale").value)
        mean = np.asarray(list(self.get_parameter("input_mean").value), dtype=np.float32)
        std = np.asarray(list(self.get_parameter("input_std").value), dtype=np.float32)
        arr = (arr - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)

        shape = self.input_meta.shape
        if len(shape) == 4 and shape[1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def _select_heatmap(self, outputs: List[np.ndarray]) -> Optional[np.ndarray]:
        requested = str(self.get_parameter("heatmap_output").value).strip()
        if requested:
            for meta, value in zip(self.output_metas, outputs):
                if meta.name == requested:
                    return value
            self.get_logger().warn(f"Configured heatmap_output='{requested}' was not found")

        num_kpts = int(self.get_parameter("num_bodypose_keypoints").value)
        candidates = []
        for value in outputs:
            arr = np.asarray(value)
            if arr.ndim not in (3, 4):
                continue
            squeezed = arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr
            if squeezed.ndim != 3:
                continue
            channels = self._infer_channel_count(squeezed)
            if channels is None:
                continue
            if num_kpts <= channels <= num_kpts + 2:
                candidates.append((channels, squeezed))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        return np.asarray(outputs[0]) if outputs else None

    def _infer_channel_count(self, arr: np.ndarray) -> Optional[int]:
        layout = str(self.get_parameter("heatmap_layout").value).lower()
        if layout == "nchw":
            return int(arr.shape[0])
        if layout == "nhwc":
            return int(arr.shape[2])

        num_kpts = int(self.get_parameter("num_bodypose_keypoints").value)
        if num_kpts <= arr.shape[0] <= num_kpts + 2:
            return int(arr.shape[0])
        if num_kpts <= arr.shape[2] <= num_kpts + 2:
            return int(arr.shape[2])
        return None

    def _as_hwc_heatmap(self, heatmap: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(heatmap)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            return None

        layout = str(self.get_parameter("heatmap_layout").value).lower()
        num_kpts = int(self.get_parameter("num_bodypose_keypoints").value)
        if layout == "nchw" or (layout == "auto" and num_kpts <= arr.shape[0] <= num_kpts + 2):
            arr = np.transpose(arr, (1, 2, 0))
        elif layout != "nhwc" and not (num_kpts <= arr.shape[2] <= num_kpts + 2):
            return None

        return arr[:, :, :num_kpts]

    def _extract_keypoints(self, heatmaps: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        heatmaps = self._as_hwc_heatmap(heatmaps)
        if heatmaps is None:
            return None

        if bool(self.get_parameter("apply_sigmoid").value):
            heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))

        min_conf = float(self.get_parameter("min_confidence").value)
        image_h, image_w = image_shape
        hm_h, hm_w = heatmaps.shape[:2]

        bodypose_xyc = np.full((int(self.get_parameter("num_bodypose_keypoints").value), 3), -1.0, dtype=np.float32)
        for bodypose_idx in range(bodypose_xyc.shape[0]):
            plane = heatmaps[:, :, bodypose_idx]
            flat_idx = int(np.argmax(plane))
            y, x = np.unravel_index(flat_idx, plane.shape)
            conf = float(plane[y, x])
            if not np.isfinite(conf) or conf < min_conf:
                continue
            bodypose_xyc[bodypose_idx, 0] = float(x) * float(image_w - 1) / float(max(hm_w - 1, 1))
            bodypose_xyc[bodypose_idx, 1] = float(y) * float(image_h - 1) / float(max(hm_h - 1, 1))
            bodypose_xyc[bodypose_idx, 2] = conf

        coco_xyzc = np.full((17, 4), -1.0, dtype=np.float32)
        for coco_idx, bodypose_idx in enumerate(BODYPOSE_TO_COCO17):
            if bodypose_idx >= bodypose_xyc.shape[0]:
                continue
            x, y, conf = bodypose_xyc[bodypose_idx]
            if conf < 0.0:
                continue
            coco_xyzc[coco_idx, 0] = x
            coco_xyzc[coco_idx, 1] = y
            coco_xyzc[coco_idx, 2] = 0.0
            coco_xyzc[coco_idx, 3] = conf

        if np.all(coco_xyzc[:, 3] < 0.0):
            return None
        return coco_xyzc

    def _valid_draw_point(self, keypoints: np.ndarray, idx: int) -> bool:
        if idx >= keypoints.shape[0]:
            return False
        x, y, _, conf = keypoints[idx]
        return bool(conf >= 0.0 and np.isfinite(x) and np.isfinite(y) and x >= 0.0 and y >= 0.0)

    def _draw_keypoints(self, image: np.ndarray, keypoints: Optional[np.ndarray]) -> np.ndarray:
        annotated = image.copy()
        if keypoints is None:
            return annotated

        for start_idx, end_idx in COCO17_SKELETON:
            if not (self._valid_draw_point(keypoints, start_idx) and self._valid_draw_point(keypoints, end_idx)):
                continue
            start = (int(round(keypoints[start_idx, 0])), int(round(keypoints[start_idx, 1])))
            end = (int(round(keypoints[end_idx, 0])), int(round(keypoints[end_idx, 1])))
            cv2.line(annotated, start, end, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        for idx in range(min(17, keypoints.shape[0])):
            if not self._valid_draw_point(keypoints, idx):
                continue
            center = (int(round(keypoints[idx, 0])), int(round(keypoints[idx, 1])))
            cv2.circle(annotated, center, 5, KEYPOINT_COLORS[idx], -1, lineType=cv2.LINE_AA)
            cv2.circle(annotated, center, 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return annotated

    def _publish_annotated_image(self, header: Header, image: np.ndarray, keypoints: Optional[np.ndarray]):
        if self.pub_annotated is None:
            return
        annotated = self._draw_keypoints(image, keypoints)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        msg.header = header
        self.pub_annotated.publish(msg)

    def _make_person_msg(self, header: Header, keypoints: np.ndarray) -> TrackedKeypoints3D:
        person = TrackedKeypoints3D()
        person.header = header
        person.track_id = 0

        points = []
        for idx in range(17):
            point = PointConf()
            point.x = float(keypoints[idx, 0])
            point.y = float(keypoints[idx, 1])
            point.z = float(keypoints[idx, 2])
            point.conf = float(keypoints[idx, 3])
            points.append(point)
        person.keypoints = points
        return person

    def _publish(self, header: Header, people: List[TrackedKeypoints3D]):
        msg = TrackedKeypoints3DArray()
        msg.header = header
        msg.people = people
        self.pub.publish(msg)

    def image_cb(self, msg):
        now_ns = time.monotonic_ns()
        if self.min_frame_period_ns > 0 and now_ns < self.next_allowed_frame_ns:
            return
        self.next_allowed_frame_ns = now_ns + self.min_frame_period_ns

        try:
            image = self._decode_image(msg)
            model_input = self._preprocess(image)
            outputs = self.session.run(None, {self.input_name: model_input})
            heatmap = self._select_heatmap(outputs)
            keypoints = self._extract_keypoints(heatmap, image.shape[:2]) if heatmap is not None else None
        except Exception as exc:
            self.get_logger().warn(f"BodyPoseNet inference failed: {exc}")
            return

        header = Header()
        header.stamp = msg.header.stamp
        configured_frame = str(self.get_parameter("frame_id").value)
        header.frame_id = configured_frame if configured_frame else msg.header.frame_id

        self._publish_annotated_image(header, image, keypoints)

        if keypoints is None:
            if bool(self.get_parameter("publish_empty").value):
                self._publish(header, [])
            return

        self._publish(header, [self._make_person_msg(header, keypoints)])


def main():
    rclpy.init()
    node = BodyPoseNetOnnxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
