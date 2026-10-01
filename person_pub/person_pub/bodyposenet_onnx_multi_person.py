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

# NVIDIA TAO bpnet_18joints.json order, converted to zero-based indices.
# Each limb uses consecutive x/y PAF channels 2*i and 2*i+1.
# This differs from the standard OpenPose PAF channel order.
# Reference: NVIDIA/tao_tensorflow1_backend, cv/bpnet/dataloaders/
# pose_configurations/bpnet_18joints.json and pose_config.py.
BODYPOSE_LIMBS = [
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 2), (2, 3), (3, 4), (2, 16), (1, 5), (5, 6),
    (6, 7), (5, 17), (1, 0), (0, 14), (0, 15), (14, 16), (15, 17),
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
        super().__init__("bodyposenet_onnx_multi_person")

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
        self.declare_parameter("paf_output", "")
        self.declare_parameter("paf_layout", "auto")
        self.declare_parameter("peak_min_distance", 3)
        self.declare_parameter("max_peaks_per_joint", 30)
        self.declare_parameter("paf_threshold", 0.05)
        self.declare_parameter("paf_min_fraction", 0.7)
        self.declare_parameter("paf_samples", 10)
        self.declare_parameter("min_person_keypoints", 3)

        if int(self.get_parameter("num_bodypose_keypoints").value) != 18:
            raise ValueError("Multi-person decoding requires the TAO 18-joint skeleton")
        for name in ("peak_min_distance", "max_peaks_per_joint", "min_person_keypoints"):
            if int(self.get_parameter(name).value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.get_parameter("paf_samples").value) < 2:
            raise ValueError("paf_samples must be at least 2")
        if not 0.0 < float(self.get_parameter("paf_min_fraction").value) <= 1.0:
            raise ValueError("paf_min_fraction must be in (0, 1]")
        for name in ("heatmap_layout", "paf_layout"):
            if str(self.get_parameter(name).value).lower() not in ("auto", "nchw", "nhwc"):
                raise ValueError(f"{name} must be auto, nchw, or nhwc")


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

    def _as_hwc(self, value: np.ndarray, kind: str) -> np.ndarray:
        """Normalize a batch-one feature tensor and validate its channels."""
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"{kind} must be a 3D or batch-one 4D tensor")
        channels = (18, 19) if kind == "heatmap" else (38,)
        layout = str(self.get_parameter(f"{kind}_layout").value).lower()
        if layout == "auto":
            first, last = arr.shape[0] in channels, arr.shape[2] in channels
            if first == last:
                raise ValueError(f"Ambiguous/invalid {kind} shape {arr.shape}; set {kind}_layout")
            layout = "nchw" if first else "nhwc"
        if layout == "nchw":
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] not in channels:
            raise ValueError(f"Invalid {kind} channel count: {arr.shape[2]}")
        return arr.astype(np.float32, copy=False)

    def _select_output(self, outputs: List[np.ndarray], kind: str) -> np.ndarray:
        requested = str(self.get_parameter(f"{kind}_output").value).strip()
        candidates = []
        for meta, value in zip(self.output_metas, outputs):
            if requested:
                if meta.name == requested:
                    return self._as_hwc(value, kind)
                continue
            try:
                candidates.append(self._as_hwc(value, kind))
            except ValueError:
                continue
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot select {kind} output; set {kind}_output and {kind}_layout. "
                "Multi-person decoding requires both heatmaps and 38-channel PAFs."
            )
        return candidates[0]

    def _extract_keypoints(
        self, heatmaps: np.ndarray, pafs: np.ndarray, image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Decode HWC heatmaps and PAFs into independent COCO-17 skeletons."""
        # The deployable model emits PAFs at 4x the heatmap resolution.
        # Sample them on the heatmap grid while keeping directions in PAF units.
        hm_h, hm_w = heatmaps.shape[:2]
        paf_scale = np.array([pafs.shape[1] / hm_w, pafs.shape[0] / hm_h],
                             dtype=np.float32)
        if heatmaps.shape[:2] != pafs.shape[:2]:
            pafs = cv2.resize(pafs, (hm_w, hm_h), interpolation=cv2.INTER_LINEAR)
        if bool(self.get_parameter("apply_sigmoid").value):
            heatmaps = 1.0 / (1.0 + np.exp(-np.clip(heatmaps, -80, 80)))
        min_conf = float(self.get_parameter("min_confidence").value)
        radius = int(self.get_parameter("peak_min_distance").value)
        max_peaks = int(self.get_parameter("max_peaks_per_joint").value)
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        peaks, by_joint = [], []
        for joint in range(18):
            plane = np.where(np.isfinite(heatmaps[:, :, joint]),
                             heatmaps[:, :, joint], -1.0).astype(np.float32)
            mask = (plane >= min_conf) & (plane > 0) & (plane == cv2.dilate(plane, kernel))
            ys, xs = np.nonzero(mask)
            order = np.argsort(-plane[ys, xs], kind="stable")
            selected = []
            for idx in order:
                x, y = int(xs[idx]), int(ys[idx])
                # Suppress flat-topped peaks as well as nearby maxima.
                if any((x - peaks[k][0]) ** 2 + (y - peaks[k][1]) ** 2 <= radius ** 2
                       for k in selected):
                    continue
                selected.append(len(peaks))
                peaks.append((x, y, float(plane[y, x])))
                if len(selected) >= max_peaks:
                    break
            by_joint.append(selected)

        samples = int(self.get_parameter("paf_samples").value)
        threshold = float(self.get_parameter("paf_threshold").value)
        min_fraction = float(self.get_parameter("paf_min_fraction").value)
        edges = []
        for limb, (joint_a, joint_b) in enumerate(BODYPOSE_LIMBS):
            candidates = []
            for a in by_joint[joint_a]:
                for b in by_joint[joint_b]:
                    start = np.asarray(peaks[a][:2], dtype=np.float32)
                    end = np.asarray(peaks[b][:2], dtype=np.float32)
                    delta = (end - start) * paf_scale
                    length = float(np.linalg.norm(delta))
                    if length < 1e-6:
                        continue
                    xy = np.rint(np.linspace(start, end, samples)).astype(int)
                    vectors = pafs[xy[:, 1], xy[:, 0], 2 * limb:2 * limb + 2]
                    alignment = vectors @ (delta / length)
                    if not np.all(np.isfinite(alignment)):
                        continue
                    score = float(np.mean(alignment))
                    if score > 0 and np.mean(alignment > threshold) >= min_fraction:
                        candidates.append((score, a, b, joint_a, joint_b))
            # One-to-one matching for this limb type.
            used_a, used_b = set(), set()
            for edge in sorted(candidates, reverse=True):
                _, a, b, _, _ = edge
                if a not in used_a and b not in used_b:
                    edges.append(edge)
                    used_a.add(a)
                    used_b.add(b)

        # Merge strongest connections first. A component may contain only one
        # candidate per joint type, preventing conflicting skeletons from merging.
        parent = list(range(len(peaks)))
        members = {}
        for joint, ids in enumerate(by_joint):
            for peak_id in ids:
                members[peak_id] = {joint: peak_id}

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for _, a, b, _, _ in sorted(edges, reverse=True):
            ra, rb = root(a), root(b)
            if ra == rb or members[ra].keys() & members[rb].keys():
                continue
            parent[rb] = ra
            members[ra].update(members.pop(rb))

        image_h, image_w = image_shape
        hm_h, hm_w = heatmaps.shape[:2]
        min_parts = int(self.get_parameter("min_person_keypoints").value)
        people = []
        for person in members.values():
            visible = sum(joint in person for joint in BODYPOSE_TO_COCO17)
            if visible < min_parts:
                continue
            coco = np.full((17, 4), -1.0, dtype=np.float32)
            for idx, joint in enumerate(BODYPOSE_TO_COCO17):
                if joint not in person:
                    continue
                x, y, confidence = peaks[person[joint]]
                coco[idx] = (
                    x * (image_w - 1) / max(hm_w - 1, 1),
                    y * (image_h - 1) / max(hm_h - 1, 1),
                    0.0, confidence,
                )
            people.append(coco)
        # Deterministic per-frame ordering; these IDs are not temporal tracks.
        people.sort(key=lambda person: float(np.mean(person[person[:, 3] >= 0, 0])))
        return people

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

    def _publish_annotated_image(self, header: Header, image: np.ndarray, people: List[np.ndarray]):
        if self.pub_annotated is None:
            return
        annotated = image.copy()
        for person_id, keypoints in enumerate(people):
            annotated = self._draw_keypoints(annotated, keypoints)
            valid = keypoints[keypoints[:, 3] >= 0]
            if len(valid):
                anchor = tuple(np.rint(valid[0, :2]).astype(int))
                cv2.putText(annotated, f"person {person_id}", anchor,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        msg.header = header
        self.pub_annotated.publish(msg)

    def _make_person_msg(
        self, header: Header, keypoints: np.ndarray, person_id: int
    ) -> TrackedKeypoints3D:
        person = TrackedKeypoints3D()
        person.header = header
        # Unique within this frame, not a persistent tracking identity.
        person.track_id = person_id

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
            heatmap = self._select_output(outputs, "heatmap")
            pafs = self._select_output(outputs, "paf")
            people = self._extract_keypoints(heatmap, pafs, image.shape[:2])
        except Exception as exc:
            self.get_logger().warn(f"BodyPoseNet inference failed: {exc}")
            return

        header = Header()
        header.stamp = msg.header.stamp
        configured_frame = str(self.get_parameter("frame_id").value)
        header.frame_id = configured_frame if configured_frame else msg.header.frame_id

        self._publish_annotated_image(header, image, people)

        if not people:
            if bool(self.get_parameter("publish_empty").value):
                self._publish(header, [])
            return

        self._publish(header, [
            self._make_person_msg(header, keypoints, person_id)
            for person_id, keypoints in enumerate(people)
        ])


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
