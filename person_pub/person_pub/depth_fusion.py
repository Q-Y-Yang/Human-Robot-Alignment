#!/usr/bin/env python3
import struct
from typing import Optional, Tuple, List

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from message_filters import ApproximateTimeSynchronizer, Subscriber

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge

from pose_interfaces.msg import TrackedKeypoints3D, TrackedKeypoints3DArray, PointConf

# COCO-17 keypoints
KP_NOSE = 0
KP_L_EYE = 1
KP_R_EYE = 2
KP_L_EAR = 3
KP_R_EAR = 4
KP_L_SHO = 5
KP_R_SHO = 6
KP_L_ELB = 7
KP_R_ELB = 8
KP_L_WRI = 9
KP_R_WRI = 10
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNE = 13
KP_R_KNE = 14
KP_L_ANK = 15
KP_R_ANK = 16

def median_depth(depth_m: np.ndarray, u: int, v: int, win: int = 5) -> float:
    """Median depth (meters) in a win×win patch around (u,v)."""
    h, w = depth_m.shape[:2]
    r = win // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_m[v0:v1, u0:u1].reshape(-1)
    patch = patch[np.isfinite(patch)]
    patch = patch[patch > 0.0]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    """Pixel (u,v) with depth z -> camera frame XYZ."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


class DepthFusionPose3DEstimator(Node):
    def __init__(self):
        super().__init__("depth_fusion_pose_3d_estimator")

        # --- Topics (Temi defaults) ---
        self.declare_parameter("keypoints_topic", "/humans/bodyposenet/keypoints")
        self.declare_parameter("depth_info_topic", "/temi/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("depth_image_topic", "/temi/camera/aligned_depth_to_color/image_raw")

        # Depth uses /compressedDepth when enabled.
        self.declare_parameter("compressed", True)

        # --- Sync ---
        self.declare_parameter("sync_queue", 10)
        self.declare_parameter("sync_slop", 0.10)

        # --- Keypoint filtering ---
        self.declare_parameter("min_kpt_conf", 0.20)
        self.declare_parameter("corner_eps", 1e-4)  # discard (x,y) near (0,0)
        self.declare_parameter("depth_patch_win", 5)

        # --- Output ---
        self.declare_parameter("pose3d_topic", "/humans/pose3d")

        self.bridge = CvBridge()

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # Publisher
        self.pub_pose3d = self.create_publisher(
            TrackedKeypoints3DArray,
            str(self.get_parameter("pose3d_topic").value),
            10,
        )

        # Subscribers + sync
        keypoints_topic = str(self.get_parameter("keypoints_topic").value)
        depth_info_topic = str(self.get_parameter("depth_info_topic").value)
        depth_image_topic = str(self.get_parameter("depth_image_topic").value)
        compressed = bool(self.get_parameter("compressed").value)

        depth_msg_t = CompressedImage if compressed else Image
        depth_sub_topic = f"{depth_image_topic}/compressedDepth" if compressed else depth_image_topic

        self.sub_keypoints = Subscriber(self, TrackedKeypoints3DArray, keypoints_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = Subscriber(self, depth_msg_t, depth_sub_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth_info = Subscriber(self, CameraInfo, depth_info_topic, qos_profile=qos_profile_sensor_data)

        q = int(self.get_parameter("sync_queue").value)
        slop = float(self.get_parameter("sync_slop").value)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_keypoints, self.sub_depth, self.sub_depth_info],
            queue_size=q,
            slop=slop,
            allow_headerless=False,
        )
        self.ts.registerCallback(self.cb)

        self.get_logger().info(
            "Subscribed:\n"
            f" KPTS:  {keypoints_topic}\n"
            f" DEPTH: {depth_sub_topic}\n"
            f" DEPI:  {depth_info_topic}\n"
            f" pose3d_topic={self.get_parameter('pose3d_topic').value}"
        )

    # ---------------- decoding & intrinsics ----------------
    def _set_intrinsics(self, cam_info: CameraInfo):
        if self.fx is not None:
            return
        K = cam_info.k
        self.fx, self.fy, self.cx, self.cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        self.get_logger().info(f"Intrinsics fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def _decode_depth(self, msg) -> np.ndarray:
        # Uncompressed depth
        if isinstance(msg, Image):
            enc = msg.encoding
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if enc == "16UC1":
                return d.astype(np.float32) * 0.001
            if enc == "32FC1":
                return d.astype(np.float32)
            return d.astype(np.float32)

        # CompressedDepth payload
        raw = bytes(msg.data)
        if ";" in msg.format:
            enc = msg.format.split(";", 1)[0].strip()
            rest = msg.format.split(";", 1)[1].strip().lower()
        else:
            enc = msg.format.strip()
            rest = ""

        png_magic = b"\x89PNG\r\n\x1a\n"
        idx = raw.find(png_magic)
        if idx < 0:
            raise RuntimeError(f"Could not find PNG header in depth payload (format='{msg.format}')")
        png = raw[idx:]
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode returned None for depth payload")

        if "compresseddepth" in rest:
            if enc == "16UC1":
                return img.astype(np.float32) * 0.001
            if enc == "32FC1":
                if len(raw) < 12:
                    raise RuntimeError("compressedDepth payload too small for 32FC1 header")
                _, depthQuantA, depthQuantB = struct.unpack("<iff", raw[:12])
                inv = img.astype(np.float32)
                out = np.zeros_like(inv, dtype=np.float32)
                mask = inv > 0
                out[mask] = depthQuantA / (inv[mask] - depthQuantB)
                return out
            raise RuntimeError(f"Unsupported compressedDepth encoding: {enc}")

        if enc == "16UC1":
            return img.astype(np.float32) * 0.001
        return img.astype(np.float32)

    # ---------------- filters ----------------
    def _kpt_valid(
        self,
        x: float,
        y: float,
        c: Optional[float],
        min_conf: float,
        corner_eps: float,
        width: int,
        height: int,
    ) -> bool:
        if (not np.isfinite(x)) or (not np.isfinite(y)):
            return False
        if x < 0.0 or x >= float(width) or y < 0.0 or y >= float(height):
            return False
        if x <= corner_eps and y <= corner_eps:
            return False
        if c is not None:
            if not np.isfinite(c):
                return False
            if c < min_conf:
                return False
        return True

    # ---------------- publishing ----------------
    def _make_tracked_msg(self, header: Header, track_id: int, xyz_17x4: np.ndarray) -> TrackedKeypoints3D:
        msg = TrackedKeypoints3D()
        msg.header = header
        msg.track_id = int(track_id)

        pts: List[PointConf] = []
        for j in range(17):
            p = PointConf()
            p.x = float(xyz_17x4[j, 0])
            p.y = float(xyz_17x4[j, 1])
            p.z = float(xyz_17x4[j, 2])
            p.conf = float(xyz_17x4[j, 3])
            pts.append(p)

        # fixed arrays require exact length; we provide exactly 17
        msg.keypoints = pts
        return msg

    def _publish_pose_array(self, header: Header, elems: List[TrackedKeypoints3D]):
        arr = TrackedKeypoints3DArray()
        arr.header = header
        arr.people = elems
        self.pub_pose3d.publish(arr)

    # ---------------- callback ----------------
    def cb(self, keypoints_msg: TrackedKeypoints3DArray, depth_msg, depth_info: CameraInfo):
        depth_m = self._decode_depth(depth_msg)
        if depth_m is None:
            return

        self._set_intrinsics(depth_info)
        if self.fx is None:
            return

        header = Header()
        header.frame_id = depth_info.header.frame_id if depth_info.header.frame_id else "camera_frame"
        header.stamp = depth_info.header.stamp

        min_kpt_conf = float(self.get_parameter("min_kpt_conf").value)
        corner_eps = float(self.get_parameter("corner_eps").value)
        win = int(self.get_parameter("depth_patch_win").value)

        h_d, w_d = depth_m.shape[:2]
        elems_list: List[TrackedKeypoints3D] = []

        for i, person in enumerate(keypoints_msg.people):
            tid = int(person.track_id) if person.track_id >= 0 else int(i)
            xyzc = np.full((17, 4), -1.0, dtype=np.float32)

            for j in range(17):
                if j >= len(person.keypoints):
                    continue

                kp = person.keypoints[j]
                x = float(kp.x)
                y = float(kp.y)
                c = float(kp.conf)

                if not self._kpt_valid(x, y, c, min_kpt_conf, corner_eps, w_d, h_d):
                    continue

                u = min(max(int(round(x)), 0), w_d - 1)
                v = min(max(int(round(y)), 0), h_d - 1)

                z = median_depth(depth_m, u, v, win=win)
                if z <= 0.0 or not np.isfinite(z):
                    continue

                X, Y, Z = backproject(u, v, z, self.fx, self.fy, self.cx, self.cy)
                if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                    continue

                xyzc[j, 0] = float(X)
                xyzc[j, 1] = float(Y)
                xyzc[j, 2] = float(Z)
                xyzc[j, 3] = c

            elems_list.append(self._make_tracked_msg(header, tid, xyzc))

        self._publish_pose_array(header, elems_list)


def main():
    rclpy.init()
    node = DepthFusionPose3DEstimator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()