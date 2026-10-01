#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from tf2_ros import Buffer, TransformException, TransformListener

from pose_interfaces.msg import TrackedKeypoints3DArray, TrackedKeypoints3D, PointConf


# ---------------- utils ----------------

def _time_from_header(stamp) -> Optional[Time]:
    if stamp is None:
        return None
    if getattr(stamp, "sec", 0) == 0 and getattr(stamp, "nanosec", 0) == 0:
        return None
    return Time(seconds=int(stamp.sec), nanoseconds=int(stamp.nanosec))


def _is_valid_xyz(x: float, y: float, z: float) -> bool:
    return np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


def _rotation_matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("transform contains an invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _chi2_thresh_3d(prob: float) -> float:
    # df=3 lookup table + linear interpolation (no scipy)
    table = [
        (0.90, 6.251),
        (0.95, 7.815),
        (0.975, 9.348),
        (0.99, 11.345),
        (0.995, 12.838),
    ]
    prob = float(prob)
    prob = max(min(prob, table[-1][0]), table[0][0])
    for i in range(len(table) - 1):
        p0, x0 = table[i]
        p1, x1 = table[i + 1]
        if p0 <= prob <= p1:
            t = (prob - p0) / (p1 - p0 + 1e-12)
            return x0 + t * (x1 - x0)
    return table[-1][1]


# ---------------- KF data ----------------

@dataclass
class KeypointKF:
    initialized: bool = False
    x: np.ndarray = None   # (6,) [x,y,z,vx,vy,vz]
    P: np.ndarray = None   # (6,6)
    last_conf: float = 0.0
    miss_count: int = 0
    last_update_ns: int = 0  # advances on EVERY predict step

    def __post_init__(self):
        if self.x is None:
            self.x = np.zeros((6,), dtype=np.float64)
        if self.P is None:
            self.P = np.eye(6, dtype=np.float64) * 1e3


@dataclass
class PersonTrack:
    keypoints: List[KeypointKF]
    last_seen_ns: int

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = [KeypointKF() for _ in range(17)]


class Pose3DKFFilterNode(Node):
    """
    Per-person, per-keypoint KF smoothing (classic flow):
      - Predict ALWAYS for initialized keypoints.
      - If measurement valid: gate on predicted state, then update.
      - If missing/lowconf/gated: prediction-only.
      - After max_predictions => invalidate/uninitialize.
    """

    def __init__(self):
        super().__init__("pose3d_kf_filter_node")

        # I/O
        self.declare_parameter("in_topic", "/humans/pose3d")
        self.declare_parameter("out_topic", "/humans/pose3d_filtered")
        self.declare_parameter("transform_timeout", 0.1)

        # Confidence logic
        self.declare_parameter("min_meas_conf", 0.20)   # below => ignore measurement (predict only)
        self.declare_parameter("init_conf", 0.30)       # required to initialize an untracked keypoint
        self.declare_parameter("max_predictions", 10)   # after this many predict-only steps => invalidate
        self.declare_parameter("conf_floor_when_pred", 0.0)

        # Gating (Mahalanobis)
        self.declare_parameter("gate_prob", 0.99)       # chi2 threshold for 3D
        self.declare_parameter("gate_thresh", -1.0)     # override if >= 0

        # KF noise
        self.declare_parameter("sigma_a", 0.1)          # process accel noise (m/s^2)
        self.declare_parameter("sigma_z", 0.08)         # measurement noise (m)
        self.declare_parameter("use_conf_in_R", True)   # scale R using confidence

        # Timing
        self.declare_parameter("dt_default", 1.0 / 15.0)
        self.declare_parameter("dt_min", 1e-3)
        self.declare_parameter("dt_max", 0.25)

        # Track management
        self.declare_parameter("track_timeout_s", 2.0)

        # QoS: keep latest only (prevents backlog)
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._tracks: Dict[int, PersonTrack] = {}
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            TrackedKeypoints3DArray,
            str(self.get_parameter("in_topic").value),
            self.cb,
            qos,
        )
        self.pub = self.create_publisher(
            TrackedKeypoints3DArray,
            str(self.get_parameter("out_topic").value),
            qos,
        )

        self.get_logger().info(
            "Pose3D KF filter (classic predict-then-gate/update):\n"
            f"  in_topic={self.get_parameter('in_topic').value}\n"
            f"  out_topic={self.get_parameter('out_topic').value}\n"
            "  target_frame=odom"
        )

    # ---------------- timing ----------------

    def _compute_dt(self, kf: KeypointKF, now_ns: int, msg_time_ns: Optional[int]) -> float:
        dt_default = float(self.get_parameter("dt_default").value)
        dt_min = float(self.get_parameter("dt_min").value)
        dt_max = float(self.get_parameter("dt_max").value)

        # Prefer message time if available
        t_ns = msg_time_ns if msg_time_ns is not None else now_ns

        if kf.last_update_ns == 0:
            kf.last_update_ns = t_ns
            return dt_default

        dt = (t_ns - kf.last_update_ns) * 1e-9
        kf.last_update_ns = t_ns

        if not np.isfinite(dt):
            dt = dt_default
        return float(np.clip(dt, dt_min, dt_max))

    # ---------------- KF math ----------------

    def _predict(self, kf: KeypointKF, dt: float):
        sigma_a = float(self.get_parameter("sigma_a").value)

        # State: [x,y,z,vx,vy,vz]
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Q for constant-velocity with white accel
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = (sigma_a ** 2) * np.array([[dt4 / 4.0, dt3 / 2.0],
                                      [dt3 / 2.0, dt2]], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[np.ix_([0, 3], [0, 3])] = q  # x,vx
        Q[np.ix_([1, 4], [1, 4])] = q  # y,vy
        Q[np.ix_([2, 5], [2, 5])] = q  # z,vz

        kf.x = F @ kf.x
        kf.P = F @ kf.P @ F.T + Q

    def _measurement_mats(self, conf: float) -> tuple[np.ndarray, np.ndarray]:
        sigma_z = float(self.get_parameter("sigma_z").value)
        use_conf_in_R = bool(self.get_parameter("use_conf_in_R").value)

        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        if use_conf_in_R:
            c = float(max(conf, 1e-3))
            eff = sigma_z / math.sqrt(c)  # higher conf => smaller R
            R = np.eye(3, dtype=np.float64) * (eff * eff)
        else:
            R = np.eye(3, dtype=np.float64) * (sigma_z * sigma_z)

        return H, R

    def _gate_and_update(self, kf: KeypointKF, z: np.ndarray, conf: float) -> bool:
        """
        Gate on predicted state, then update if accepted.
        Returns True if update performed.
        """
        H, R = self._measurement_mats(conf)

        y = z - (H @ kf.x)
        S = H @ kf.P @ H.T + R

        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        d2 = float(y.T @ Sinv @ y)

        gate_thresh_param = float(self.get_parameter("gate_thresh").value)
        if gate_thresh_param >= 0.0:
            gate = gate_thresh_param
        else:
            gate_prob = float(self.get_parameter("gate_prob").value)
            gate = _chi2_thresh_3d(gate_prob)

        if not np.isfinite(d2) or d2 > gate:
            return False

        # Update
        K = kf.P @ H.T @ Sinv
        kf.x = kf.x + K @ y
        I = np.eye(6, dtype=np.float64)
        kf.P = (I - K @ H) @ kf.P
        return True

    def _init_kf(self, kf: KeypointKF, z: np.ndarray, conf: float, msg_time_ns: Optional[int], now_ns: int):
        kf.x[:] = 0.0
        kf.x[0:3] = z
        kf.x[3:6] = 0.0

        # init covariance
        pos_var = 0.05 ** 2
        vel_var = 1.0 ** 2
        P = np.zeros((6, 6), dtype=np.float64)
        P[0, 0] = pos_var
        P[1, 1] = pos_var
        P[2, 2] = pos_var
        P[3, 3] = vel_var
        P[4, 4] = vel_var
        P[5, 5] = vel_var
        kf.P = P

        kf.initialized = True
        kf.last_conf = float(conf)
        kf.miss_count = 0

        kf.last_update_ns = (msg_time_ns if msg_time_ns is not None else now_ns)

    # ---------------- callback ----------------

    def cb(self, msg: TrackedKeypoints3DArray):
        now_ns = self.get_clock().now().nanoseconds
        msg_time = _time_from_header(msg.header.stamp) if hasattr(msg, "header") else None
        msg_time_ns = msg_time.nanoseconds if msg_time is not None else None

        source_frame = str(msg.header.frame_id).strip()
        if not source_frame:
            self.get_logger().warn("Dropping pose array with an empty frame_id")
            return

        rotation_to_odom = np.eye(3, dtype=np.float64)
        translation_to_odom = np.zeros(3, dtype=np.float64)
        if source_frame != "odom":
            transform_time = msg_time if msg_time is not None else Time()
            try:
                transform = self.tf_buffer.lookup_transform(
                    "odom",
                    source_frame,
                    transform_time,
                    timeout=Duration(
                        seconds=float(self.get_parameter("transform_timeout").value)
                    ),
                )
                q = transform.transform.rotation
                rotation_to_odom = _rotation_matrix_from_quaternion(
                    q.x, q.y, q.z, q.w
                )
                t = transform.transform.translation
                translation_to_odom = np.array([t.x, t.y, t.z], dtype=np.float64)
            except (TransformException, ValueError) as exc:
                self.get_logger().warn(
                    f"Could not transform pose array from {source_frame} to odom: {exc}",
                    throttle_duration_sec=2.0,
                )
                return

        min_meas_conf = float(self.get_parameter("min_meas_conf").value)
        init_conf = float(self.get_parameter("init_conf").value)
        max_predictions = int(self.get_parameter("max_predictions").value)
        conf_floor_pred = float(self.get_parameter("conf_floor_when_pred").value)

        track_timeout_s = float(self.get_parameter("track_timeout_s").value)
        track_timeout_ns = int(track_timeout_s * 1e9)

        # Purge stale person tracks
        stale = [tid for tid, tr in self._tracks.items() if (now_ns - tr.last_seen_ns) > track_timeout_ns]
        for tid in stale:
            del self._tracks[tid]

        out = TrackedKeypoints3DArray()
        out.header = msg.header
        out.header.frame_id = "odom"
        out_people: List[TrackedKeypoints3D] = []

        for person in msg.people:
            tid = int(person.track_id)

            if tid not in self._tracks:
                self._tracks[tid] = PersonTrack(keypoints=[KeypointKF() for _ in range(17)], last_seen_ns=now_ns)
            tr = self._tracks[tid]
            tr.last_seen_ns = now_ns

            out_person = TrackedKeypoints3D()
            out_person.header = out.header
            out_person.track_id = tid

            in_kps = list(person.keypoints)
            if len(in_kps) < 17:
                for _ in range(17 - len(in_kps)):
                    p = PointConf()
                    p.x = p.y = p.z = -1.0
                    p.conf = 0.0
                    in_kps.append(p)
            elif len(in_kps) > 17:
                in_kps = in_kps[:17]

            out_kps: List[PointConf] = []

            for j in range(17):
                kf = tr.keypoints[j]
                kp = in_kps[j]

                meas_conf = float(getattr(kp, "conf", 0.0))
                mx, my, mz = float(kp.x), float(kp.y), float(kp.z)

                meas_valid = _is_valid_xyz(mx, my, mz) and (meas_conf >= min_meas_conf)
                if meas_valid and source_frame != "odom":
                    measurement = rotation_to_odom @ np.array(
                        [mx, my, mz], dtype=np.float64
                    ) + translation_to_odom
                    mx, my, mz = map(float, measurement)

                # ---- init ----
                if not kf.initialized:
                    if meas_valid and (meas_conf >= init_conf):
                        z = np.array([mx, my, mz], dtype=np.float64)
                        self._init_kf(kf, z, meas_conf, msg_time_ns, now_ns)
                        ox, oy, oz = float(kf.x[0]), float(kf.x[1]), float(kf.x[2])
                        oconf = float(meas_conf)
                    else:
                        ox, oy, oz = -1.0, -1.0, -1.0
                        oconf = 0.0

                    pc = PointConf()
                    pc.x, pc.y, pc.z, pc.conf = ox, oy, oz, oconf
                    out_kps.append(pc)
                    continue

                # ---- classic: predict ALWAYS, then maybe update ----
                dt = self._compute_dt(kf, now_ns=now_ns, msg_time_ns=msg_time_ns)
                self._predict(kf, dt)

                accepted = False
                if meas_valid:
                    z = np.array([mx, my, mz], dtype=np.float64)
                    accepted = self._gate_and_update(kf, z, meas_conf)

                if accepted:
                    kf.miss_count = 0
                    kf.last_conf = float(meas_conf)
                    ox, oy, oz = float(kf.x[0]), float(kf.x[1]), float(kf.x[2])
                    oconf = float(meas_conf)
                else:
                    kf.miss_count += 1
                    if kf.miss_count > max_predictions:
                        # lost: invalidate and allow re-init
                        kf.initialized = False
                        kf.last_update_ns = 0
                        ox, oy, oz = -1.0, -1.0, -1.0
                        oconf = 0.0
                    else:
                        ox, oy, oz = float(kf.x[0]), float(kf.x[1]), float(kf.x[2])
                        oconf = float(max(kf.last_conf, conf_floor_pred))

                pc = PointConf()
                pc.x, pc.y, pc.z, pc.conf = ox, oy, oz, oconf
                out_kps.append(pc)

            out_person.keypoints = out_kps
            out_people.append(out_person)

        out.people = out_people
        self.pub.publish(out)


def main():
    rclpy.init()
    node = Pose3DKFFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
