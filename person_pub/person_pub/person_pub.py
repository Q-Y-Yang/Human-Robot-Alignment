#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped

from pose_interfaces.msg import TrackedKeypoints3DArray

from .position import *
from .orientation import *



L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12


class PersonPublisherNode(Node):
    def __init__(self):
        super().__init__("person_pub_node")

        self.declare_parameter("input_topic", "/humans/pose3d_filtered")
        self.declare_parameter("publish_all_tracks", True)

        self.in_topic = self.get_parameter("input_topic").value
        self.publish_all = bool(self.get_parameter("publish_all_tracks").value)

        self.sub = self.create_subscription(
            TrackedKeypoints3DArray,
            self.in_topic,
            self.cb,
            qos_profile_sensor_data,
        )

        self.person_pub = self.create_publisher(
            PoseStamped,
            "/person",
            qos_profile_sensor_data,
        )

    def cb(self, msg: TrackedKeypoints3DArray):
        # ---- Empty message guard: publish nothing ----
        print("callback")
        if not hasattr(msg, "people") or msg.people is None or len(msg.people) == 0:
            print("returned")
            return

        stamp = msg.header.stamp
        frame_id = msg.header.frame_id

        published = False
        for track_id, keypoints, conf in self._iter_tracks(msg):
            joints = np.asarray([[p.x, p.y, p.z] for p in keypoints], dtype=np.float64)
            conf = np.asarray([p.conf for p in keypoints], dtype=np.float64)

            if np.mean([conf[L_SHOULDER], conf[R_SHOULDER], conf[L_HIP], conf[R_HIP]]) <= 0.5:
                print("<=0.5")
                continue
            print(">0.5")
            try:
                position_3d, _ = compute_human_position_from_joints(joints, conf)
                print("position:", position_3d)
            except Exception as e:
                self.get_logger().warn(f"compute_position failed (id={track_id}): {e}")
                continue

            try:
                x, y, z, w = map(float, compute_orientation(joints, conf, layout="coco17"))
                print("x,y,z,w:", x,y,z,w)
            except Exception as e:
                self.get_logger().warn(f"compute_orientation failed (id={track_id}): {e}")
                continue

            # normalize (safe)
            n = (x*x + y*y + z*z + w*w) ** 0.5
            if n > 1e-12:
                x, y, z, w = x / n, y / n, z / n, w / n

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = frame_id

            pose.pose.position.x = float(position_3d[0])
            pose.pose.position.y = float(position_3d[1])
            pose.pose.position.z = float(position_3d[2])

            pose.pose.orientation.x = x
            pose.pose.orientation.y = y
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w

            self.person_pub.publish(pose)
            print("person pose:", pose)

            published = True
            if not self.publish_all:
                break

        if not published:
            self.get_logger().debug("No valid person pose published for this message.")

    def _iter_tracks(self, msg):
        # Case 1: msg.keypoints (single human)
        if hasattr(msg, "keypoints"):
            yield (getattr(msg, "track_id", None), msg.keypoints, getattr(msg, "confidence", None))
            return

        # Case 2: common container fields (multi human)
        for field in ("tracks", "humans", "people", "detections", "bodies", "poses"):
            if hasattr(msg, field):
                seq = getattr(msg, field)
                if isinstance(seq, (list, tuple)) and len(seq) and hasattr(seq[0], "keypoints"):
                    for t in seq:
                        track_id = getattr(t, "track_id", getattr(t, "id", None))
                        conf = getattr(t, "confidence", getattr(t, "conf", None))
                        yield (track_id, t.keypoints, conf)
                    return

        self.get_logger().warn("Could not find keypoints in TrackedKeypoints3DArray.")


def main():
    rclpy.init()
    node = PersonPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
