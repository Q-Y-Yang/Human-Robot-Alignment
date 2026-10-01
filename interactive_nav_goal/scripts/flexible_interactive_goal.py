#!/usr/bin/env python3
"""Publish argmax(gaussian + density - cost) on a /publish_goal Trigger.

Run with Python from a sourced ROS Jazzy environment. The utility's X/Y grid
is placed at the latest person_topic pose, transformed into grid_frame
(default map). The person heading defines grid +Y (grid yaw = human yaw - pi/2).
Density rows/columns must match that grid. OccupancyGrid costs (0..100)
are scaled to [0, 1]; overlapping maps use their maximum. Global
coverage is required, while the rolling local map only contributes inside
its bounds. Unknown and lethal cells are excluded. Goals face the person.
"""

import importlib
import math
import os
from pathlib import Path
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_matrix


def utility_directory():
    """Locate assets in either the installed package or the source tree."""
    script_dir = Path(__file__).resolve().parent
    for directory in (script_dir / 'utils', script_dir.parent / 'utils'):
        if (directory / 'fformation_gaus.py').is_file():
            return directory
    raise FileNotFoundError('Cannot locate utils/fformation_gaus.py beside the node')


def load_gaussian_utility():
    """Support the existing utility's import-time relative np.load call."""
    package_dir = utility_directory().parent
    sys.path.insert(0, str(package_dir))
    previous_dir = Path.cwd()
    try:
        os.chdir(package_dir / 'utils')
        return importlib.import_module('utils.fformation_gaus')
    finally:
        os.chdir(previous_dir)
        sys.path.pop(0)


def transform_points(points, translation, rotation):
    matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
    return points @ matrix[:3, :3].T + np.array(
        [translation.x, translation.y, translation.z]
    )


def sample_costmap(msg, points):
    """Return raw costs and coverage for points in the costmap header frame."""
    info = msg.info
    if (info.width <= 0 or info.height <= 0 or info.resolution <= 0
            or not math.isfinite(info.resolution)
            or len(msg.data) != info.width * info.height):
        raise ValueError('Invalid costmap dimensions, resolution, or data length')
    origin = info.origin
    q = origin.orientation
    rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    offset = np.array([origin.position.x, origin.position.y, origin.position.z])
    coordinates = (points - offset) @ rotation
    cells = np.floor(coordinates[:, :2] / info.resolution).astype(np.int64)
    col, row = cells[:, 0], cells[:, 1]
    inside = (col >= 0) & (col < info.width) & (row >= 0) & (row < info.height)
    costs = np.full(len(points), np.nan)
    data = np.asarray(msg.data).reshape(info.height, info.width)
    costs[inside] = data[row[inside], col[inside]]
    return costs, inside


class FlexibleInteractiveGoal(Node):
    def __init__(self):
        super().__init__('flexible_interactive_goal')
        defaults = {
            'density_path': str(utility_directory() / 'interaction_positions_gaussian_filtered.npy'),
            'global_costmap_topic': '/global_costmap/costmap',
            'local_costmap_topic': '/local_costmap/costmap',
            'grid_frame': 'map',
            'person_topic': '/person',
            'distance': 0.9,
            'sigma': 0.1,
            'lethal_cost': 100,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.config = {name: self.get_parameter(name).value for name in defaults}
        for name in ('distance', 'sigma'):
            if not math.isfinite(self.config[name]):
                raise ValueError(f'{name} must be finite')
        if self.config['sigma'] <= 0 or self.config['distance'] < 0:
            raise ValueError('sigma must be positive and distance non-negative')
        if not self.config['grid_frame'] or not 1 <= self.config['lethal_cost'] <= 100:
            raise ValueError('grid_frame must be nonempty and lethal_cost in 1..100')

        # Import before spinning: the utility temporarily needs its own cwd.
        utility = load_gaussian_utility()
        x, y, gaussian, _ = utility.create_weighted_gaussian_map(
            X=utility.X, Y=utility.Y, directions=utility.directions,
            distance=self.config['distance'], sigma=self.config['sigma'],
        )
        density = np.load(self.config['density_path'], allow_pickle=False)
        if density.shape != gaussian.shape or density.ndim != 2:
            raise ValueError(f'Density shape {density.shape} must match {gaussian.shape}')
        self.base_score = np.asarray(gaussian + density, dtype=float).ravel()
        self.local_points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        self.points = None
        self.human_pose = None
        self.person_topic = self.config['person_topic']
        self.maps = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscriptions_ = [
            self.create_subscription(
                OccupancyGrid, self.config[f'{name}_costmap_topic'],
                lambda msg, key=name: self.maps.__setitem__(key, msg), qos,
            ) for name in ('global', 'local')
        ]
        self.person_sub = self.create_subscription(
            PoseStamped, self.person_topic, self.on_human_pose, qos_profile_sensor_data
        )
        self.goal_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.service = self.create_service(Trigger, '/publish_goal', self.publish_goal)
        self.get_logger().info('Ready: /publish_goal evaluates both costmaps and publishes /goal_pose')

    def on_human_pose(self, msg):
        self.human_pose = msg

    def update_grid_from_human(self):
        """Place the planar grid with +Y along the person's forward heading."""
        msg = self.human_pose
        if msg is None:
            raise ValueError(f'Waiting for a human pose on {self.person_topic}')
        if not msg.header.frame_id:
            raise ValueError('Human pose has an empty frame_id')
        pose = msg.pose
        values = [pose.position.x, pose.position.y, pose.position.z,
                  pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w]
        if not np.all(np.isfinite(values)) or np.linalg.norm(values[3:]) < 1e-12:
            raise ValueError('Human pose must be finite with a nonzero quaternion')
        if msg.header.frame_id != self.config['grid_frame']:
            transform = self.tf_buffer.lookup_transform(
                self.config['grid_frame'], msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
            )
            pose = do_transform_pose(pose, transform)
        q = pose.orientation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])
        human_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        grid_yaw = human_yaw - math.pi / 2
        c, s = math.cos(grid_yaw), math.sin(grid_yaw)
        x, y = self.local_points[:, 0], self.local_points[:, 1]
        self.person_position = (pose.position.x, pose.position.y)
        self.points = np.column_stack((
            c * x - s * y + pose.position.x,
            s * x + c * y + pose.position.y,
            np.zeros_like(x),
        ))

    def costs_in_grid(self, msg):
        if not msg.header.frame_id:
            raise ValueError('Costmap has an empty frame_id')
        points = self.points
        if msg.header.frame_id != self.config['grid_frame']:
            # Use the map timestamp to account for map/odom motion when it was made.
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, self.config['grid_frame'],
                Time.from_msg(msg.header.stamp),
            ).transform
            points = transform_points(points, transform.translation, transform.rotation)
        return sample_costmap(msg, points)

    def publish_goal(self, request, response):
        del request
        try:
            self.update_grid_from_human()
            if 'global' not in self.maps or 'local' not in self.maps:
                raise ValueError('Waiting for both global and local costmaps')
            global_cost, global_inside = self.costs_in_grid(self.maps['global'])
            local_cost, local_inside = self.costs_in_grid(self.maps['local'])
            threshold = self.config['lethal_cost']
            valid = (global_inside & (global_cost >= 0) & (global_cost < threshold)
                     & np.isfinite(self.base_score))
            valid &= ~local_inside | ((local_cost >= 0) & (local_cost < threshold))
            costs = np.maximum(global_cost, np.where(local_inside, local_cost, 0.0))
            costs = np.clip(costs / 100.0, 0.0, 1.0)
            scores = np.where(valid, self.base_score - costs, -np.inf)
            if not np.any(valid):
                raise ValueError('No finite, known, nonlethal candidate in the costmaps')
            index = int(np.argmax(scores))
            goal = PoseStamped()
            goal.header.frame_id = self.config['grid_frame']
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = float(self.points[index, 0])
            goal.pose.position.y = float(self.points[index, 1])
            goal_yaw = math.atan2(
                self.person_position[1] - goal.pose.position.y,
                self.person_position[0] - goal.pose.position.x,
            )
            goal.pose.orientation.z = math.sin(goal_yaw / 2)
            goal.pose.orientation.w = math.cos(goal_yaw / 2)
            self.goal_publisher.publish(goal)
            response.success = True
            response.message = (f'Published ({goal.pose.position.x:.3f}, '
                                f'{goal.pose.position.y:.3f}) in {goal.header.frame_id}; '
                                f'score={scores[index]:.6g}')
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().warning(f'Goal not published: {exc}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FlexibleInteractiveGoal()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
