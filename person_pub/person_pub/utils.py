#!/usr/bin/env python3
import numpy as np
from tf_transformations import quaternion_from_matrix
import rclpy


_logger = rclpy.logging.get_logger(__name__)  

def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        _logger.error("Zero/near-zero vector, cannot normalize.")
        return
    return v / n

def quat_from_forward_up_tf(forward, up, eps=1e-12, enforce_w_positive=True):
    """
    Build quaternion from forward and up using ROS REP-103 axes:
      x = forward, y = left, z = up
    Returns (x, y, z, w).
    """
    x = _normalize(forward, eps)   # forward
    z0 = _normalize(up, eps)       # approximate up

    if x is None or z0 is None:
        return

    # Make z orthogonal to x (Gram-Schmidt)
    z = z0 - np.dot(z0, x) * x
    z = _normalize(z, eps)

    # y = z × x (left)
    y = _normalize(np.cross(z, x), eps)

    # Recompute z to ensure orthonormal frame
    z = np.cross(x, y)

    # Rotation matrix with columns = body axes in world coords
    R = np.column_stack((x, y, z))

    # Convert to 4x4 for tf_transformations
    T = np.eye(4, dtype=float)
    T[:3, :3] = R

    q = np.asarray(quaternion_from_matrix(T), dtype=float)  # (x,y,z,w)

    # Optional: stabilize sign over time (q and -q are same rotation)
    if enforce_w_positive and q[3] < 0:
        q = -q

    return q
