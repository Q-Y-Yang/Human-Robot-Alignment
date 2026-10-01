#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable
# from tf_transformations import quaternion_from_matrix  # ROS2: pip install tf-transformations
from person_pub.utils import _normalize, quat_from_forward_up_tf
import rclpy

# ----------------------------- Joint Layouts -----------------------------

@dataclass(frozen=True)
class JointLayout:
    l_hip: int
    r_hip: int
    spine: Optional[int]     # thorax or spine mid (None if not present)
    l_shoulder: int
    r_shoulder: int
    l_ear: Optional[int]
    r_ear: Optional[int]

# H36M_LAYOUT = JointLayout(
    # H36M 17J (adapt if your indices differ)
    # 0: Hip(root), 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
    # 7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head, 11: LShoulder, 12: LElbow,
    # 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
#     l_hip=4, r_hip=1, spine=8, l_shoulder=11, r_shoulder=14
# )

COCO17_LAYOUT = JointLayout(
    # 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar, 5: LShoulder, 6: RShoulder,
    # 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist, 11: LHip, 12: RHip,
    # 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
    l_hip=11, r_hip=12, spine=None, l_shoulder=5, r_shoulder=6, l_ear=3, r_ear=4
)

_LAYOUTS: Dict[str, JointLayout] = {
    # "h36m": H36M_LAYOUT,
    "coco17": COCO17_LAYOUT,
}

# COCO-17 indices
L_IDS = np.array([1, 3, 5, 7, 9, 11, 13, 15], dtype=int)   # LEye, LEar, LShoulder, LElbow, LWrist, LHip, LKnee, LAnkle
R_IDS = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=int)  # REye, REar, RShoulder, RElbow, RWrist, RHip, RKnee, RAnkle

_logger = rclpy.logging.get_logger(__name__)  

def pair_distance(joints_3d: np.ndarray, i: int, j: int) -> float:
    """
    Returns Euclidean distance between joints_3d[i] and joints_3d[j].
    joints_3d shape: (17, 3)
    """

    joints_3d = np.asarray(joints_3d, dtype=float)
    if joints_3d.shape != (17, 3):
        raise ValueError(f"Expected joints_3d shape (17,3), got {joints_3d.shape}")
    
    a = joints_3d[i]
    b = joints_3d[j]
    return float(np.linalg.norm(a - b))

def compute_visibility_scores(
    joints_3d: np.ndarray,
    layout: Dict,
    joints_conf: Optional[np.ndarray] = None,
    conf_diff: float = 0.15,
    left_ids: Iterable[int] = L_IDS,
    right_ids: Iterable[int] = R_IDS,
    weights = None, #Optional[Dict[int, float]] = DEFAULT_WEIGHTS,
    threshold_d:float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Returns per-side visibility:
      score: weighted mean confidence over VALID joints on that side
      valid_frac: fraction of side joints considered valid
      valid_count: number of valid joints on that side
      mean_conf_all: mean conf over all joints on that side (includes invalids)
      mean_conf_valid: mean conf over valid joints on that side
    """
    j = np.asarray(joints_3d, dtype=float)
    if j.shape != (17, 3):
        raise ValueError(f"Expected joints_3d shape (17,3), got {j.shape}")

    if joints_conf is None:
        c = np.ones((17,), dtype=float)
    else:
        c = np.asarray(joints_conf, dtype=float).reshape(-1)
        if c.shape[0] != 17:
            raise ValueError(f"Expected joints_conf length 17, got {c.shape[0]}")
        c = np.clip(c, 0.0, 1.0)

    valid = np.isfinite(j).all(axis=1)

    left_ids = np.array(list(left_ids), dtype=int)
    right_ids = np.array(list(right_ids), dtype=int)

    def side_stats(ids: np.ndarray) -> Dict[str, float]:
        ids = np.asarray(ids, dtype=int)
        if ids.size == 0:
            return {"score": 0.0, "valid_ids": 0.0}

        w = np.ones_like(ids, dtype=float)
        if weights is not None:
            w = np.array([float(weights.get(int(i), 1.0)) for i in ids], dtype=float)

        valid_ids = ids[valid[ids]]
        if valid_ids.size == 0:
            mean_all = float(np.mean(c[ids])) if ids.size else 0.0
            return {
                "score": 0.0,
                "valid_ids": 0.0
            }

        # Weighted score over valid joints
        w_valid = np.array([float(weights.get(int(i), 1.0)) for i in valid_ids], dtype=float) if weights else np.ones_like(valid_ids, dtype=float)
        score = float(np.sum(c[valid_ids] * w_valid) / (np.sum(w_valid) + 1e-12))

        return {
            "score": score,
            "valid_ids": valid_ids
        }

    out = {
        "left": side_stats(left_ids),
        "right": side_stats(right_ids),
    }

    d_shoulders = pair_distance(joints_3d, layout.l_shoulder , layout.r_shoulder)   # LShoulder-RShoulder
    d_hips      = pair_distance(joints_3d, layout.l_hip, layout.r_hip) # LHip-RHip
    not_side_view = (
        (d_shoulders is not None and 1.0 > d_shoulders > threshold_d) or
        (d_hips      is not None and 1.0 > d_hips      > threshold_d)
    )

    # Convenience: who is more visible?
    diff = out["right"]["score"] - out["left"]["score"]
    out["summary"] = {
        "right_minus_left": float(diff),
        "dominant_side": "balanced" if not_side_view
        else ("right" if diff > conf_diff else ("left" if diff < conf_diff else "balanced")),
    }
    return out


def plane_normal(ear, shoulder, hip, right: bool, eps=1e-12):
    """
    ear, shoulder, hip: array-like, shape (3,)
    return: unit normal vector, shape (3,)
    """
    ear = np.asarray(ear, dtype=float)
    ear[-1] = ear[-1] + 0.05    #increase the z value of ear by 5 cm in the camera frame
    shoulder = np.asarray(shoulder, dtype=float)
    hip = np.asarray(hip, dtype=float)

    ear[-1] = ear[-1] + 0.05

    v1 = ear - shoulder
    v2 = hip - shoulder # downwards

    if right is True:
        n = np.cross(v2, v1)  # normal (not normalized), right-hand rule from side view
    else:
        n = np.cross(v1, v2)    # if the left ones are visible.

    norm = np.linalg.norm(n)

    if norm < eps:
        _logger.error("Three points are almost collinear or coincident, impossible to stably define the plane normal vector.")
        return None

    return quat_from_forward_up_tf(forward=n, up=-v2)   #q_xyzw



def compute_torso_frame(
    joints_3d: np.ndarray,
    joints_conf: Optional[np.ndarray],
    L: Dict,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Estimate torso orientation from a single 3D pose (N x 3), returning:
      - quat_wxyz: rotation (wxyz) whose columns correspond to [right, up, forward]
      - forward, right, up: unit vectors in WORLD coords
      - confidence: [0..1] reliability score

    Frame construction (right-handed):
      up      = normalize(spine_or_shoulder_mid - pelvis_mid)
      forward = normalize(cross(shoulder_axis, up))             # shoulder_axis = RShoulder - LShoulder
      right   = normalize(cross(up, forward))
      (Gram-Schmidt clean-up for orthogonality)
    """


    # Pelvis midpoint
    lhip = joints_3d[L.l_hip]
    rhip = joints_3d[L.r_hip]
    pelvis = 0.5 * (lhip + rhip)

    # Spine/Thorax or shoulder midpoint
    if L.spine is not None:
        spine = joints_3d[L.spine]
    else:
        lsh = joints_3d[L.l_shoulder]
        rsh = joints_3d[L.r_shoulder]
        spine = 0.5 * (lsh + rsh)

    # Unit up direction: pelvis -> spine
    up = _normalize(spine - pelvis, eps)

    # Shoulder axis = anatomical right (R - L)
    shoulder_axis = _normalize(joints_3d[L.r_shoulder] - joints_3d[L.l_shoulder], eps)

    hip_axis = _normalize(joints_3d[L.r_hip] - joints_3d[L.l_hip], eps)

        # Angle between vectors
    cosang = float(np.clip(np.dot(shoulder_axis, hip_axis), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    angle_thresh_deg = 10

    # Pair confidence sums
    s_conf = float(joints_conf[L.l_shoulder] + joints_conf[L.r_shoulder])
    h_conf = float(joints_conf[L.l_hip] + joints_conf[L.r_hip])

    if angle_deg < angle_thresh_deg:
        denom = s_conf + h_conf
        weight = (s_conf / denom) if denom > eps else 0.5  # fallback if all confs ~0
        right = _normalize(weight * shoulder_axis + (1.0 - weight) * hip_axis, eps)
    else:
        # pick higher-confidence axis
        right = shoulder_axis if s_conf >= h_conf else hip_axis

    # Pin 'right' to shoulder axis
    # right = _normalize(shoulder_axis, eps)

    # Define forward as up × right (front of chest)
    forward = _normalize(np.cross(up, right), eps)

    # Re-orthogonalize up so it's perpendicular to right
    up = _normalize(np.cross(right, forward), eps) #(Gram-Schmidt)

    # Rotation matrix with columns [right, up, forward]
    # R_np = np.stack([right, up, forward], axis=1).astype(np.float32)  # (3,3) #right = X, up = Y, forward = Z.

    # Confidence: combine joint confidences (if provided) and geometric stability
    conf_j = 1.0
    if joints_conf is not None:
        needed = [L.l_hip, L.r_hip, L.l_shoulder, L.r_shoulder]
        if L.spine is not None:
            needed.append(L.spine)
        conf_j = float(np.clip(np.mean(joints_conf[needed]), 0.0, 1.0)) #averaging the conf to (0,1)

    ortho = (np.dot(right, up) ** 2 + np.dot(up, forward) ** 2 + np.dot(forward, right) ** 2)
    lengths = float(np.clip((np.linalg.norm(forward) + np.linalg.norm(up) + np.linalg.norm(right)) / 3.0, 0.0, 1.0))
    conf_g = float(np.clip(1.0 - ortho, 0.0, 1.0)) * lengths
    confidence = float(np.clip(0.5 * conf_j + 0.5 * conf_g, 0.0, 1.0))

    q = quat_from_forward_up_tf(forward=forward, up=up)

    return q#, forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32), confidence

def compute_orientation(
        joints_3d: np.ndarray,
        joints_conf: Optional[np.ndarray],
        conf_diff: float = 0.15,
        layout: str = "coco",):

    if layout not in _LAYOUTS:
        raise ValueError(f"Unknown layout '{layout}'. Known: {list(_LAYOUTS)}")
    L = _LAYOUTS[layout]

    if joints_3d.ndim != 2 or joints_3d.shape[1] != 3:
        raise ValueError("joints_3d must be (N x 3)")
    
    vis = compute_visibility_scores(
            joints_3d=joints_3d,
            layout=L,
            joints_conf=joints_conf,
            conf_diff=conf_diff,
            )
    
    if vis["summary"]["dominant_side"] == "left":
        q = plane_normal(ear=joints_3d[L.l_ear], shoulder=joints_3d[L.l_shoulder], hip = joints_3d[L.l_hip], right=False)
    elif vis["summary"]["dominant_side"] == "right":
        q = plane_normal(ear=joints_3d[L.r_ear], shoulder=joints_3d[L.r_shoulder], hip = joints_3d[L.r_hip], right=True)
    else:
        q = compute_torso_frame(joints_3d=joints_3d, joints_conf=joints_conf, L=L)

    if q is None:
        _logger.error("orientation failed; skipping this frame/update.")
        return 
    return q
