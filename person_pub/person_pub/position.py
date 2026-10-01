#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np


# COCO-17 keypoint order (common):
# 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
# 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
# 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
# 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
@dataclass(frozen=True)
class Coco17:
    l_shoulder: int = 5
    r_shoulder: int = 6
    l_hip: int = 11
    r_hip: int = 12


def compute_human_position_from_joints(
    joints_3d: np.ndarray,
    joint_conf: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    *,
    # invalid_sentinel: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
    # return_center_3d: bool = False,
    K: Coco17 = Coco17(),
) -> Tuple[float, float] | Tuple[float, float, np.ndarray, Dict]:
    """
    Compute human (x, y) position in the camera optical frame from COCO-17 joints_3d.

    Strategy:
      # 1) Drop invalid points == [-1,-1,-1] AND also drop NaN/Inf.
      2) Keep only joint_conf >= threshold (if provided).
      3) Prefer torso:
         - hips (midpoint of L/R if both, else the valid one)
         - else shoulders (midpoint if both, else the valid one)
         - else fallback: median of ALL valid points (per-dim median)

    Args:
        joints_3d: (17,3) array in camera optical frame.
        joint_conf: optional (17,) confidence array in [0,1] (or comparable scale).
        conf_threshold: minimum confidence to keep a joint.
        invalid_sentinel: sentinel used for invalid points.
        return_center_3d: if True, also returns chosen 3D center and some debug info.
        K: COCO index mapping.

    Returns:
        (x, y) in optical frame, or (x, y, center_3d, debug_info) if return_center_3d=True.
        If no valid points exist, returns (nan, nan) (and nan center_3d if requested).
    """
    joints_3d = np.asarray(joints_3d, dtype=float)
    if joints_3d.shape != (17, 3):
        raise ValueError(f"Expected joints_3d shape (17,3), got {joints_3d.shape}")

    # --- 1) validity mask: not sentinel AND infinite ---
    # sentinel = np.asarray(invalid_sentinel, dtype=float).reshape(1, 3)
    # is_sentinel = np.all(joints_3d == sentinel, axis=1)
    # is_finite = np.isfinite(joints_3d).all(axis=1)
    # valid = (~is_sentinel) & is_finite
    valid = np.isfinite(joints_3d).all(axis=1)

    # --- 2) confidence mask ---
    if joint_conf is not None:
        joint_conf = np.asarray(joint_conf, dtype=float).reshape(-1)
        if joint_conf.shape != (17,):
            raise ValueError(f"Expected joint_conf shape (17,), got {joint_conf.shape}")
        valid &= (joint_conf >= conf_threshold)

    def pick_mid_or_single(i: int, j: int) -> Tuple[Optional[np.ndarray], str]:
        """Return midpoint if both valid, else single if one valid, else None."""
        vi, vj = valid[i], valid[j]
        if vi and vj:
            return 0.5 * (joints_3d[i] + joints_3d[j]), "midpoint"
        if vi:
            return joints_3d[i], "single_left"
        if vj:
            return joints_3d[j], "single_right"
        return None, "none"

    source: Literal["hips", "shoulders", "median", "none"] = "none"
    mode: str = "none"

    # --- 3) torso priority: hips -> shoulders -> median fallback ---
    center_3d, mode = pick_mid_or_single(K.l_hip, K.r_hip)
    if center_3d is not None:
        source = "hips"
    else:
        center_3d, mode = pick_mid_or_single(K.l_shoulder, K.r_shoulder)
        if center_3d is not None:
            source = "shoulders"
        else:
            pts = joints_3d[valid]
            if pts.shape[0] > 0:
                center_3d = np.median(pts, axis=0)
                source = "median"
                mode = f"median_n={pts.shape[0]}"
            else:
                center_3d = np.array([np.nan, np.nan, np.nan], dtype=float)
                source = "none"
                mode = "no_valid_points"

    # x, y = float(center_3d[0]), float(center_3d[1])

    # if not return_center_3d:
    #     return x, y

    debug = {
        "source": source,
        "mode": mode,
        "num_valid": int(valid.sum()),
        "conf_threshold": float(conf_threshold),
        # "invalid_sentinel": tuple(map(float, invalid_sentinel)),
    }
    return center_3d, debug


# ------------------------ Example ------------------------

# c3d, info = compute_human_xy_from_joints(
#         joints, conf, conf_threshold=0.3, return_center_3d=True
#     )
#     print("x,y =", x, y, "center_3d =", c3d, "info =", info)
