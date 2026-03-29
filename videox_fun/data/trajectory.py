# videox_fun/data/trajectory.py
"""
SE(3) trajectory utilities for AniML walkthrough generation.

All poses are c2w (camera-to-world) [4, 4] float32 numpy arrays.
Coordinate convention: right-handed, Y-up (VGGT / GLD convention).

Produces:
  keyframe_poses: [n_keyframes, 4, 4]  intermediate poses for GLD novel views
  video_poses:    [n_video_frames, 4, 4]  per-frame poses for Plücker embedding
"""

from __future__ import annotations
import numpy as np


# ── SO(3) Lie algebra ──────────────────────────────────────────────────────────

def so3_hat(omega: np.ndarray) -> np.ndarray:
    """Axis-angle vector [3,] → skew-symmetric matrix [3,3]."""
    wx, wy, wz = omega
    return np.array([
        [  0, -wz,  wy],
        [ wz,   0, -wx],
        [-wy,  wx,   0],
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """
    Rodrigues formula: axis-angle vector [3,] → rotation matrix [3,3].
    Handles angle ≈ 0 (returns identity).
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    K = so3_hat(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrix [3,3] → axis-angle vector [3,].
    Handles angle ≈ 0 and angle ≈ π.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if abs(theta - np.pi) < 1e-6:
        # Degenerate case: angle ≈ π
        # Extract axis from diagonal of (R + I) / 2
        diag = np.diag(R)
        i = int(np.argmax(diag))
        axis = R[:, i] + np.eye(3)[i]
        axis = axis / np.linalg.norm(axis)
        return axis * theta
    return (theta / (2.0 * np.sin(theta))) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)


# ── SE(3) Lie algebra ──────────────────────────────────────────────────────────

def se3_log(T: np.ndarray) -> np.ndarray:
    """
    SE(3) log map. T [4,4] → twist xi [6,] = [omega (3), v (3)].
    Uses the coupled (omega, v) formulation.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    theta = np.linalg.norm(omega)

    if theta < 1e-8:
        # Pure translation
        v = t
    else:
        K = so3_hat(omega / theta)
        A_inv = (
            np.eye(3)
            - 0.5 * theta * K
            + (1.0 - theta / (2.0 * np.tan(theta / 2.0))) * (K @ K)
        )
        v = A_inv @ t

    return np.concatenate([omega, v])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    SE(3) exp map. Twist xi [6,] = [omega (3), v (3)] → T [4,4].
    """
    omega = xi[:3]
    v     = xi[3:]
    theta = np.linalg.norm(omega)
    R = so3_exp(omega)

    if theta < 1e-8:
        t = v
    else:
        K = so3_hat(omega / theta)
        A = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / theta) * K
            + ((theta - np.sin(theta)) / theta) * (K @ K)
        )
        t = A @ v

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T


# ── Interpolation ──────────────────────────────────────────────────────────────

def interpolate_se3(
    T0: np.ndarray,
    T1: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    """
    SE(3) geodesic interpolation between T0 and T1.

    Args:
        T0, T1:  [4, 4] c2w matrices
        alphas:  [K,] values in [0, 1]

    Returns:
        poses [K, 4, 4]
    """
    T_rel = np.linalg.inv(T0) @ T1
    xi    = se3_log(T_rel)
    poses = np.stack([
        T0 @ se3_exp(float(a) * xi).astype(np.float64)
        for a in alphas
    ]).astype(np.float32)
    return poses


# ── Main trajectory builder ────────────────────────────────────────────────────

def build_walkthrough_trajectory(
    pose_first: np.ndarray,
    pose_last: np.ndarray,
    n_keyframes: int = 4,
    n_video_frames: int = 81,
    normalize_path: bool = True,
) -> dict:
    """
    Build walkthrough trajectory from first-frame pose to last-frame pose.

    Args:
        pose_first:       [4,4] float32 c2w of first frame (from GLD source poses[0])
        pose_last:        [4,4] float32 c2w of last frame  (from GLD source poses[-1])
        n_keyframes:      number of INTERMEDIATE keyframe poses for GLD novel view
                          synthesis. Endpoints are excluded because they are real
                          source images. Default 4 → alphas = [0.2, 0.4, 0.6, 0.8].
        n_video_frames:   total frames in the output video. Must be 4n+1 (Wan VAE
                          temporal stride requirement). Default 81.
        normalize_path:   if True, normalise all translation magnitudes so the total
                          Euclidean path length (first→last camera origin) = 1.0.
                          Required to match the translation scale distribution that
                          VideoX-Fun's SimpleAdapter was trained on.

    Returns:
        dict:
          'keyframe_poses'  : np.ndarray [n_keyframes, 4, 4]  float32
          'video_poses'     : np.ndarray [n_video_frames, 4, 4]  float32
          'keyframe_alphas' : np.ndarray [n_keyframes,]  float32
          'path_length_m'   : float  (pre-normalisation, approximate metres)
    """
    assert (n_video_frames - 1) % 4 == 0, \
        f"n_video_frames must be 4n+1, got {n_video_frames}"

    T0 = pose_first.astype(np.float64)
    T1 = pose_last.astype(np.float64)

    # Path length before normalisation
    path_length_m = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))

    # Normalise translations so path_length = 1.0
    if normalize_path and path_length_m > 1e-6:
        T0_n = T0.copy(); T0_n[:3, 3] /= path_length_m
        T1_n = T1.copy(); T1_n[:3, 3] /= path_length_m
    else:
        T0_n, T1_n = T0, T1

    # Keyframe alphas: uniformly spaced, excluding endpoints
    kf_alphas   = np.linspace(0.0, 1.0, n_keyframes + 2)[1:-1]  # [0.2, 0.4, 0.6, 0.8]
    vid_alphas  = np.linspace(0.0, 1.0, n_video_frames)          # [0, ..., 1]

    keyframe_poses = interpolate_se3(T0_n, T1_n, kf_alphas)   # [n_kf, 4, 4]
    video_poses    = interpolate_se3(T0_n, T1_n, vid_alphas)  # [n_video, 4, 4]

    return {
        'keyframe_poses' : keyframe_poses,
        'video_poses'    : video_poses,
        'keyframe_alphas': kf_alphas.astype(np.float32),
        'path_length_m'  : path_length_m,
    }