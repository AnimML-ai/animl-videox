# videox_fun/data/gld_stub.py
"""
GLD (Geometric Latent Diffusion) interface stub.

Real integration replaces run_gld() with a call to the actual GLD model.
All downstream code depends only on GLDOutput — do not change the dataclass fields.

GLD paper: arXiv:2603.22275  (KAIST, Jang et al., 2026)
Backbone:  DA3-Base (Depth Anything 3), ViT-B
Latent:    Intermediate features at transformer block 7 (level 1 in GLD terminology).
           Shape: [V, (H/14)*(W/14), 1536]
           Normalised: zero mean, unit variance per channel (GLD training statistics).
Coordinate convention: right-handed, Y-up, metric scale (VGGT convention).
                       Translations are in approximate metres.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from PIL import Image


@dataclass
class GLDOutput:
    poses_c2w: torch.Tensor
    """[N_total_views, 4, 4]  float32  camera-to-world matrices.
    N_total_views = N_source + N_target.
    First N_source rows: source images, same order as input list.
    Last  N_target rows: target novel views, same order as target_poses_c2w input.
    """

    intrinsics_K: torch.Tensor
    """[N_total_views, 4]  float32  per-view intrinsics (fx, fy, cx, cy) in pixels."""

    f1_latents: torch.Tensor
    """[N_total_views, T_tokens, 1536]  float32
    DA3-Base level-1 features (transformer block 7 output).
    T_tokens = (H_gld // 14) * (W_gld // 14).
    At default 504px: T_tokens = 36 * 36 = 1296.
    Channel-normalised to zero mean / unit variance (GLD convention).
    """

    H_gld: int
    """Height (px) at which GLD processed images. Default 504."""

    W_gld: int
    """Width (px) at which GLD processed images. Default 504."""


def run_gld(
    source_images: list[Image.Image],
    target_poses_c2w: torch.Tensor,   # [N_target, 4, 4]; pass zeros(0,4,4) for pose-only pass
    H_gld: int = 504,
    W_gld: int = 504,
) -> GLDOutput:
    """
    STUB — returns zero-filled tensors of the correct shapes.

    Args:
        source_images:      2–8 PIL.Image listing photos.
                            GLD resizes internally to (H_gld, W_gld).
        target_poses_c2w:   [N_target, 4, 4] novel view poses in world frame
                            (same frame as the source poses GLD estimates).
                            Pass torch.zeros(0, 4, 4) on the first pass when you
                            only need source poses back.
        H_gld, W_gld:       Resolution GLD runs at. Default 504×504.

    Returns:
        GLDOutput with stub (zero) tensors. Replace body with real GLD call.

    Two-pass usage pattern:
        pass1 = run_gld(images, torch.zeros(0,4,4))   # get source poses
        # ... build keyframe_poses from pass1.poses_c2w ...
        pass2 = run_gld(images, keyframe_poses)        # get novel view latents
        f1_kf = pass2.f1_latents[-n_keyframes:]        # keyframe latents only
    """
    n_source = len(source_images)
    n_target = int(target_poses_c2w.shape[0])
    n_total  = n_source + n_target
    T_tokens = (H_gld // 14) * (W_gld // 14)

    # Stub intrinsics: pinhole, ~60° FOV
    focal = W_gld / (2.0 * math.tan(math.pi / 6.0))
    K_row = torch.tensor([focal, focal, W_gld / 2.0, H_gld / 2.0], dtype=torch.float32)
    intrinsics_K = K_row.unsqueeze(0).expand(n_total, -1).clone()

    # Stub poses: identity for source views
    source_poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(n_source, -1, -1).clone()
    if n_target > 0:
        all_poses = torch.cat([source_poses, target_poses_c2w.float()], dim=0)
    else:
        all_poses = source_poses

    return GLDOutput(
        poses_c2w=all_poses,
        intrinsics_K=intrinsics_K,
        f1_latents=torch.zeros(n_total, T_tokens, 1536, dtype=torch.float32),
        H_gld=H_gld,
        W_gld=W_gld,
    )
