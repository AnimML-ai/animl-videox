# AniML Walkthrough Generation — Architecture & Scaffolding Plan

## Purpose of This Document

Instructions for Claude Code to scaffold the new modules required for GLD-conditioned
walkthrough video generation on top of VideoX-Fun. This document is exhaustive —
Claude Code should not need to ask clarifying questions.

---

## Environment & Deployment Model

### Machines and roles

| Machine | Role |
|---|---|
| **Mac** | Development. Claude Code runs here. PyCharm IDE runs here. All file edits happen here. |
| **Azure** | Execution only. Ubuntu, 2×A100 80GB. Training and inference run here via SSH terminal. |

### Critical constraint for Claude Code
- **Edit only Mac paths.** Never attempt to run Python. Never SSH anywhere.
- All `python` commands in this document are for the human to run manually on Azure.
- PyCharm deployment sync pushes Mac edits to Azure automatically before execution.

### Paths

| | Mac | Azure |
|---|---|---|
| Repo | `~/Documents/umbrella/animl_dev/VideoX-Fun` | `/home/ubuntu/dev/VideoX-Fun` |
| Models | `~/Documents/umbrella/animl_dev/models/` | `/home/ubuntu/dev/models/` |
| Training data | n/a | `/home/ubuntu/dev/data/` (real estate images/videos already present) |
| Conda env (Azure only) | n/a | `conda activate videox` |

### Base checkpoint
`Wan-AI/Wan2.2-Fun-Control-Camera` (14B). Not yet on Azure.
Human runs on Azure when ready:
```bash
conda activate videox
huggingface-cli download Wan-AI/Wan2.2-Fun-Control-Camera \
  --local-dir /home/ubuntu/dev/models/Wan2.2-Fun-Control-Camera
```

---

## Architecture Overview

We extend VideoX-Fun's `Wan2.2-Fun-Control-Camera` pipeline with three conditioning
streams that together enable walkthrough generation from sparse, unposed listing photos.

### Three conditioning streams

| Label | Stream | Mechanism in DiT | What it carries |
|---|---|---|---|
| A | FLF2V boundaries | `y` mask path — existing, extended | VAE-encoded first + last frame as hard temporal anchors |
| B | Camera trajectory | `SimpleAdapter` → additive patch embedding — existing | Full 81-frame Plücker ray embeddings |
| C | GLD geometry | New `GLDProjector` → appended to T5 context | DA3-Base level-1 features from GLD novel views |

Streams are orthogonal by design: B conditions through spatial embedding addition,
C through cross-attention, A through the mask path. All three can be trained or
ablated independently.

### Data flow (inference, end-to-end)

```
Listing photos (N sparse, unposed)
        │
        ▼
   GLD pass 1: source images only
   → poses_c2w [N, 4, 4] + intrinsics_K [N, 4]
        │
        ▼
   trajectory.py: build_walkthrough_trajectory(pose_first, pose_last)
        ├── keyframe_poses [4, 4, 4]   → GLD pass 2 (novel view synthesis)
        └── video_poses    [81, 4, 4]  → ray_condition() → Plücker [1,6,81,H,W]
                                                 │
                                          SimpleAdapter (existing)   [Stream B]
                                                 │
                                        additive patch embed bias

   GLD pass 2: source images + keyframe_poses
   → f1_latents [4, T_tokens, 1536]  (keyframe novel views only)
        │
        ▼
   GLDProjector: Linear(1536→5120) + keyframe pos embed
   → GLD context tokens [4*T_tokens, 5120]
   → cat with T5 tokens → cross-attention context              [Stream C]

   VAE.encode(first_frame_image) → z_first
   VAE.encode(last_frame_image)  → z_last
   mask = [1, 0, ..., 0, 1]   (FLF2V)
   y = zeros; y[:,0]=z_first; y[:,-1]=z_last                  [Stream A]

        All three streams
              │
              ▼
        Wan DiT (fine-tuned)
              │
              ▼
        VAE.decode → 81-frame 720p walkthrough video
```

---

## Files to Create / Modify

```
VideoX-Fun/
├── videox_fun/
│   ├── data/
│   │   ├── gld_stub.py              CREATE  — GLD interface contract (stub)
│   │   ├── trajectory.py            CREATE  — SE(3) geodesic + trajectory builder
│   │   └── utils.py                 MODIFY  — add get_flf2v_mask(), do not touch existing code
│   ├── models/
│   │   ├── gld_projector.py         CREATE  — DA3 latent → Wan cross-attn dim
│   │   └── wan_transformer3d.py     MODIFY  — wire GLDProjector into __init__ and forward()
│   └── pipeline/
│       └── pipeline_animl.py        CREATE  — assembles all 3 streams, calls DiT
├── scripts/
│   └── animl_inference.py           CREATE  — CLI entry point (runs on Azure)
└── ANIML_WALKTHROUGH_PLAN.md        CREATE  — copy of this file at repo root
```

---

## Detailed Specifications

---

### FILE 1: `videox_fun/data/gld_stub.py`  (CREATE)

Interface contract for GLD. Real GLD code replaces `run_gld()` later.
Stub returns zero tensors of correct shapes so all downstream code can be
developed and tested immediately.

```python
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
```

---

### FILE 2: `videox_fun/data/trajectory.py`  (CREATE)

SE(3) geodesic interpolation. Pure numpy, no external dependencies.

```python
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
```

---

### FILE 3: `videox_fun/models/gld_projector.py`  (CREATE)

```python
# videox_fun/models/gld_projector.py
"""
GLD Feature Projector (AniML).

Projects DA3-Base level-1 features [N_keyframes, T_tokens, 1536]
into Wan's cross-attention dimension [N_keyframes * T_tokens, wan_dim].

Output is concatenated onto the T5 context sequence so the DiT
cross-attends to geometry + appearance information from GLD novel views.

wan_dim = 5120 for Wan2.2-Fun-Control-Camera (A14B).
"""

import torch
import torch.nn as nn


class GLDProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = 1536,
        out_dim: int = 5120,
        n_keyframes: int = 4,
        hidden_dim: int = 3072,
    ):
        """
        Args:
            in_dim:       DA3-Base level-1 channel dim. Fixed at 1536.
            out_dim:      Wan DiT cross-attention dim. 5120 for A14B.
            n_keyframes:  number of intermediate GLD keyframe views.
                          A learned embedding is added per keyframe index
                          so the DiT can distinguish trajectory positions.
            hidden_dim:   MLP hidden dim. Default 3072 (= 2 × in_dim).
        """
        super().__init__()

        # Learned embedding per keyframe index [n_keyframes, in_dim]
        # Added (broadcast over T_tokens) before projection.
        self.keyframe_embedding = nn.Embedding(n_keyframes, in_dim)

        self.norm = nn.LayerNorm(in_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Zero-init the final linear layer so that at training start the GLD
        # tokens contribute exactly zero to cross-attention output.
        # The model starts from the pretrained T5-only conditioning and
        # gradually learns to use GLD features — ensures training stability.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, f1_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f1_latents: [N_keyframes, T_tokens, 1536]
                        DA3-Base level-1 features, channel-normalised
                        (zero mean, unit variance — GLD convention).

        Returns:
            context_tokens: [N_keyframes * T_tokens, out_dim]
                            Concatenate onto T5 context along the sequence dim.
        """
        N, T, C = f1_latents.shape

        # Keyframe position embedding broadcast over spatial tokens
        kf_idx = torch.arange(N, device=f1_latents.device)
        kf_emb = self.keyframe_embedding(kf_idx)             # [N, C]
        kf_emb = kf_emb.unsqueeze(1).expand(-1, T, -1)       # [N, T, C]

        x = f1_latents + kf_emb                              # [N, T, C]
        x = self.norm(x)
        x = self.proj(x)                                     # [N, T, out_dim]
        x = x.reshape(N * T, -1)                             # [N*T, out_dim]
        return x
```

---

### FILE 4: `videox_fun/data/utils.py`  (MODIFY)

Find the existing `get_random_mask` function. Add the following new function
**directly below it** without modifying any existing code:

```python
def get_flf2v_mask(
    video_length: int,
    height: int,
    width: int,
    vae_stride_t: int = 4,
    vae_stride_h: int = 8,
    vae_stride_w: int = 8,
) -> torch.Tensor:
    """
    First-Last-Frame-to-Video (FLF2V) conditioning mask.

    Returns a binary mask in VAE latent space where frame 0 (first) and
    frame -1 (last) are set to 1.0 (pinned / conditioned) and all
    intermediate latent frames are 0.0 (free / to be generated).

    Args:
        video_length:  number of video frames in pixel space. Must satisfy
                       (video_length - 1) % vae_stride_t == 0.
                       Default Wan config: video_length=81, vae_stride_t=4
                       → lat_t = (81-1)//4 + 1 = 21.
        height, width: video resolution in pixels.
                       Default 720p: lat_h = 720//8 = 90, lat_w = 1280//8 = 160.
        vae_stride_*:  VAE compression strides. For Wan2.1 VAE (I2V-A14B):
                       temporal=4, spatial=8×8.

    Returns:
        mask: torch.Tensor [1, 1, lat_t, lat_h, lat_w]  float32
              Values: 1.0 at latent frames 0 and -1, 0.0 elsewhere.

    Usage in pipeline:
        mask  = get_flf2v_mask(81, 720, 1280)          # [1,1,21,90,160]
        y     = torch.zeros(1, vae_c, lat_t, lat_h, lat_w)
        y[:, :, 0]  = z_first                          # VAE latent of frame 0
        y[:, :, -1] = z_last                           # VAE latent of last frame
        # Then pass mask and y into the DiT conditioning path (same as I2V).
    """
    assert (video_length - 1) % vae_stride_t == 0, (
        f"video_length={video_length} incompatible with vae_stride_t={vae_stride_t}. "
        f"video_length must be vae_stride_t * n + 1."
    )
    lat_t = (video_length - 1) // vae_stride_t + 1
    lat_h = height // vae_stride_h
    lat_w = width  // vae_stride_w

    mask = torch.zeros(1, 1, lat_t, lat_h, lat_w, dtype=torch.float32)
    mask[:, :,  0, :, :] = 1.0
    mask[:, :, -1, :, :] = 1.0
    return mask
```

Also add `get_flf2v_mask` to the imports exported from `videox_fun/data/__init__.py`
(or whichever `__init__.py` exports data utilities) alongside `get_random_mask`.

---

### FILE 5: `videox_fun/models/wan_transformer3d.py`  (MODIFY)

Two targeted additions. Do not modify any existing logic.
When `add_gld_projector=False` (the default), behaviour is identical to the
original model — this is a strict requirement.

#### Addition 1 — `__init__` method

Find the `__init__` method of the main transformer class
(likely `Wan2_2Transformer3DModel`). Find the block that handles
`add_control_adapter` / `self.control_adapter`. Immediately after that block, add:

```python
        # AniML: GLD geometry projector
        if add_gld_projector:
            from .gld_projector import GLDProjector
            self.gld_projector = GLDProjector(
                in_dim=1536,
                out_dim=dim,
                n_keyframes=n_gld_keyframes,
            )
        else:
            self.gld_projector = None
```

Add these two parameters to the `__init__` signature (with defaults so existing
code requires no changes):
```python
add_gld_projector: bool = False,
n_gld_keyframes:   int  = 4,
```

If the class uses `@register_to_config`, add both parameters there too.

#### Addition 2 — `forward()` method

Find the `forward()` method. Add `gld_f1_latents: torch.Tensor = None` to its
signature.

Then find the line where `encoder_hidden_states` is first used in the attention
computation (typically passed into the first transformer block or assembled before
the block loop). Immediately before that line, add:

```python
        # AniML: append GLD geometry tokens to T5 context
        if self.gld_projector is not None and gld_f1_latents is not None:
            # gld_f1_latents: [N_keyframes, T_tokens, 1536]  (on same device as model)
            gld_tokens = self.gld_projector(gld_f1_latents)
            # gld_tokens: [N_kf * T_tokens, out_dim]
            # encoder_hidden_states: [B, seq_t5, out_dim]
            B = encoder_hidden_states.shape[0]
            gld_tokens = gld_tokens.unsqueeze(0).expand(B, -1, -1)
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, gld_tokens], dim=1
            )
```

Note: the exact variable name for the T5 context may differ
(`encoder_hidden_states`, `context`, `cross_attn_context`, etc.).
Find the correct name by searching for where the T5 text features enter the DiT
blocks. Use that name consistently.

---

### FILE 6: `videox_fun/pipeline/pipeline_animl.py`  (CREATE)

```python
# videox_fun/pipeline/pipeline_animl.py
"""
AniML Walkthrough Inference Pipeline.

Assembles three conditioning streams and runs the Wan DiT:
  A) FLF2V mask + VAE-encoded first/last frame
  B) Plücker camera trajectory via SimpleAdapter
  C) GLD projected geometry tokens appended to T5 context

Note: from_pretrained() and the DiT denoising loop are stubs (NotImplementedError).
Implement them after the scaffold is verified end-to-end.
"""

from __future__ import annotations
import torch
import numpy as np
from PIL import Image

from ..data.gld_stub import run_gld, GLDOutput
from ..data.trajectory import build_walkthrough_trajectory
from ..data.utils import get_flf2v_mask, ray_condition


class AniMLPipeline:
    """
    High-level pipeline for AniML walkthrough generation.

    Minimal usage (scaffold test — runs without model weights):
        from videox_fun.pipeline.pipeline_animl import AniMLPipeline
        from videox_fun.data.gld_stub import run_gld
        from videox_fun.data.trajectory import build_walkthrough_trajectory
        from videox_fun.data.utils import get_flf2v_mask
        # All imports should succeed; pipeline instantiation will raise NotImplementedError.
    """

    def __init__(self, transformer, vae, text_encoder, scheduler, device):
        self.transformer  = transformer
        self.vae          = vae
        self.text_encoder = text_encoder
        self.scheduler    = scheduler
        self.device       = torch.device(device)

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cuda', **kwargs):
        """
        Load all components from a VideoX-Fun checkpoint directory.
        Implement following the pattern in videox_fun/pipeline/__init__.py
        once weights are on Azure and scaffold is verified.
        """
        raise NotImplementedError(
            "Implement from_pretrained() after scaffold verification. "
            "Follow loading pattern in the existing Wan2_2FunControlPipeline."
        )

    def _build_plucker(
        self,
        video_poses: np.ndarray,      # [T, 4, 4]
        intrinsics_K: torch.Tensor,   # [N_total, 4]  — use first row
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build Plücker tensor [1, 6, T, height, width] for SimpleAdapter."""
        T = video_poses.shape[0]
        video_poses_t = torch.from_numpy(video_poses).float()   # [T, 4, 4]
        K = intrinsics_K[0:1].unsqueeze(0).expand(1, T, -1)    # [1, T, 4]
        c2w = video_poses_t.unsqueeze(0)                        # [1, T, 4, 4]
        plucker = ray_condition(
            K.to(self.device), c2w.to(self.device), height, width, self.device
        )  # [1, T, H, W, 6]
        return plucker.permute(0, 4, 1, 2, 3)                   # [1, 6, T, H, W]

    def __call__(
        self,
        source_images: list[Image.Image],
        first_frame_image: Image.Image,
        last_frame_image: Image.Image,
        prompt: str,
        negative_prompt: str = '',
        n_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        n_gld_keyframes: int = 4,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 40,
        seed: int = 42,
        vae_stride_t: int = 4,
        vae_stride_h: int = 8,
        vae_stride_w: int = 8,
    ) -> torch.Tensor:
        """
        Args:
            source_images:      2–8 listing photos. First = video start, last = video end.
            first_frame_image:  PIL image for the first video frame (usually source_images[0]).
            last_frame_image:   PIL image for the last video frame  (usually source_images[-1]).
            prompt:             text description of the walkthrough.
            n_frames:           must be 4k+1. Default 81.
            height, width:      output resolution. Default 720×1280.
            n_gld_keyframes:    intermediate novel views synthesised by GLD. Default 4.

        Returns:
            video: torch.Tensor [T, C, H, W] float32 in [0, 1]
            (once denoising loop is implemented)
        """

        # ── Stage 1: GLD pass 1 — get source poses ────────────────────────────
        gld_src: GLDOutput = run_gld(
            source_images=source_images,
            target_poses_c2w=torch.zeros(0, 4, 4),
        )
        pose_first = gld_src.poses_c2w[0].cpu().numpy()   # c2w of first source image
        pose_last  = gld_src.poses_c2w[-1].cpu().numpy()  # c2w of last source image

        # ── Stage 2: Build trajectory ──────────────────────────────────────────
        traj = build_walkthrough_trajectory(
            pose_first=pose_first,
            pose_last=pose_last,
            n_keyframes=n_gld_keyframes,
            n_video_frames=n_frames,
            normalize_path=True,
        )
        keyframe_poses_np = traj['keyframe_poses']   # [4, 4, 4]
        video_poses_np    = traj['video_poses']      # [81, 4, 4]

        # ── Stage 3: GLD pass 2 — novel view synthesis at keyframe poses ──────
        keyframe_poses_t = torch.from_numpy(keyframe_poses_np).float()
        gld_kf: GLDOutput = run_gld(
            source_images=source_images,
            target_poses_c2w=keyframe_poses_t,
        )
        # Last n_gld_keyframes rows are the novel-view latents
        gld_f1_latents = gld_kf.f1_latents[-n_gld_keyframes:].to(self.device)
        # Shape: [n_gld_keyframes, T_tokens, 1536]

        # ── Stream A: FLF2V mask ───────────────────────────────────────────────
        mask = get_flf2v_mask(
            n_frames, height, width, vae_stride_t, vae_stride_h, vae_stride_w
        ).to(self.device)
        # TODO: encode first_frame_image and last_frame_image through self.vae
        # Following the pattern in videox_fun/wan/image2video.py:
        #   z_first = self.vae.encode(preprocess(first_frame_image))
        #   z_last  = self.vae.encode(preprocess(last_frame_image))
        #   y = torch.zeros(1, vae_c, lat_t, lat_h, lat_w, device=self.device)
        #   y[:, :,  0] = z_first
        #   y[:, :, -1] = z_last
        z_first, z_last, y = None, None, None  # placeholder

        # ── Stream B: Plücker camera trajectory ───────────────────────────────
        control_camera_video = self._build_plucker(
            video_poses_np, gld_kf.intrinsics_K, height, width
        )
        # Shape: [1, 6, n_frames, height, width]

        # ── Stage 4: DiT denoising ─────────────────────────────────────────────
        # Wire z_first, z_last, mask, control_camera_video, gld_f1_latents
        # into the Wan2_2FunControlPipeline call. Implement following the
        # existing predict_v2v_control_camera.py pattern.
        raise NotImplementedError(
            "DiT denoising loop — implement after from_pretrained() is working "
            "and scaffold is verified end-to-end."
        )
```

---

### FILE 7: `scripts/animl_inference.py`  (CREATE)

CLI script. **Runs on Azure only — never invoked by Claude Code.**

```python
#!/usr/bin/env python3
"""
AniML walkthrough inference — CLI entry point.

Run on Azure via SSH terminal:
    conda activate videox
    cd /home/ubuntu/dev/VideoX-Fun
    python scripts/animl_inference.py \
        --images /home/ubuntu/dev/data/listing/*.jpg \
        --prompt "A smooth walkthrough of a modern living room." \
        --output /home/ubuntu/dev/outputs/test_walkthrough.mp4 \
        --model_path /home/ubuntu/dev/models/Wan2.2-Fun-Control-Camera

Note: pipeline.from_pretrained() and denoising are stubs until weights
are downloaded and the pipeline wiring is complete.
"""

import argparse
from pathlib import Path
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AniML Walkthrough Inference')
    p.add_argument('--images',      nargs='+', required=True,
                   help='Listing photo paths (2–8). Order = video start→end.')
    p.add_argument('--prompt',      type=str, required=True)
    p.add_argument('--neg_prompt',  type=str, default='')
    p.add_argument('--output',      type=str, required=True,
                   help='Output mp4 path on Azure.')
    p.add_argument('--model_path',  type=str,
                   default='/home/ubuntu/dev/models/Wan2.2-Fun-Control-Camera')
    p.add_argument('--n_frames',    type=int,   default=81)
    p.add_argument('--height',      type=int,   default=720)
    p.add_argument('--width',       type=int,   default=1280)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--steps',       type=int,   default=40)
    p.add_argument('--cfg',         type=float, default=3.5)
    p.add_argument('--n_keyframes', type=int,   default=4,
                   help='Number of intermediate GLD keyframe views.')
    p.add_argument('--device',      type=str,   default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    image_paths = sorted(args.images)
    assert 2 <= len(image_paths) <= 8, \
        f"Provide 2–8 listing photos, got {len(image_paths)}."

    images = [Image.open(p).convert('RGB') for p in image_paths]
    print(f"[AniML] {len(images)} source images loaded.")
    print(f"[AniML] First frame: {image_paths[0]}")
    print(f"[AniML] Last frame:  {image_paths[-1]}")

    from videox_fun.pipeline.pipeline_animl import AniMLPipeline
    pipe = AniMLPipeline.from_pretrained(
        model_path=args.model_path,
        device=args.device,
    )

    video = pipe(
        source_images=images,
        first_frame_image=images[0],
        last_frame_image=images[-1],
        prompt=args.prompt,
        negative_prompt=args.neg_prompt,
        n_frames=args.n_frames,
        height=args.height,
        width=args.width,
        n_gld_keyframes=args.n_keyframes,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # TODO: save video tensor to mp4 using save_videos_grid or imageio
    print(f"[AniML] Saved to {args.output}")


if __name__ == '__main__':
    main()
```

---

## Execution Order for Claude Code

Work through these steps in order. Verify each step before proceeding.

**Step 1** — Create `videox_fun/data/gld_stub.py`
Verify: `from videox_fun.data.gld_stub import run_gld, GLDOutput` imports cleanly.
Sanity check: `run_gld([Image.new('RGB',(512,512))], torch.zeros(0,4,4))` returns a
`GLDOutput` with `f1_latents.shape == (1, 1296, 1536)`.

**Step 2** — Create `videox_fun/data/trajectory.py`
Verify all imports resolve. Sanity check (numpy only):
```python
import numpy as np
from videox_fun.data.trajectory import se3_exp, se3_log, build_walkthrough_trajectory
T = np.eye(4); T[:3,3] = [1,0,0]; T[:3,:3] = so3_exp(np.array([0,0.5,0]))
assert np.allclose(se3_exp(se3_log(T)), T, atol=1e-6)
traj = build_walkthrough_trajectory(np.eye(4), T)
assert traj['keyframe_poses'].shape == (4,4,4)
assert traj['video_poses'].shape    == (81,4,4)
```

**Step 3** — Create `videox_fun/models/gld_projector.py`
Verify:
```python
import torch
from videox_fun.models.gld_projector import GLDProjector
proj = GLDProjector()
out  = proj(torch.randn(4, 1296, 1536))
assert out.shape == (4*1296, 5120)
# Zero-init check: output should be exactly zero at init
assert torch.allclose(out, torch.zeros_like(out))
```

**Step 4** — Modify `videox_fun/data/utils.py` (add `get_flf2v_mask` only)
Verify:
```python
from videox_fun.data.utils import get_flf2v_mask
mask = get_flf2v_mask(81, 720, 1280)
assert mask.shape == (1, 1, 21, 90, 160)
assert mask.sum().item() == 2 * 90 * 160   # exactly 2 latent frames are 1.0
assert mask[0,0,0].sum()  == 90*160        # first frame all-ones
assert mask[0,0,-1].sum() == 90*160        # last frame all-ones
assert mask[0,0,1].sum()  == 0             # middle frames zero
```

**Step 5** — Modify `videox_fun/models/wan_transformer3d.py`
Rule: existing behaviour when `add_gld_projector=False` must be unchanged.
Verify the existing model instantiation with default args still works
(import only — do not run forward pass without weights).

**Step 6** — Create `videox_fun/pipeline/pipeline_animl.py`
Verify: `from videox_fun.pipeline.pipeline_animl import AniMLPipeline` imports
without error. Instantiation will raise NotImplementedError — that is expected.

**Step 7** — Create `scripts/animl_inference.py`
Verify: `python scripts/animl_inference.py --help` prints usage without error.
(This verification runs on Azure — Claude Code only needs to confirm the file
has no syntax errors by checking it parses cleanly.)

**Step 8** — Copy this document to repo root as `ANIML_WALKTHROUGH_PLAN.md`.

---

## What Is Out of Scope for This Scaffolding Task

The following are explicitly deferred. Claude Code should leave them as
`NotImplementedError` or `# TODO` comments:

- Downloading model weights (manual on Azure)
- `AniMLPipeline.from_pretrained()` implementation
- DiT denoising loop in `pipeline_animl.py`
- VAE encoding of first/last frame images in the pipeline
- Real GLD integration (stub only)
- Training scripts and data loaders
- Any Python execution

---

## Reference: Key Tensor Shapes

| Tensor | Shape | Notes |
|---|---|---|
| GLD f1_latents (504px input) | `[N_views, 1296, 1536]` | (504/14)²=1296 tokens, C=1536 |
| GLDProjector output | `[4×1296, 5120]` | 4 keyframes × 1296 tokens, wan_dim=5120 |
| Plücker (SimpleAdapter input) | `[1, 6, 81, H, W]` | after permute |
| FLF2V mask (720p, 81 frames) | `[1, 1, 21, 90, 160]` | lat_t=21, lat_hw=90×160 |
| VAE latent (Wan2.1 VAE) | `[1, 16, 21, 90, 160]` | C=16 channels |
| y conditioning tensor | `[1, 17, 21, 90, 160]` | C+1: mask channel prepended |
