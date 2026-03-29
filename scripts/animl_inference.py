#!/usr/bin/env python3
"""
AniML walkthrough inference — CLI entry point.

Streams active:
  A) I2V  — first frame conditioning via inpaint mask
  B) Plücker — camera trajectory from pose_first → pose_last
  C) GLD  — stub (None); wired in second pass once GLDProjector is trained

--- Mode 1: manual image + trajectory ---
    python scripts/animl_inference.py \\
        --image /path/to/room.jpg \\
        --prompt "A smooth walkthrough." \\
        --output out.mp4

--- Mode 2: load a real sample from the da3 JSONL ---
    python scripts/animl_inference.py \\
        --from_jsonl /home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl \\
        --jsonl_idx 0 \\
        --output out.mp4

    Automatically loads first_frame, pose_first/pose_last from da3_w2c_raw[0/-1],
    and intrinsics from da3_intrinsics_raw[0], scaled to --height/--width.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ── Pose / intrinsic helpers ──────────────────────────────────────────────────

def w2c_3x4_to_c2w_4x4(w2c_3x4: list) -> np.ndarray:
    """Convert a 3×4 w2c list-of-lists to a float32 c2w [4,4] ndarray."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = np.array(w2c_3x4, dtype=np.float32)
    return np.linalg.inv(w2c)


def scale_intrinsics(K_3x3: list, src_h: float, src_w: float,
                     tgt_h: int, tgt_w: int) -> torch.Tensor:
    """
    Extract [fx, fy, cx, cy] from a 3×3 K matrix and scale to target resolution.
    Returns [1, 4] float32 tensor.
    """
    fx = K_3x3[0][0] * (tgt_w / src_w)
    fy = K_3x3[1][1] * (tgt_h / src_h)
    cx = K_3x3[0][2] * (tgt_w / src_w)
    cy = K_3x3[1][2] * (tgt_h / src_h)
    return torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)


def load_jsonl_sample(jsonl_path: str, idx: int, tgt_h: int, tgt_w: int):
    """
    Load a single sample from the da3 JSONL dataset.

    Returns:
        first_frame:    PIL.Image
        video_poses:    [T, 4, 4] float32 c2w  — all per-frame DA3 poses
        intrinsics_K:   [1, 4] float32 tensor  [fx, fy, cx, cy]
        caption:        str
        last_frame_path: str
    """
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i == idx:
                s = json.loads(line)
                break
        else:
            raise IndexError(f"JSONL has fewer than {idx+1} lines.")

    first_frame = Image.open(s["first_frame"]).convert("RGB")

    # All per-frame c2w poses — re-expressed relative to first frame (identity at t=0)
    video_poses = np.stack([w2c_3x4_to_c2w_4x4(w) for w in s["da3_w2c_raw"]])  # [T, 4, 4]
    if np.isnan(video_poses).any():
        raise ValueError(f"Sample {idx} has NaN poses (DA3 reconstruction failed) — pick a different --jsonl_idx.")
    T0_inv = np.linalg.inv(video_poses[0])
    video_poses = np.einsum('ij,njk->nik', T0_inv, video_poses).astype(np.float32)

    # Debug: pose scale info
    translations = video_poses[:, :3, 3]  # [T, 3]
    path_length  = float(np.linalg.norm(np.diff(translations, axis=0), axis=1).sum())
    end_dist     = float(np.linalg.norm(translations[-1]))
    print(f"[DEBUG] pose[0] t={video_poses[0,:3,3]}  (should be ~0)")
    print(f"[DEBUG] pose[-1] t={video_poses[-1,:3,3]}")
    print(f"[DEBUG] path_length={path_length:.4f}  end_dist={end_dist:.4f}")

    # DA3 intrinsics: K is 3×3; source resolution ≈ cx*2 × cy*2
    K0    = s["da3_intrinsics_raw"][0]
    src_w = K0[0][2] * 2.0
    src_h = K0[1][2] * 2.0
    intrinsics_K = scale_intrinsics(K0, src_h, src_w, tgt_h, tgt_w)

    return first_frame, video_poses, intrinsics_K, s["caption"], s["first_frame"], s["last_frame"]


# ── Comparison helper ─────────────────────────────────────────────────────────

def save_lastframe_comparison(video_path: str, gt_first_frame_path: str,
                               gt_last_frame_path: str, output_path: str):
    """Stitch [GT start frame | GT last frame | Inference last frame]."""
    import subprocess
    import tempfile
    from PIL import Image, ImageDraw

    # Extract last frame from generated mp4
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ['ffmpeg', '-y', '-sseof', '-1', '-i', video_path,
         '-vframes', '1', tmp_path, '-loglevel', 'error'],
        check=True,
    )

    gen   = Image.open(tmp_path).convert('RGB')
    gt_s  = Image.open(gt_first_frame_path).convert('RGB')
    gt_e  = Image.open(gt_last_frame_path).convert('RGB')
    Path(tmp_path).unlink(missing_ok=True)

    # Resize all to the same height as the generated frame
    h = gen.height
    def _resize(img):
        return img.resize((int(img.width * h / img.height), h), Image.LANCZOS)
    gt_s = _resize(gt_s)
    gt_e = _resize(gt_e)

    pad   = 10
    lbl_h = 36
    total_w = gt_s.width + gt_e.width + gen.width + pad * 4
    canvas  = Image.new('RGB', (total_w, h + lbl_h + pad * 2), (30, 30, 30))

    x0 = pad
    x1 = x0 + gt_s.width + pad
    x2 = x1 + gt_e.width + pad
    canvas.paste(gt_s, (x0, lbl_h + pad))
    canvas.paste(gt_e, (x1, lbl_h + pad))
    canvas.paste(gen,  (x2, lbl_h + pad))

    draw = ImageDraw.Draw(canvas)
    draw.text((x0 + gt_s.width // 2 - 55, 8), 'GT start',          fill=(220, 220, 220))
    draw.text((x1 + gt_e.width // 2 - 45, 8), 'GT end',            fill=(220, 220, 220))
    draw.text((x2 + gen.width  // 2 - 75, 8), 'Inference last frame', fill=(100, 220, 100))

    canvas.save(output_path, quality=92)
    print(f"[AniML] Comparison saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='AniML Walkthrough Inference')

    # --- Mode 1: manual ---
    p.add_argument('--image',       type=str, default=None,
                   help='First frame image path (Mode 1).')
    p.add_argument('--prompt',      type=str, default=None)

    # --- Mode 2: from JSONL ---
    p.add_argument('--from_jsonl',  type=str, default=None,
                   help='Path to da3 JSONL; enables Mode 2.')
    p.add_argument('--jsonl_idx',   type=int, default=0,
                   help='0-based sample index in the JSONL (default 0).')
    p.add_argument('--compare',     action='store_true',
                   help='(Mode 2 only) Save a side-by-side comparison of the '
                        'generated last frame vs the GT last frame.')

    # --- Shared ---
    p.add_argument('--neg_prompt',  type=str, default='')
    p.add_argument('--output',      type=str, required=True,
                   help='Output mp4 path.')
    p.add_argument('--model_path',  type=str,
                   default='/home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera')
    p.add_argument('--n_frames',    type=int,   default=81)
    p.add_argument('--height',      type=int,   default=480,
                   help='Output height. Default 480 (matches da3 3:2 aspect).')
    p.add_argument('--width',       type=int,   default=720,
                   help='Output width.  Default 720 (matches da3 3:2 aspect).')
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--steps',       type=int,   default=40)
    p.add_argument('--cfg',         type=float, default=6.0)

    # --- Mode 1 trajectory ---
    p.add_argument('--forward_m',   type=float, default=1.0,
                   help='Forward translation (metres) for the default trajectory. '
                        'Ignored when --pose_first/--pose_last or --from_jsonl are used.')
    p.add_argument('--pose_first',  type=float, nargs=16, metavar='V',
                   help='Row-major 4×4 c2w matrix for the first camera pose (16 floats).')
    p.add_argument('--pose_last',   type=float, nargs=16, metavar='V',
                   help='Row-major 4×4 c2w matrix for the last camera pose (16 floats).')

    p.add_argument('--no_normalize_path', action='store_true',
                   help='Disable SE(3) path length normalisation. Use to test '
                        'whether VGGT translation scale affects Plücker quality.')
    p.add_argument('--gpu_memory_mode', type=str, default='sequential_cpu_offload',
                   choices=['sequential_cpu_offload', 'model_cpu_offload',
                            'model_full_load', 'multi_gpu'],
                   help='multi_gpu splits low/high transformer across cuda:0 and cuda:1.')
    p.add_argument('--device',      type=str,   default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve inputs ────────────────────────────────────────────────────────
    intrinsics_K = None

    video_poses = None  # if set, bypasses trajectory building in pipeline

    if args.from_jsonl is not None:
        # Mode 2: load everything from JSONL — use all DA3 per-frame poses directly
        first_frame, video_poses, intrinsics_K, caption, gt_first_frame_path, gt_last_frame_path = load_jsonl_sample(
            args.from_jsonl, args.jsonl_idx, args.height, args.width
        )
        prompt = args.prompt or caption
        print(f"[AniML] JSONL sample {args.jsonl_idx}: {caption}")
        print(f"[AniML] video_poses: {video_poses.shape}, scale={np.linalg.norm(video_poses[-1,:3,3] - video_poses[0,:3,3]):.3f}")
        print(f"[AniML] intrinsics_K: {intrinsics_K}")
    else:
        # Mode 1: manual image + optional explicit poses
        if args.image is None:
            raise ValueError("Provide --image (Mode 1) or --from_jsonl (Mode 2).")
        if args.prompt is None:
            raise ValueError("Provide --prompt.")
        first_frame = Image.open(args.image).convert('RGB')
        prompt = args.prompt
        print(f"[AniML] First frame: {args.image}")

        if args.pose_first is not None:
            pose_first = np.array(args.pose_first, dtype=np.float32).reshape(4, 4)
        else:
            pose_first = np.eye(4, dtype=np.float32)

        if args.pose_last is not None:
            pose_last = np.array(args.pose_last, dtype=np.float32).reshape(4, 4)
        else:
            pose_last = np.eye(4, dtype=np.float32)
            pose_last[2, 3] = args.forward_m

    # ── Load pipeline & run ───────────────────────────────────────────────────
    from videox_fun.pipeline.pipeline_animl import AniMLPipeline
    pipe = AniMLPipeline.from_pretrained(
        model_path=args.model_path,
        device=args.device,
        gpu_memory_mode=args.gpu_memory_mode,
    )

    result = pipe(
        first_frame_image=first_frame,
        prompt=prompt,
        negative_prompt=args.neg_prompt,
        n_frames=args.n_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed,
        pose_first=pose_first if video_poses is None else None,
        pose_last=pose_last  if video_poses is None else None,
        video_poses=video_poses,
        intrinsics_K=intrinsics_K,
        normalize_path=not args.no_normalize_path,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from videox_fun.utils.utils import save_videos_grid
    save_videos_grid(result.videos, args.output, fps=16)
    print(f"[AniML] Saved to {args.output}")

    if args.compare and args.from_jsonl is not None:
        comparison_path = str(Path(args.output).with_suffix('')) + '_comparison.jpg'
        save_lastframe_comparison(args.output, gt_first_frame_path, gt_last_frame_path, comparison_path)


if __name__ == '__main__':
    main()