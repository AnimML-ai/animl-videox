#!/usr/bin/env python3
"""
AniML LoRA Training — Pass 1: I2V + Plücker, zero GLD.

Trains LoRA adapters on Wan2.2-Fun-A14B-Control-Camera
(both low_noise_model and high_noise_model) using the DA3 JSONL dataset.

Conditioning streams:
  A) I2V: first-frame inpaint mask  (y = [mask_latents | masked_video_latents])
  B) Plücker: per-frame camera rays (y_camera, packed to 24-ch)
  C) GLD: None / zero (stub — wired in pass 2)

The boundary_type controls which transformer sub-model is trained:
  "low"  — only train low_noise_model  (timesteps < boundary)
  "high" — only train high_noise_model (timesteps >= boundary)
  "both" — train both in a single run, routing per batch

Usage — single GPU:
    python scripts/wan2.2_fun/train_animl_lora.py \\
        --model_path /home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera \\
        --jsonl_path /home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl \\
        --output_dir /home/ubuntu/dev/animl_videox/checkpoints/lora_pass1 \\
        --rank 32 --network_alpha 16 \\
        --learning_rate 1e-4 --train_batch_size 1 \\
        --max_train_steps 2000 --checkpointing_steps 200 \\
        --height 256 --width 384 --n_frames 21

Usage — two A100s (data-parallel, single-transformer mode):
    accelerate launch --num_processes 2 scripts/wan2.2_fun/train_animl_lora.py [same args]
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import save_file
from tqdm.auto import tqdm

current_file_path = os.path.abspath(__file__)
for _root in [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from videox_fun.data.utils import ray_condition
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.utils.lora_utils import create_network
from videox_fun.utils.utils import filter_kwargs

logger = get_logger(__name__, log_level="INFO")


# ── Helpers ────────────────────────────────────────────────────────────────────

def resize_mask(mask, latent, process_first_frame_only=True):
    """Trilinear-resize mask to match latent spatial dimensions."""
    latent_size = latent.size()
    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_resized = F.interpolate(
            mask[:, :, 0:1], size=target_size, mode='trilinear', align_corners=False
        )
        target_size[0] = latent_size[2] - 1
        if target_size[0] > 0:
            rest_resized = F.interpolate(
                mask[:, :, 1:], size=target_size, mode='trilinear', align_corners=False
            )
            return torch.cat([first_resized, rest_resized], dim=2)
        return first_resized
    return F.interpolate(mask, size=list(latent_size[2:]), mode='trilinear', align_corners=False)


def w2c_3x4_to_c2w_4x4(w2c_3x4: list) -> np.ndarray:
    """Convert 3×4 w2c list-of-lists → float32 c2w [4, 4]."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = np.array(w2c_3x4, dtype=np.float32)
    return np.linalg.inv(w2c)


def scale_intrinsics(K_3x3: list, src_h: float, src_w: float,
                     tgt_h: int, tgt_w: int) -> torch.Tensor:
    """Scale a 3×3 K matrix to a new resolution → [1, 4] float32 [fx, fy, cx, cy]."""
    fx = K_3x3[0][0] * (tgt_w / src_w)
    fy = K_3x3[1][1] * (tgt_h / src_h)
    cx = K_3x3[0][2] * (tgt_w / src_w)
    cy = K_3x3[1][2] * (tgt_h / src_h)
    return torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)


def build_y_camera(plucker: torch.Tensor) -> torch.Tensor:
    """
    Pack Plücker tensor to y_camera format for the transformer.

    Args:
        plucker: [B, 6, T, H, W]

    Returns:
        y_camera: [B, 24, T_lat, H, W]
            where T_lat = (T + 3) / 4  (VAE temporal stride 4, first frame repeated 4×)
    """
    # Repeat first frame 4× to align with temporal VAE stride, then pack 4 px-frames per latent frame
    packed = torch.concat([
        torch.repeat_interleave(plucker[:, :, 0:1], repeats=4, dim=2),
        plucker[:, :, 1:]
    ], dim=2).transpose(1, 2)                               # [B, T+3, 6, H, W]
    B, F, C, H, W = packed.shape
    packed = packed.contiguous().view(B, F // 4, 4, C, H, W).transpose(2, 3)
    packed = packed.contiguous().view(B, F // 4, C * 4, H, W).transpose(1, 2)
    return packed                                           # [B, 24, T_lat, H, W]


def build_inpaint_y(pixel_values: torch.Tensor, vae, device, dtype) -> torch.Tensor:
    """
    Build the I2V inpaint control tensor y = [mask_latents | masked_video_latents].

    First frame is the conditioning frame (mask=0); all other frames are free (mask=1).

    Args:
        pixel_values: [B, T, 3, H, W] in [-1, 1]

    Returns:
        y: [B, 4 + vae.latent_channels, T_lat, H_lat, W_lat]
    """
    B, T, C_px, H, W = pixel_values.shape

    # Build per-pixel mask: 0 at t=0 (conditioned), 1 elsewhere (free)
    mask_cond = torch.ones(B, 1, T, H, W, device=device, dtype=torch.float32)
    mask_cond[:, :, 0] = 0.0

    # Masked video: only first frame survives, rest zeroed
    video_BCTHW = pixel_values.permute(0, 2, 1, 3, 4)          # [B, 3, T, H, W]
    masked_video = video_BCTHW * (mask_cond.expand(-1, C_px, -1, -1, -1) < 0.5)
    # mask_cond=0 at t=0 → condition=True → keep video; mask_cond=1 elsewhere → False → zero

    # Encode masked video with frozen VAE
    with torch.no_grad():
        masked_latents = vae.encode(masked_video.to(dtype))[0].mode()      # [B, 16, T_lat, H_lat, W_lat]

    # Pack mask to match latent temporal stride (same pattern as pipeline)
    mask_packed = torch.concat([
        torch.repeat_interleave(mask_cond[:, :, 0:1], repeats=4, dim=2),
        mask_cond[:, :, 1:]
    ], dim=2)                                                               # [B, 1, T+3, H, W]
    mask_packed = mask_packed.view(B, mask_packed.shape[2] // 4, 4, H, W).transpose(1, 2)
    # [B, 4, T_lat, H, W]

    # Spatially resize to latent dims
    mask_latents = resize_mask(1.0 - mask_packed, masked_latents, True).to(device, dtype)
    # 1-mask_packed: 1 at t=0 (frame is conditioned), 0 elsewhere

    y = torch.cat([mask_latents, masked_latents], dim=1)
    expected_y_ch = 4 + vae.latent_channels
    assert y.shape[1] == expected_y_ch, (
        f"y channel count {y.shape[1]} != expected {expected_y_ch}. "
        f"Verify transformer control adapter in_channels."
    )
    return y                                                                 # [B, 20, T_lat, H_lat, W_lat]


# ── Dataset ────────────────────────────────────────────────────────────────────

class AniMLDataset(torch.utils.data.Dataset):
    """
    Loads training samples from the DA3 JSONL.

    Each sample provides:
        pixel_values  [T, 3, H, W]  float32 in [-1, 1]
        plucker       [6, T, H, W]  float32  Plücker ray embeddings
        caption       str
    """

    def __init__(self, jsonl_path: str, height: int, width: int, n_frames: int):
        assert (height % 16 == 0) and (width % 16 == 0), \
            f"height and width must be divisible by 16, got {height}×{width}"
        assert (n_frames - 1) % 4 == 0, \
            f"n_frames must be 4k+1 (VAE stride 4), got {n_frames}"

        self.height = height
        self.width  = width
        self.n_frames = n_frames

        # Load and validate all samples upfront
        self.samples = []
        nan_count = 0
        with open(jsonl_path) as f:
            for line in f:
                s = json.loads(line.strip())
                # NaN check across all poses
                try:
                    all_poses = np.array(s["da3_w2c_raw"], dtype=np.float32)  # [N, 3, 4]
                    if np.isnan(all_poses).any():
                        nan_count += 1
                        continue
                except Exception:
                    nan_count += 1
                    continue
                self.samples.append(s)

        logger.info(
            f"AniMLDataset: {len(self.samples)} valid samples "
            f"({nan_count} skipped for NaN poses) from {jsonl_path}"
        )

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, video_path: str, indices: np.ndarray) -> np.ndarray:
        """Load specific frame indices from an mp4. Returns [N, H, W, 3] uint8."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices_clipped = np.clip(indices, 0, max(total - 1, 0))

        prev_idx = -1
        for idx in indices_clipped:
            if idx != prev_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # Repeat last valid frame on failure
                frames.append(frames[-1] if frames else np.zeros((self.height, self.width, 3), dtype=np.uint8))
                prev_idx = idx
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            prev_idx = idx

        cap.release()
        return np.stack(frames, axis=0)   # [N, H, W, 3]

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # ── Poses (relative to frame 0) ───────────────────────────────────────
        n_video_poses = len(s["da3_w2c_raw"])
        all_c2w = np.stack([w2c_3x4_to_c2w_4x4(w) for w in s["da3_w2c_raw"]])  # [N_v, 4, 4]
        T0_inv = np.linalg.inv(all_c2w[0])
        all_c2w = np.einsum('ij,njk->nik', T0_inv, all_c2w).astype(np.float32)  # relative

        # Frame indices for subsampling
        frame_indices = np.linspace(0, n_video_poses - 1, self.n_frames).astype(int)
        poses = all_c2w[frame_indices]   # [n_frames, 4, 4]

        # ── Intrinsics ────────────────────────────────────────────────────────
        K0 = s["da3_intrinsics_raw"][0]
        src_w = K0[0][2] * 2.0
        src_h = K0[1][2] * 2.0
        K_tensor = scale_intrinsics(K0, src_h, src_w, self.height, self.width)  # [1, 4]

        # ── Plücker embeddings ────────────────────────────────────────────────
        c2w_t  = torch.from_numpy(poses).float().unsqueeze(0)              # [1, T, 4, 4]
        K_exp  = K_tensor[0:1].unsqueeze(0).expand(1, self.n_frames, -1)   # [1, T, 4]
        plucker = ray_condition(K_exp, c2w_t, self.height, self.width, 'cpu')  # [1, T, H, W, 6]
        plucker = plucker[0].permute(3, 0, 1, 2)                           # [6, T, H, W]

        # ── Video frames ──────────────────────────────────────────────────────
        frames_np = self._load_video_frames(s["video"], frame_indices)     # [T, H, W, 3] uint8
        pixel_values = (
            torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        )  # [T, 3, H, W] in [-1, 1]

        return {
            "pixel_values": pixel_values,   # [T, 3, H, W]
            "plucker":       plucker,        # [6, T, H, W]
            "caption":       s.get("caption", ""),
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "plucker":      torch.stack([b["plucker"]      for b in batch]),
        "caption":      [b["caption"] for b in batch],
    }


# ── Text encoding ──────────────────────────────────────────────────────────────

def encode_prompts(text_encoder, tokenizer, prompts, device, max_length=512):
    """Encode a list of prompts → list of variable-length [seq, dim] embeddings."""
    ids = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(
            ids.input_ids.to(device),
            attention_mask=ids.attention_mask.to(device),
        )[0]
    seq_lens = ids.attention_mask.gt(0).sum(dim=1).long()
    return [e[:l] for e, l in zip(embeds, seq_lens)]


# ── Sigma helpers ──────────────────────────────────────────────────────────────

def get_sigmas(scheduler, timesteps, n_dim, dtype, device):
    """Look up flow-matching sigma values for a batch of timesteps."""
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_ts = scheduler.timesteps.to(device)
    timesteps_dev = timesteps.to(device)
    step_indices = [(schedule_ts == t).nonzero().item() for t in timesteps_dev]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="AniML LoRA Training (Pass 1)")

    # Model
    p.add_argument("--model_path", type=str,
                   default="/home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera")
    p.add_argument("--config_path", type=str, default=None,
                   help="Override for model config yaml (default: config/wan2.2/wan_civitai_i2v.yaml)")

    # Data
    p.add_argument("--jsonl_path", type=str, required=True,
                   help="Path to da3 JSONL dataset.")
    p.add_argument("--height",    type=int, default=256)
    p.add_argument("--width",     type=int, default=384)
    p.add_argument("--n_frames",  type=int, default=21,
                   help="Frames per training clip (must be 4k+1). Default 21.")

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--checkpointing_steps", type=int, default=200)
    p.add_argument("--checkpoints_total_limit", type=int, default=None)

    # LoRA
    p.add_argument("--rank",          type=int,   default=32)
    p.add_argument("--network_alpha", type=float, default=16.0)
    p.add_argument("--target_name",   type=str,   default=None,
                   help="Comma-separated layer names to LoRA-ify. None = all Linear layers.")
    p.add_argument("--lora_skip_name", type=str, default=None,
                   help="Comma-separated layer names to skip.")

    # Training
    p.add_argument("--boundary_type", type=str, default="both",
                   choices=["low", "high", "both"],
                   help="Which transformer to train. 'both' routes per timestep.")
    p.add_argument("--train_batch_size",          type=int,   default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_train_steps",           type=int,   default=2000)
    p.add_argument("--learning_rate",             type=float, default=1e-4)
    p.add_argument("--lr_scheduler",              type=str,   default="constant_with_warmup")
    p.add_argument("--lr_warmup_steps",           type=int,   default=100)
    p.add_argument("--max_grad_norm",             type=float, default=1.0)
    p.add_argument("--weighting_scheme",          type=str,   default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--seed",                      type=int,   default=42)
    p.add_argument("--num_workers",               type=int,   default=2)

    # Memory
    p.add_argument("--hint_dropout_prob", type=float, default=0.1,
                   help="Probability of zeroing each conditioning signal per step "
                        "(inpaint y and camera y_camera independently). "
                        "Required for CFG to work at inference.")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--vae_mini_batch",         type=int, default=1)

    # Validation
    p.add_argument("--validation_jsonl_idx",  type=int,   default=None,
                   help="If set, run a validation inference at each checkpoint.")
    p.add_argument("--validation_steps",      type=int,   default=None)
    p.add_argument("--validation_n_frames",   type=int,   default=21)
    p.add_argument("--validation_steps_cfg",  type=int,   default=20)

    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.boundary_type == "both" and args.train_batch_size > 1:
        raise ValueError(
            "--boundary_type=both with batch_size > 1 routes all samples in a "
            "batch to a single transformer based on mean timestep. "
            "Use --train_batch_size 1 or --boundary_type low/high."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        log_with=None,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Config ────────────────────────────────────────────────────────────────
    config_path = args.config_path or "config/wan2.2/wan_civitai_i2v.yaml"
    config = OmegaConf.load(config_path)
    tkw = config["transformer_additional_kwargs"]
    boundary      = tkw.get("boundary", 0.875)
    low_subpath   = tkw.get("transformer_low_noise_model_subpath",  "low_noise_model")
    high_subpath  = tkw.get("transformer_high_noise_model_subpath", "high_noise_model")
    combination   = tkw.get("transformer_combination_type", "moe")
    weight_dtype  = torch.bfloat16

    # ── Models ────────────────────────────────────────────────────────────────
    from videox_fun.models import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path,
                     config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_path,
                     config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    vkw = config["vae_kwargs"]
    Chosen_VAE = {"AutoencoderKLWan": AutoencoderKLWan,
                  "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
                      vkw.get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_VAE.from_pretrained(
        os.path.join(args.model_path, vkw.get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(vkw),
    ).to(weight_dtype)

    transformer_kwargs = dict(
        transformer_additional_kwargs=OmegaConf.to_container(tkw),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Determine which sub-models to load based on boundary_type
    transformer_low = transformer_high = None
    if args.boundary_type in ("low", "both"):
        transformer_low = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.model_path, low_subpath), **transformer_kwargs
        )
    if args.boundary_type in ("high", "both"):
        transformer_high = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.model_path, high_subpath), **transformer_kwargs
        ) if combination == "moe" else None

    # ── Freeze everything ─────────────────────────────────────────────────────
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if args.boundary_type in ("low", "both"):
        transformer_low.requires_grad_(False)
    if args.boundary_type in ("high", "both") and transformer_high is not None:
        transformer_high.requires_grad_(False)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    network_low = network_high = None

    if args.boundary_type in ("low", "both"):
        network_low = create_network(
            1.0, args.rank, args.network_alpha,
            text_encoder, transformer_low,
            target_name=args.target_name,
            skip_name=args.lora_skip_name,
        ).to(weight_dtype)
        network_low.apply_to(text_encoder, transformer_low, False, True)

    if args.boundary_type in ("high", "both") and transformer_high is not None:
        network_high = create_network(
            1.0, args.rank, args.network_alpha,
            text_encoder, transformer_high,
            target_name=args.target_name,
            skip_name=args.lora_skip_name,
        ).to(weight_dtype)
        network_high.apply_to(text_encoder, transformer_high, False, True)

    if args.gradient_checkpointing:
        if args.boundary_type in ("low", "both"):
            transformer_low.enable_gradient_checkpointing()
        if args.boundary_type in ("high", "both") and transformer_high is not None:
            transformer_high.enable_gradient_checkpointing()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable_params = []
    if network_low  is not None: trainable_params += list(network_low.parameters())
    if network_high is not None: trainable_params += list(network_high.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler,
                        OmegaConf.to_container(config["scheduler_kwargs"]))
    )
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps,
                                  device=accelerator.device)

    # ── Dataset & dataloader ──────────────────────────────────────────────────
    dataset = AniMLDataset(args.jsonl_path, args.height, args.width, args.n_frames)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ── LR scheduler ──────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ── Store patch_size before Accelerate wraps the models ──────────────────
    _ref_transformer = (transformer_low if args.boundary_type in ("low", "both")
                        else transformer_high)
    patch_h = _ref_transformer.config.patch_size[1]
    patch_w = _ref_transformer.config.patch_size[2]

    # ── Accelerate prepare ────────────────────────────────────────────────────
    if network_low  is not None: network_low  = accelerator.prepare(network_low)
    if network_high is not None: network_high = accelerator.prepare(network_high)
    optimizer, dataloader, lr_scheduler = accelerator.prepare(optimizer, dataloader, lr_scheduler)

    device = accelerator.device
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    if args.boundary_type == "both" and torch.cuda.device_count() >= 2:
        # Model-parallel split: low on cuda:0, high on cuda:1.
        # Mirrors the inference pipeline's multi_gpu mode so both 14B transformers
        # fit on two 80 GB A100s without OOM.
        transformer_low.to(torch.device("cuda:0"))
        if transformer_high is not None:
            transformer_high.to(torch.device("cuda:1"))
        logger.info("boundary_type=both: transformer_low→cuda:0, transformer_high→cuda:1")
    else:
        if args.boundary_type in ("low",  "both"): transformer_low.to(device)
        if args.boundary_type in ("high", "both") and transformer_high is not None:
            transformer_high.to(device)

    # ── Checkpoint saving helper ───────────────────────────────────────────────
    def save_checkpoint(step: int):
        if not accelerator.is_main_process:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        def _save_network(net, name):
            if net is None:
                return
            state = {
                k: v.to(weight_dtype)
                for k, v in accelerator.unwrap_model(net).state_dict().items()
            }
            save_file(state, os.path.join(ckpt_dir, f"lora_{name}.safetensors"),
                      metadata={"format": "pt", "step": str(step)})

        _save_network(network_low,  "low_noise")
        _save_network(network_high, "high_noise")
        logger.info(f"Saved LoRA checkpoint to {ckpt_dir}")

        # Prune old checkpoints
        if args.checkpoints_total_limit is not None:
            ckpts = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1])
            )
            while len(ckpts) > args.checkpoints_total_limit:
                shutil.rmtree(os.path.join(args.output_dir, ckpts.pop(0)))

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    progress_bar = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    boundary_t = boundary * noise_scheduler.config.num_train_timesteps

    for epoch in range(num_epochs):
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)  # [B, T, 3, H, W]
            plucker      = batch["plucker"].to(device, dtype=weight_dtype)        # [B, 6, T, H, W]
            captions     = batch["caption"]

            B, T, C, H, W = pixel_values.shape

            # accumulate context: include all trainable networks
            _accum_models = [m for m in [network_low, network_high] if m is not None]
            with accelerator.accumulate(*_accum_models):
                # ── VAE encode video ──────────────────────────────────────────
                video_BCTHW = pixel_values.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
                with torch.no_grad():
                    latents = vae.encode(video_BCTHW)[0].mode()   # [B, 16, T_lat, H_lat, W_lat]

                # ── Build camera conditioning (y_camera) ──────────────────────
                y_camera = build_y_camera(plucker)   # [B, 24, T_lat, H, W]

                # ── Build I2V inpaint conditioning (y) ────────────────────────
                y = build_inpaint_y(pixel_values, vae, device, weight_dtype)
                # [B, 20, T_lat, H_lat, W_lat]

                # ── Hint dropout (for CFG compatibility) ──────────────────────
                if args.hint_dropout_prob > 0.0:
                    if torch.rand(1).item() < args.hint_dropout_prob:
                        y = torch.zeros_like(y)
                    if torch.rand(1).item() < args.hint_dropout_prob:
                        y_camera = torch.zeros_like(y_camera)

                # ── Encode text ───────────────────────────────────────────────
                with torch.no_grad():
                    prompt_embeds = encode_prompts(
                        text_encoder, tokenizer, captions, device
                    )

                # ── Flow matching noise ───────────────────────────────────────
                noise = torch.randn_like(latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=B,
                    logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device)

                sigmas = get_sigmas(noise_scheduler, timesteps, latents.ndim, latents.dtype, device)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents

                # seq_len for transformer positional encoding
                bsz, ch, t_lat, h_lat, w_lat = latents.shape
                seq_len = math.ceil(h_lat * w_lat / (patch_h * patch_w) * t_lat)

                # ── Route to transformer ──────────────────────────────────────
                # For batch_size=1 route by scalar timestep; for larger batches
                # we use the majority timestep to pick one transformer per step.
                avg_t = timesteps.float().mean().item()
                if args.boundary_type == "low":
                    active_transformer = transformer_low
                elif args.boundary_type == "high":
                    active_transformer = transformer_high
                else:  # "both" — route per batch
                    active_transformer = (
                        transformer_high if avg_t >= boundary_t else transformer_low
                    )

                # ── Forward pass ──────────────────────────────────────────────
                # Move inputs to wherever this transformer lives (multi-GPU safe).
                t_dev = next(active_transformer.parameters()).device
                def _to(x):
                    return x.to(t_dev) if isinstance(x, torch.Tensor) else x
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(t_dev):
                    noise_pred = active_transformer(
                        x=_to(noisy_latents),
                        context=[_to(e) for e in prompt_embeds],
                        t=_to(timesteps),
                        seq_len=seq_len,
                        y=_to(y),
                        y_camera=_to(y_camera),
                        full_ref=None,
                        gld_f1_latents=None,
                    )
                noise_pred = noise_pred.to(device)

                # ── Loss ──────────────────────────────────────────────────────
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                diff = noise_pred.float() - target.float()
                # Clip outlier pixels (> 50 sigma) to stabilise early training
                mask_clip = (diff.abs() <= 50.0).float()
                loss = (F.mse_loss(noise_pred.float(), target.float(), reduction="none")
                        * mask_clip * weighting.float()).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    t=f"{avg_t:.0f}",
                )

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(global_step)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(global_step)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()