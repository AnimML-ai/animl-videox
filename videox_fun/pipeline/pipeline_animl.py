# videox_fun/pipeline/pipeline_animl.py
"""
AniML Walkthrough Inference Pipeline — I2V + Plücker + (stub) GLD.

Conditioning streams:
  A) I2V: first frame via inpaint mask (mask = 0 at t=0, 255 elsewhere)
  B) Plücker camera trajectory via control_camera_video (SimpleAdapter)
  C) GLD projected geometry tokens — currently None (stub); wired in second pass

Camera trajectory is built from pose_first / pose_last (default: identity →
1-unit forward); caller may supply explicit c2w matrices from any source.
"""

from __future__ import annotations
import math
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

from ..data.trajectory import build_walkthrough_trajectory
from ..data.utils import ray_condition


# ── Default intrinsics ─────────────────────────────────────────────────────────

def _default_intrinsics(height: int, width: int) -> torch.Tensor:
    """
    Pinhole intrinsics assuming ~60° horizontal FoV.
    Returns [1, 4] float32 tensor [fx, fy, cx, cy].
    """
    fx = width / (2.0 * math.tan(math.radians(30.0)))
    fy = fx
    cx = width  / 2.0
    cy = height / 2.0
    return torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)


# ── Default poses ──────────────────────────────────────────────────────────────

def _default_poses() -> tuple[np.ndarray, np.ndarray]:
    """
    Identity first pose; 1-unit forward (+Z) last pose.
    Both are c2w [4, 4] float32.
    """
    pose_first = np.eye(4, dtype=np.float32)
    pose_last  = np.eye(4, dtype=np.float32)
    pose_last[2, 3] = 1.0
    return pose_first, pose_last


class AniMLPipeline:
    """
    Thin wrapper around Wan2_2FunControlPipeline that wires AniML conditioning.

    Usage:
        pipe = AniMLPipeline.from_pretrained(
            "/home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera"
        )
        result = pipe(
            first_frame_image=Image.open("room.jpg"),
            prompt="A smooth walkthrough of a modern living room.",
        )
    """

    DEFAULT_CONFIG = "config/wan2.2/wan_civitai_i2v.yaml"

    def __init__(self, wan_pipe, boundary: float, device: torch.device, weight_dtype: torch.dtype):
        self._pipe        = wan_pipe
        self._boundary    = boundary
        self.device       = device
        self.weight_dtype = weight_dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
        gpu_memory_mode: str = "sequential_cpu_offload",
        config_path: str = None,
    ) -> "AniMLPipeline":
        """
        Load all components from a VideoX-Fun checkpoint directory.

        Args:
            model_path:       path to Wan2.2-Fun-A14B-Control-Camera directory.
            device:           torch device string, default "cuda".
            weight_dtype:     torch.bfloat16 (default) or torch.float16.
            gpu_memory_mode:  "model_full_load"         — all weights on cuda:0
                              "model_cpu_offload"       — offload after each use
                              "sequential_cpu_offload"  — layer-by-layer (saves most VRAM)
                              "multi_gpu"               — low_noise → cuda:0,
                                                          high_noise → cuda:1,
                                                          VAE + T5   → cuda:0
            config_path:      override for the OmegaConf yaml.
        """
        from ..models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                              AutoTokenizer, Wan2_2Transformer3DModel,
                              WanT5EncoderModel)
        from ..utils.fp8_optimization import replace_parameters_by_name
        from ..utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from .pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline

        if config_path is None:
            config_path = cls.DEFAULT_CONFIG
        config = OmegaConf.load(config_path)

        tkw = config["transformer_additional_kwargs"]
        boundary     = tkw.get("boundary", 0.875)
        low_subpath  = tkw.get("transformer_low_noise_model_subpath",  "low_noise_model")
        high_subpath = tkw.get("transformer_high_noise_model_subpath", "high_noise_model")
        combination  = tkw.get("transformer_combination_type", "moe")

        transformer_kwargs = dict(
            transformer_additional_kwargs=OmegaConf.to_container(tkw),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_path, low_subpath), **transformer_kwargs
        )
        transformer_2 = (
            Wan2_2Transformer3DModel.from_pretrained(
                os.path.join(model_path, high_subpath), **transformer_kwargs
            ) if combination == "moe" else None
        )

        vkw = config["vae_kwargs"]
        Chosen_VAE = {"AutoencoderKLWan": AutoencoderKLWan,
                      "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[
                          vkw.get("vae_type", "AutoencoderKLWan")]
        vae = Chosen_VAE.from_pretrained(
            os.path.join(model_path, vkw.get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(vkw),
        ).to(weight_dtype)

        ekw = config["text_encoder_kwargs"]
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_path, ekw.get("tokenizer_subpath", "tokenizer"))
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_path, ekw.get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=OmegaConf.to_container(ekw),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()

        from ..utils.utils import filter_kwargs
        skw = config["scheduler_kwargs"]
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(skw))
        )

        pipe = Wan2_2FunControlPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
        )

        dev = torch.device(device)
        if gpu_memory_mode == "sequential_cpu_offload":
            replace_parameters_by_name(transformer, ["modulation"], device=dev)
            transformer.freqs = transformer.freqs.to(device=dev)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation"], device=dev)
                transformer_2.freqs = transformer_2.freqs.to(device=dev)
            pipe.enable_sequential_cpu_offload(device=dev)
        elif gpu_memory_mode == "model_cpu_offload":
            pipe.enable_model_cpu_offload(device=dev)
        elif gpu_memory_mode == "multi_gpu":
            # Split the two MoE transformers across two GPUs.
            # low_noise  → cuda:0  (handles timesteps below boundary)
            # high_noise → cuda:1  (handles timesteps above boundary)
            # VAE + T5   → cuda:0
            dev0 = torch.device("cuda:0")
            dev1 = torch.device("cuda:1")
            replace_parameters_by_name(transformer, ["modulation"], device=dev0)
            transformer.freqs = transformer.freqs.to(dev0)
            transformer.to(dev0)
            if transformer_2 is not None:
                replace_parameters_by_name(transformer_2, ["modulation"], device=dev1)
                transformer_2.freqs = transformer_2.freqs.to(dev1)
                transformer_2.to(dev1)
            vae.to(dev0)
            text_encoder.to(dev0)
            dev = dev0   # primary device for latents / scheduling
        else:  # model_full_load
            pipe.to(device=dev)

        return cls(pipe, boundary=boundary, device=dev, weight_dtype=weight_dtype)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_plucker(
        self,
        video_poses: np.ndarray,      # [T, 4, 4] c2w
        intrinsics_K: torch.Tensor,   # [N, 4]  [fx, fy, cx, cy]
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Build Plücker tensor [1, 6, T, H, W] for SimpleAdapter (stream B)."""
        T   = video_poses.shape[0]
        c2w = torch.from_numpy(video_poses).float().unsqueeze(0)       # [1, T, 4, 4]
        K   = intrinsics_K[0:1].unsqueeze(0).expand(1, T, -1)         # [1, T, 4]
        plucker = ray_condition(
            K.to(self.device), c2w.to(self.device), height, width, self.device
        )                                                               # [1, T, H, W, 6]
        plucker = plucker.permute(0, 4, 1, 2, 3)                      # [1, 6, T, H, W]
        print(f"[DEBUG] plucker  min={plucker.min():.4f}  max={plucker.max():.4f}  mean={plucker.mean():.4f}")
        return plucker

    @staticmethod
    def _build_i2v_input(
        first_frame: Image.Image,
        n_frames: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build pixel-space video tensor and mask for I2V (stream A).

        Only frame 0 is conditioned (mask = 0); all other frames are free (mask = 255).

        Returns:
            video      [1, 3, n_frames, H, W]  float32 in [0, 1]
            mask_video [1, 1, n_frames, H, W]  float32
        """
        img = first_frame.resize((width, height))
        f   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()  # [3, H, W]

        video = f.unsqueeze(0).unsqueeze(2).expand(1, -1, n_frames, -1, -1).clone()
        video = video / 255.0

        mask = torch.full((1, 1, n_frames, height, width), 255.0, dtype=torch.float32)
        mask[:, :, 0] = 0.0   # condition only the first frame

        return video, mask

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(
        self,
        first_frame_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        n_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        seed: int = 42,
        shift: int = 5,
        pose_first: np.ndarray = None,
        pose_last: np.ndarray = None,
        video_poses: np.ndarray = None,
        intrinsics_K: torch.Tensor = None,
        normalize_path: bool = True,
    ):
        """
        Args:
            first_frame_image:  PIL image to condition the video on (frame 0).
            prompt:             text description.
            n_frames:           must be 4k+1. Default 81.
            height, width:      output resolution. Default 720×1280.
            pose_first:         [4,4] float32 c2w for start of walkthrough.
                                Defaults to identity (camera at origin).
            pose_last:          [4,4] float32 c2w for end of walkthrough.
                                Defaults to 1-unit forward translation.
            intrinsics_K:       [N, 4] float32 [fx, fy, cx, cy].
                                Defaults to ~60° FoV pinhole model.

        Returns:
            WanPipelineOutput  (.videos tensor [B, C, T, H, W] in [0, 1])
        """

        # ── Defaults ──────────────────────────────────────────────────────────
        if intrinsics_K is None:
            intrinsics_K = _default_intrinsics(height, width)

        # ── Trajectory ────────────────────────────────────────────────────────
        if video_poses is None:
            # Build from pose_first / pose_last via SE(3) interpolation
            if pose_first is None or pose_last is None:
                pose_first, pose_last = _default_poses()
            traj = build_walkthrough_trajectory(
                pose_first=pose_first,
                pose_last=pose_last,
                n_keyframes=4,
                n_video_frames=n_frames,
                normalize_path=normalize_path,
            )
            video_poses = traj["video_poses"]   # [n_frames, 4, 4]
        # else: caller supplied per-frame poses directly (e.g. from DA3)

        # ── Stream A: I2V first-frame conditioning ────────────────────────────
        video, mask_video = self._build_i2v_input(first_frame_image, n_frames, height, width)

        # ── Stream B: Plücker camera trajectory ──────────────────────────────
        control_camera_video = self._build_plucker(
            video_poses, intrinsics_K, height, width
        )  # [1, 6, n_frames, H, W]

        # ── Stream C: GLD geometry tokens — stub (None) ───────────────────────
        # GLDProjector not yet trained; will be wired in second training pass.
        gld_f1_latents = None

        # ── Denoising ─────────────────────────────────────────────────────────
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=n_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            video=video,
            mask_video=mask_video,
            control_camera_video=control_camera_video,
            boundary=self._boundary,
            shift=shift,
            gld_f1_latents=gld_f1_latents,
        )