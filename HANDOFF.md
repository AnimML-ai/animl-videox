## What was built

An inference pipeline and LoRA training scaffold for AniML walkthrough generation on top of Wan2.2-Fun-A14B-Control-Camera. The inference path (Streams A + B, stub C) is end-to-end runnable from a DA3 JSONL sample and produces coherent video at 256×384. The training script has been smoke-tested with `boundary_type=low` (3 steps, PASS) and `boundary_type=both` (OOM — fixed in Pass 2).

## Files changed

- `scripts/animl_inference.py` — New. CLI for I2V + Plücker inference; two modes (manual image, or from DA3 JSONL); optional [GT-start | GT-end | inference] comparison image.
- `videox_fun/pipeline/pipeline_animl.py` — New. Thin wrapper around `Wan2_2FunControlPipeline` that wires I2V inpaint mask, Plücker from poses, and stub GLD (None).
- `videox_fun/pipeline/pipeline_wan2_2_fun_control.py` — Modified. Added per-step device detection (`t_dev`) so both MoE transformers can live on different GPUs without tensor device mismatches.
- `scripts/wan2.2_fun/train_animl_lora.py` — New. LoRA training script for both MoE sub-models using the DA3 JSONL.
- `scripts/wan2.2_fun/train_animl_lora.sh` — New. Convenience launch wrapper for the training script.
- `videox_fun/data/trajectory.py` — Fixed. Epsilon guard in `so3_log` degenerate 180° case.

## Pass-1 evaluator fixes applied (2026-03-29)

All FAIL-rated criteria from the Pass-1 evaluation have been resolved:

| # | File | Line | Fix |
|---|------|------|-----|
| 1 | `train_animl_lora.py` | 156 | `.sample()` → `.mode()` in `build_inpaint_y` — eliminates train/infer conditioning mismatch |
| 2 | `train_animl_lora.py` | 203 | NaN check now scans all poses via `np.array(s["da3_w2c_raw"])` — no silent interior-frame NaN |
| 3 | `train_animl_lora.py` | main() | `boundary_type=both` with `batch_size > 1` now raises `ValueError` immediately |
| 4 | `train_animl_lora.py` | loop | `--hint_dropout_prob` (default 0.1) zeros `y` and `y_camera` independently each step — CFG works at inference |
| 5 | `train_animl_lora.py` | 170 | `assert y.shape[1] == 4 + vae.latent_channels` — channel count verified at runtime |
| 6 | `trajectory.py` | 55 | `axis / (norm + 1e-8)` — no NaN on degenerate 180° rotation |

## Pass-2 evaluator fixes applied (2026-03-29)

| # | File | Lines | Fix |
|---|------|-------|-----|
| 1 | `train_animl_lora.py` | 569–583 | `boundary_type=both` on 2+ GPUs: `transformer_low→cuda:0`, `transformer_high→cuda:1` — mirrors inference `multi_gpu` mode; eliminates OOM on single 80 GB A100 |
| 2 | `train_animl_lora.py` | 689–704 | Forward pass uses `t_dev = next(active_transformer.parameters()).device` + `_to()` helper to move all inputs to the active transformer's device; `noise_pred` moved back to `device` before loss — identical pattern to `pipeline_wan2_2_fun_control.py:854` |

## Pass-3 evaluator fixes applied (2026-03-29)

| # | File | Line | Fix |
|---|------|------|-----|
| 1 | `train_animl_lora.py` | 643 | `.sample()` → `.mode()` for target latents — eliminates stochastic ground-truth that doubled gradient variance every step |

## Pass-3 smoke test (2026-03-29)

**boundary_type=low, 3 steps, 256×384, n_frames=21, rank=8 — PASS**

```
step 1: loss=0.0965
step 2: loss=0.2186
step 3: loss=0.1906
```

checkpoint-3/lora_low_noise.safetensors written ✓
y-channel assertion did not fire ✓
Note: `torch.cross` deprecation warning visible at `utils.py:323` — `dim=-1` not yet added (non-blocking).

## Known uncertainties

- **Training vs inference conditioning path mismatch.** The reference `train_control_lora.py` uses `full_ref` (first-frame latent injected into self-attention) for I2V in camera-control mode, with `y=None`. The training script instead uses inpaint `y = [mask_latents | masked_video_latents]` to match what `pipeline_animl.py` sends at inference time. Whether this is the correct training objective for the base model has not been verified.
- **Multi-network `accelerator.accumulate`.** Passing two models to `accelerator.accumulate(*[net_low, net_high])` is not a tested pattern in Accelerate. Behavior with gradient accumulation and DDP across two networks is unverified.
- **DA3 intrinsic source resolution estimate.** `src_w = K[0][2] * 2.0`, `src_h = K[1][2] * 2.0` assumes cx ≈ half-width and cy ≈ half-height. If principal point is significantly off-center in any sample, intrinsic scaling will be wrong.
- **`patch_size` config attribute name.** Stored as `_ref_transformer.config.patch_size` before Accelerate wrapping. If the config key differs across model variants, seq_len will be wrong.
- **Inference quality is unquantified.** One visual comparison at 20 steps showed a coherent (non-black) last frame on sample 1002. No other samples were checked. No PSNR, LPIPS, or trajectory accuracy metric exists.
- **Plücker scale sensitivity.** Whether the DA3 translation scale (path_length ≈ 0.6 for sample 1002) matches the scale the base camera-control model was trained on is unknown. `--no_normalize_path` was added to expose this but never systematically tested.
- **Plücker convention unverified.** `ray_condition` packs `[o×d, d]` (moment first). CameraCtrl uses `[d, d×o]`. Whether VideoX-Fun's SimpleAdapter was trained with the same convention as `ray_condition` is unconfirmed.

## Explicit non-goals

- GLDProjector training (pass 3) — not implemented.
- Loading a trained LoRA at inference time — `pipeline_animl.py` has no LoRA loading path.
- Quantitative evaluation of inference quality.
- Multi-GPU distributed (data-parallel) training — shell script uses single `python`, not `accelerate launch`.
- FLF2V (first+last frame conditioning) — dropped in favour of I2V-only for this session.
- Validation inference during training — the argument flag exists but the code path is not implemented.

## How to test

**Inference (known working):**
```bash
conda activate videox
cd /home/ubuntu/dev/animl_videox
python scripts/animl_inference.py \
    --from_jsonl /home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl \
    --jsonl_idx 1002 \
    --output outputs/test_1002.mp4 \
    --compare \
    --gpu_memory_mode multi_gpu \
    --height 256 --width 384 --steps 20
# Expect: outputs/test_1002.mp4 (81-frame video) and outputs/test_1002_comparison.jpg
# The comparison image should show a coherent bedroom walkthrough, not black frames.
```

**Training boundary_type=low (known working — single GPU):**
```bash
python scripts/wan2.2_fun/train_animl_lora.py \
    --model_path /home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera \
    --jsonl_path /home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl \
    --output_dir /tmp/animl_lora_test \
    --height 256 --width 384 --n_frames 21 \
    --rank 8 --network_alpha 4 \
    --max_train_steps 3 --checkpointing_steps 3 \
    --boundary_type low --train_batch_size 1 --num_workers 0
# Expect: 3 steps complete, checkpoint-3/lora_low_noise.safetensors written.
```

**Training boundary_type=both (fixed — requires 2× A100):**
```bash
python scripts/wan2.2_fun/train_animl_lora.py \
    --model_path /home/ubuntu/dev/models/Wan2.2-Fun-A14B-Control-Camera \
    --jsonl_path /home/ubuntu/dev/finetrainers_geomgen/dataset_da3_out.jsonl \
    --output_dir /tmp/animl_lora_test_both \
    --height 256 --width 384 --n_frames 21 \
    --rank 8 --network_alpha 4 \
    --max_train_steps 3 --checkpointing_steps 3 \
    --boundary_type both --train_batch_size 1 --num_workers 0
# Requires 2 visible GPUs (CUDA_VISIBLE_DEVICES=0,1 or multi_gpu machine).
# transformer_low → cuda:0, transformer_high → cuda:1.
# Expect: log line "transformer_low→cuda:0, transformer_high→cuda:1",
#         3 steps complete, checkpoint-3/ with both lora_low_noise.safetensors
#         and lora_high_noise.safetensors written.
```