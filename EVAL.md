# AniML Pass-2 Evaluation Report
**Date:** 2026-03-29
**Reviewer:** Evaluator (claude-sonnet-4-6)
**Scope:** All files from Pass-1 re-evaluated with Pass-1 fixes applied.
`scripts/wan2.2_fun/train_animl_lora.py`, `videox_fun/data/trajectory.py`,
`videox_fun/pipeline/pipeline_animl.py`, `videox_fun/pipeline/pipeline_wan2_2_fun_control.py`,
`videox_fun/data/utils.py:ray_condition`

**Inference smoke test:** PASS — sample 1002 at 256×384, 20 steps, multi_gpu.
Coherent bedroom walkthrough produced. Output at `outputs/test_1002.mp4`.

---

## Pass-1 Fixes Verification

All 6 items from the Pass-1 BLOCK were applied and verified present in the code:

| # | Fix | Status |
|---|-----|--------|
| 1 | `build_inpaint_y:156` `.sample()` → `.mode()` | CONFIRMED ✓ |
| 2 | NaN scan across all poses via `np.array(s["da3_w2c_raw"])` | CONFIRMED ✓ |
| 3 | `boundary_type=both` + `batch_size>1` raises `ValueError` | CONFIRMED ✓ |
| 4 | `--hint_dropout_prob` zeros `y` and `y_camera` independently | CONFIRMED ✓ |
| 5 | `assert y.shape[1] == 4 + vae.latent_channels` | CONFIRMED ✓ |
| 6 | `axis / (np.linalg.norm(axis) + 1e-8)` | CONFIRMED ✓ |

Pipeline `.mode()` usage also verified at `pipeline_wan2_2_fun_control.py:356,368`. Train/infer conditioning paths match.

---

## Evaluation

```
CRITERION 1 — Plücker Correctness: UNCERTAIN

  videox_fun/data/utils.py:323
  Cross product: torch.cross(rays_o, rays_d) = o × d, packed as [o×d, d].
  CameraCtrl reference packs [d, d×o] — different channel order and negated
  moment. train/infer are self-consistent (same ray_condition function used
  in both). Whether SimpleAdapter's pretrained weights expect [o×d, d] or
  [d, d×o] cannot be confirmed without the base model's training script.
  Risk is real but pre-existing; not introduced by this diff.

  videox_fun/data/utils.py:319
  Comment claims rays_d is [B, V, 3, HW]; actual matmul output is
  [B, V, HW, 3]. Comment is wrong; shape computation is correct.

  No zero-length ray risk: zs=1 always, ||direction|| ≥ 1. ✓


CRITERION 2 — Tensor Shapes: UNCERTAIN

  train_animl_lora.py:121-128 (build_y_camera) exactly replicates the
  pipeline's reshape at pipeline_wan2_2_fun_control.py:693-703. ✓

  train_animl_lora.py:163 — mask_packed.view() silently absorbs C=1
  channel dim. Fragile but matches the pipeline's identical pattern at
  pipeline_wan2_2_fun_control.py:678. Functionally correct for C=1.

  y_camera passed at pixel-space H, W — verified correct; pipeline also
  passes pixel-space to the transformer (SimpleAdapter handles spatial
  downsampling internally). ✓

  Channel interleaving order within y_camera (6 Plücker × 4 temporal → 24)
  matches the pipeline exactly; whether SimpleAdapter's pretrained weights
  expect this interleaving is unverified.


CRITERION 3 — Numerical Stability: PASS

  trajectory.py:56 — epsilon guard confirmed: axis / (norm + 1e-8). ✓
  trajectory.py:85 — tan(theta/2) near pi produces large-finite float,
    not inf; theta/(2*tan) → 0 gracefully. ✓
  ray_condition direction normalization: zs=1 always, safe. ✓
  No NaN/Inf paths identified in training forward/backward loop.


CRITERION 4 — VAE / Conditioning Compatibility: UNCERTAIN

  build_inpaint_y:156 uses .mode(). ✓  Pipeline uses .mode(). ✓
  Hint dropout: y and y_camera zeroed independently per step. ✓
  full_ref=None in both inference and training — consistent. ✓
  y channel assertion fires at first forward pass if channel count is
    wrong. ✓
  Base model pretraining regime (full_ref vs y for I2V) unverifiable
  without original training code. Pre-existing uncertainty; not a
  regression.


CRITERION 5 — GLD Integration: N/A (not implemented)

  gld_f1_latents=None throughout. Keyframe poses are computed but not
  forwarded to any memory bank. Criterion deferred to Pass 3.


CRITERION 6 — z-Buffer Occlusion: N/A (not implemented)

  No point cloud reprojection code. Criterion deferred to Pass 3.
```

---

## Verdict

**APPROVE**

| Criterion | Pass-1 | Pass-2 |
|-----------|--------|--------|
| 1. Plücker Correctness | UNCERTAIN | UNCERTAIN (pre-existing) |
| 2. Tensor Shapes | FAIL | UNCERTAIN (resolved FAILs) |
| 3. Numerical Stability | FAIL | PASS |
| 4. VAE / Conditioning Compatibility | FAIL | UNCERTAIN (resolved FAILs) |
| 5. GLD Integration | UNCERTAIN (stub) | N/A |
| 6. z-Buffer Occlusion | UNCERTAIN (stub) | N/A |

No FAILs remain. All Pass-1 fixes confirmed present and correct.

---

## Inference Smoke Test

```
Sample:   JSONL idx 1002 ("Move forward.")
Config:   256×384, 81 frames, 20 steps, multi_gpu
Result:   PASS — coherent bedroom walkthrough, no black frames
Timing:   ~64s total (~3.2s/step)
Outputs:  outputs/test_1002.mp4, outputs/test_1002_comparison.jpg
```

---

## Training Smoke Test

```
Command:  boundary_type=both, 3 steps, rank=8, 256×384, n_frames=21
Result:   FAIL — OOM on first forward pass

  torch.OutOfMemoryError: CUDA out of memory.
  GPU 0: 79.14 GiB total, 78.28 GiB allocated by PyTorch.
  Both low_noise + high_noise transformers loaded to cuda:0.
  Two A14B models + VAE + text encoder exceeds one 80 GB A100.
  boundary_type=both requires the same two-GPU split as inference
  but train_animl_lora.py has no multi-GPU device placement for training.

Workaround: boundary_type=low (single transformer, single GPU)

Command:  boundary_type=low, 3 steps, rank=8, 256×384, n_frames=21
Result:   PASS

  3/3 steps completed (~4.5s/step)
  Losses: 0.0488 → 0.2924 → 0.1081  (non-zero, non-NaN)
  checkpoint-3/lora_low_noise.safetensors written ✓
  y-channel assertion did not fire ✓ (channel count correct)
```

**Training verdict: PASS with caveat** — basic loop works, but `boundary_type=both`
is broken on a single 80 GB GPU. Training both sub-models requires either
`--boundary_type low` + `--boundary_type high` in separate runs, or adding
multi-GPU device placement to the training script (analogous to `multi_gpu` mode
in inference).

---

## Open Risks (carry forward to Pass 3)

| # | Risk | Severity |
|---|------|----------|
| 1 | `boundary_type=both` OOM on single GPU — needs multi-GPU training path | **High (new)** |
| 2 | Plücker convention vs SimpleAdapter weights unverified | Medium |
| 3 | Base model I2V pretraining used `full_ref` vs `y` — unknown | Medium |
| 4 | `accelerator.accumulate(*two_networks)` with alternating active network — unverified under gradient accumulation | Low |
| 5 | DA3 intrinsic resolution estimate (`cx*2, cy*2`) breaks if principal point is off-center | Low |
| 6 | GLD integration not implemented | Deferred |
| 7 | z-buffer occlusion not implemented | Deferred |

---

# AniML Pass-3 Evaluation Report
**Date:** 2026-03-29
**Reviewer:** Evaluator (claude-sonnet-4-6)
**Scope:** `scripts/wan2.2_fun/train_animl_lora.py`, `videox_fun/data/trajectory.py`,
`videox_fun/pipeline/pipeline_animl.py`, `videox_fun/pipeline/pipeline_wan2_2_fun_control.py`,
`videox_fun/data/utils.py:ray_condition`, `scripts/animl_inference.py`

---

## Evaluation

```
CRITERION 1 — Plücker Correctness: UNCERTAIN
  videox_fun/data/utils.py:323 — torch.cross(rays_o, rays_d) called without
  explicit `dim` argument. Both tensors are [B, V, HW, 3] (3-element in last
  dim). In PyTorch ≥ 1.9 the undeclared-dim path emits a UserWarning and may
  silently misbehave on a PyTorch version change. Must be
  torch.cross(rays_o, rays_d, dim=-1).

  videox_fun/data/utils.py:319 — misleading comment says "# B, V, 3, HW" but
  actual rays_d shape after `directions @ R^T` is [B, V, HW, 3].

  Convention (o×d packed as [o×d, d]) is internally consistent — same
  ray_condition used in both training and inference. Pre-existing; not a
  new regression.

CRITERION 2 — Tensor Shapes: PASS
  build_y_camera (train_animl_lora.py:121–128) exactly replicates the
  pipeline's control_camera_video packing (pipeline_wan2_2_fun_control.py:
  693–703). Both produce [B, 24, T_lat, H, W]. ✓

  build_inpaint_y runtime assertion at train_animl_lora.py:172 catches
  channel mismatches at training time. ✓

  One fragility: train_animl_lora.py:163 — mask_packed.view() implicitly
  squeezes channel=1 via flat arithmetic. Matches the same fragile pattern
  in the pipeline. Correct for C=1.

CRITERION 3 — Numerical Stability: FAIL
  train_animl_lora.py:643 — vae.encode(video_BCTHW)[0].sample() for the
  regression TARGET latents. The Pass-1 fix (.sample() → .mode()) was applied
  only to build_inpaint_y:156 (the conditioning path). The main target
  encoding at line 643 still draws a stochastic posterior sample.
  Both noisy_latents and target use the same sampled `latents`, doubling
  gradient variance; every training step produces a different ground-truth
  target for the same video.
  The pipeline (prepare_mask_latents:356,368) always calls .mode().
  Fix: change line 643 to .mode().

  trajectory.py:56 — epsilon guard on degenerate 180° SO(3) log. ✓
  se3_log:85 — tan(θ/2) near π → finite limit via L'Hôpital. ✓
  ray_condition direction norm: zs=1 always, ||direction|| ≥ 1. ✓

CRITERION 4 — VAE / Conditioning Compatibility: PASS
  Mask range: inference 0/255 normalised by mask_processor.preprocess → [0,1];
  training uses 0/1 directly. Both compute (1 − packed_mask) for mask_latents.
  Consistent. ✓

  Hint dropout (train_animl_lora.py:654–657): y and y_camera zeroed
  independently per step. Correct for CFG. ✓

  dtype: y and y_camera cast to weight_dtype (bfloat16) at point of use;
  transformer weights are bfloat16. No dtype mismatch. ✓

CRITERION 5 — GLD Integration: N/A
  gld_f1_latents=None throughout. Not implemented; deferred to Pass 4.

CRITERION 6 — z-Buffer Occlusion: N/A
  No point cloud reprojection code. Deferred to Pass 4.
```

---

## Verdict

**BLOCK**

| Criterion | Pass-2 | Pass-3 |
|-----------|--------|--------|
| 1. Plücker Correctness | UNCERTAIN | UNCERTAIN (pre-existing) |
| 2. Tensor Shapes | UNCERTAIN | PASS |
| 3. Numerical Stability | PASS | **FAIL** (line 643 .sample()) |
| 4. VAE / Conditioning Compatibility | UNCERTAIN | PASS |
| 5. GLD Integration | N/A | N/A |
| 6. z-Buffer Occlusion | N/A | N/A |

---

## Required Fix

`train_animl_lora.py:643` — change:
```python
latents = vae.encode(video_BCTHW)[0].sample()
```
to:
```python
latents = vae.encode(video_BCTHW)[0].mode()
```

---

## Recommended Fix (not blocking)

`videox_fun/data/utils.py:323` — add `dim=-1` to `torch.cross`:
```python
rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
```

---

# AniML Pass-4 Evaluation Report
**Date:** 2026-03-29
**Reviewer:** Evaluator (claude-sonnet-4-6)
**Scope:** `scripts/wan2.2_fun/train_animl_lora.py`, `videox_fun/data/trajectory.py`,
`videox_fun/data/utils.py:ray_condition`

---

## Pass-3 Fix Verification

| # | Fix | Status |
|---|-----|--------|
| 1 | `train_animl_lora.py:643` `.sample()` → `.mode()` for target latents | CONFIRMED ✓ |

Both VAE encode sites now call `.mode()`:
- `line 156`: `masked_latents = vae.encode(...)[0].mode()` ✓
- `line 643`: `latents = vae.encode(...)[0].mode()` ✓

---

## Evaluation

```
CRITERION 1 — Plücker Correctness: UNCERTAIN

  videox_fun/data/utils.py:323
  torch.cross(rays_o, rays_d) still lacks explicit dim=-1.
  Both tensors are [B, V, HW, 3]. PyTorch searches for the first
  dim of size 3 from dim 0. When B=1 and T≠3, dim=-1 is unambiguous
  — but if T=3 (e.g. a 3-frame debug clip), PyTorch will pick dim=1
  instead of dim=-1 and emit no warning, silently computing wrong
  moment vectors. Recommended fix from Pass 3 not applied.

  videox_fun/data/utils.py:319
  Comment "# B, V, 3, HW" still wrong; actual shape is [B, V, HW, 3].
  Non-functional but misleading.

  Pack order (o×d, d) internally consistent across train and infer.
  Pre-existing; no regression.


CRITERION 2 — Tensor Shapes: PASS

  build_y_camera produces [B, 24, T_lat, H, W] matching pipeline. ✓
  build_inpaint_y assert at line 172 catches channel mismatches. ✓
  Target latents and noisy_latents share the same [B, 16, T_lat, H_lat, W_lat]
  shape — both derive from the same vae.encode call at line 643. ✓
  No new shape issues detected.


CRITERION 3 — Numerical Stability: PASS

  train_animl_lora.py:643 — Pass-3 FAIL resolved.
    latents = vae.encode(video_BCTHW)[0].mode()  ✓
  Training ground-truth is now deterministic; gradient variance halved.

  trajectory.py:56 — epsilon guard confirmed. ✓
  se3_log:85 — tan(θ/2) near π: finite; theta ≈ π caught by
    so3_log before reaching A_inv computation. ✓
  ray_condition direction norm: zs=1, ||direction|| ≥ 1. ✓
  No NaN/Inf paths identified.


CRITERION 4 — VAE / Conditioning Compatibility: PASS

  Hint dropout lines 654–657: y and y_camera zeroed independently. ✓
  weight_dtype cast applied before transformer forward. ✓
  full_ref=None consistent between train and infer. ✓
  No changes since Pass 3.


CRITERION 5 — GLD Integration: N/A

  gld_f1_latents=None throughout. Deferred to Pass 5.


CRITERION 6 — z-Buffer Occlusion: N/A

  No point cloud reprojection code. Deferred to Pass 5.
```

---

## Verdict

**APPROVE**

| Criterion | Pass-3 | Pass-4 |
|-----------|--------|--------|
| 1. Plücker Correctness | UNCERTAIN | UNCERTAIN (pre-existing) |
| 2. Tensor Shapes | PASS | PASS |
| 3. Numerical Stability | **FAIL** | **PASS** (line 643 fixed) |
| 4. VAE / Conditioning Compatibility | PASS | PASS |
| 5. GLD Integration | N/A | N/A |
| 6. z-Buffer Occlusion | N/A | N/A |

No FAILs. Pass-3 required fix confirmed applied.

---

## Recommended Fix (carry forward, not blocking)

`videox_fun/data/utils.py:323` — add `dim=-1` to `torch.cross`:
```python
rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
```
Risk: silent wrong output if T=3 is ever used (3-frame clips, debug runs).

---

## Open Risks (carry forward to Pass 5)

| # | Risk | Severity |
|---|------|----------|
| 1 | `torch.cross` without `dim=-1` — wrong output if T=3 | Low (typical T≠3) |
| 2 | Plücker convention vs SimpleAdapter weights unverified | Medium |
| 3 | Base model I2V pretraining used `full_ref` vs `y` — unknown | Medium |
| 4 | `accelerator.accumulate(*two_networks)` with one active — unverified under DDP | Low |
| 5 | DA3 intrinsic resolution estimate breaks if principal point is off-center | Low |
| 6 | GLD integration not implemented | Deferred |
| 7 | z-buffer occlusion not implemented | Deferred |