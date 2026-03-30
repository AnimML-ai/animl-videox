# Session Intent

Adapt Wan2.2-Fun-Camera-Control to accept GLD-derived latents as a cross-attention
conditioning signal for real estate walkthrough video generation.

## Scope
1. **GLD latent extraction** — extract and format latents from GLD's DA3/VGGT latent
   space into a tensor compatible with Wan2.2's cross-attention interface.
2. **Wan2.2 architecture modifications** — add a cross-attention conditioning path
   that ingests GLD latents in both the forward (inference) and training passes.
   Camera pose conditioning (Plücker) is out of scope for this session.

## Inputs
- GLD latents: novel-view features from DA3/VGGT latent space, derived from sparse
  listing photos.
- Wan2.2-Fun-Camera-Control model code as baseline.

## Output
- Modified Wan2.2 model that accepts GLD latents via cross-attention in both
  inference and training paths.
- GLD latent extraction utility that produces tensors in the format expected
  by the new conditioning interface.

## Explicit non-goals
- Plücker embedding / camera pose conditioning changes.
- Training loop, loss, or data pipeline changes.
- Integration with Captain Safari memory bank.
