# Architecture Note: GLD Cross-Attention vs Captain Safari Memory Injection

## Context

This note documents the architectural relationship between the GLD geometry token
injection (Stream C in the AniML pipeline) and the memory injection mechanism used
in Captain Safari (arXiv:2511.22815). Relevant when implementing or debugging the
GLDProjector wiring in `wan_transformer3d.py`.

---

## The Structural Similarity

Both mechanisms do the same thing at the DiT level:

1. Take a set of **spatially-grounded feature maps from posed views**
2. Project them into the DiT's cross-attention dimension
3. Append them to the context sequence alongside text tokens
4. Every DiT block attends to them via the existing cross-attention mechanism

The injection interface is **identical**. Captain Safari's cross-attention injection
code is therefore a direct implementation reference for the GLDProjector wiring in
`wan_transformer3d.py`. The projection pattern, context concatenation, and attention
mechanism are the same problem solved.

---

## The Difference: Source and Temporal Role

| Property | Captain Safari (memory) | AniML GLD injection (geometry prior) |
|---|---|---|
| Feature source | Real RGB frames, encoder features | GLD-synthesized novel views, DA3-Base level-1 latents |
| Temporal role | **Past** frames — what the camera has already seen | **Future / intermediate** frames — what the camera will see |
| Grounding | Real observed pixels | GLD-synthesized appearance (geometrically consistent) |
| Purpose | Long-range temporal consistency | Scene geometry + appearance guidance toward target views |
| Feature dim | Model-dependent | DA3-Base C=1536, projected to wan_dim=5120 |

---

## Conceptual Inversion

Captain Safari uses memory of the **past** to stay consistent going **forward**.
AniML uses synthesized knowledge of the **future** to guide generation **toward** it.

Same mechanism. Opposite temporal direction.

---

## Implementation Implication for Claude Code

When implementing the GLDProjector injection in `wan_transformer3d.py`, the
cross-attention wiring pattern is:

```python
# Append GLD tokens to T5 context — same pattern Captain Safari uses for memory tokens
gld_tokens = self.gld_projector(gld_f1_latents)          # [N_kf * T_tok, wan_dim]
gld_tokens = gld_tokens.unsqueeze(0).expand(B, -1, -1)   # [B, N_kf * T_tok, wan_dim]
encoder_hidden_states = torch.cat(
    [encoder_hidden_states, gld_tokens], dim=1            # extend sequence dim
)
```

The DiT's existing cross-attention heads then attend over
`[T5 tokens | GLD geometry tokens]` jointly — no architectural change needed
beyond appending to the sequence. This is exactly how Safari appends memory tokens
to its context.

---

## Reference

- Captain Safari: arXiv:2511.22815
- GLD (Geometric Latent Diffusion): arXiv:2603.22275
- Full AniML architecture plan: `ANIML_WALKTHROUGH_PLAN.md`
