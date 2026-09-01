"""
steer.py — STUB.

Steering-vector construction and injection. Not needed for Phase 1/2, but
scaffolded now since the contrastive-direction step of Phase 3
(research_proposal.docx Section 4.8, Step A) builds directly on this, and
it's useful earlier as a sanity-check tool: if a Phase 1 probe direction
ALSO works as a steering vector (i.e. adding it measurably shifts generated
text toward the target affect), that's independent evidence the direction
is doing real representational work, not just linearly separable noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_steering_vector(
    positive_activations: np.ndarray, negative_activations: np.ndarray
) -> np.ndarray:
    """
    TODO: Compute a contrastive direction as
    mean(positive_activations) - mean(negative_activations) across the
    example axis. This is the "diff-of-means" construction used in most
    activation-steering literature (simpler and more robust than using raw
    probe weight vectors, which can be poorly scaled).

    Returns a (d_model,) vector, NOT yet normalized — normalization choice
    is left to the caller since target norm depends on the injection scale
    used at generation time.
    """
    raise NotImplementedError("build_steering_vector: see TODOs above")


def make_steering_hook(vector: np.ndarray, layer: int, coefficient: float):
    """
    TODO: Return a TransformerLens-compatible hook function
    `hook_fn(activation, hook)` that adds `coefficient * vector` to the
    residual stream at every token position at the given layer, for use
    with `model.run_with_hooks(..., fwd_hooks=[(hook_name, hook_fn)])`
    where hook_name = f"blocks.{layer}.hook_resid_post".

    Convert `vector` to a torch tensor matching the activation's
    device/dtype inside the hook (don't assume it's already on-device).
    """
    raise NotImplementedError("make_steering_hook: see TODOs above")


def generate_with_steering(
    model: Any, prompt: str, vector: np.ndarray, layer: int, coefficient: float,
    max_new_tokens: int = 60,
) -> str:
    """
    TODO: Tokenize prompt, register the steering hook from
    make_steering_hook via model.run_with_hooks during generation
    (model.generate doesn't take fwd_hooks directly in TransformerLens —
    wrap the sampling loop, or use model.generate inside a
    `with model.hooks(fwd_hooks=[...]):` context manager), decode and
    return the generated continuation as a string.
    """
    raise NotImplementedError("generate_with_steering: see TODOs above")
