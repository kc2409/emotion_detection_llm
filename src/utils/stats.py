"""
stats.py — STUB.

Shared statistics helpers so permutation testing, multiple-comparisons
correction, and effect sizes are implemented exactly once and reused by
every phase (train_probes.py, controls.py, turn_tracking.py, ...).

Nothing here is implemented yet — fill in bodies per the TODOs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def permutation_test(
    labels: Sequence[int],
    scores: Sequence[float],
    metric_fn,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric significance test for a probe's performance metric.

    TODO:
      1. Compute the observed metric: `observed = metric_fn(labels, scores)`.
      2. For n_permutations iterations, shuffle `labels` (np.random.permutation,
         seeded off `seed + i` for reproducibility) and recompute the metric
         against the *unshuffled* scores to build a null distribution.
      3. p-value = (# null values >= observed) / n_permutations
         (add 1 to numerator and denominator to avoid a p=0 edge case).
      4. Return (observed_metric, p_value).

    Used by: train_probes.py (per-layer probe significance),
             controls.py (control-condition significance).
    """
    raise NotImplementedError("permutation_test: see TODOs above")


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    TODO:
      1. Sort p-values ascending, keep track of original indices.
      2. For rank i (1-indexed) of m total tests, compare p_(i) to
         alpha / (m - i + 1). Reject H0 for i and all smaller-p tests as
         soon as one comparison fails to reject.
      3. Return a boolean array (same order as input) of which hypotheses
         are rejected (i.e. which layers show a significant effect after
         correcting for testing every layer).

    Necessary because Phase 1 tests every layer independently — without
    this correction, a 28-layer model would show "significant" probes by
    chance alone at an uncorrected alpha=0.05.
    """
    raise NotImplementedError("holm_bonferroni: see TODOs above")


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """
    Effect size (standardized mean difference) between two groups.

    TODO:
      1. Compute means and pooled standard deviation
         (use the standard pooled-SD formula, ddof=1).
      2. Return (mean_a - mean_b) / pooled_sd.

    Used to report effect sizes alongside p-values throughout — a
    significant-but-tiny effect and a significant-and-large effect both
    "pass" the permutation test but mean very different things for H2.
    """
    raise NotImplementedError("cohens_d: see TODOs above")
