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
    Non-parametric permutation test.

    Labels are shuffled while scores remain fixed to construct
    the null distribution.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    observed = float(metric_fn(labels, scores))

    null_values = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):
        rng = np.random.default_rng(seed + i)
        shuffled_labels = rng.permutation(labels)
        null_values[i] = metric_fn(shuffled_labels, scores)

    # Add-one correction prevents p=0.
    p_value = (
        1.0 + np.sum(null_values >= observed)
    ) / (n_permutations + 1.0)

    return observed, float(p_value)


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Holm-Bonferroni step-down multiple-comparisons correction.

    Returns a boolean array indicating which hypotheses are rejected.
    """
    p_values = np.asarray(p_values, dtype=float)

    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")

    if len(p_values) == 0:
        return np.array([], dtype=bool)

    if np.any(~np.isfinite(p_values)):
        raise ValueError("p_values must contain only finite values")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be between 0 and 1")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    m = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    rejected_sorted = np.zeros(m, dtype=bool)

    # Holm step-down:
    # p_(i) <= alpha / (m-i)
    # with i zero-indexed.
    for i, p in enumerate(sorted_p):
        threshold = alpha / (m - i)

        if p <= threshold:
            rejected_sorted[i] = True
        else:
            # Once a hypothesis fails, all subsequent hypotheses fail.
            break

    rejected = np.zeros(m, dtype=bool)
    rejected[order] = rejected_sorted

    return rejected


def cohens_d(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    """
    Cohen's d using the standard pooled sample standard deviation.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("groups must be one-dimensional")

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "Each group must contain at least two observations"
        )

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_variance = (
        ((len(a) - 1) * var_a)
        + ((len(b) - 1) * var_b)
    ) / (len(a) + len(b) - 2)

    pooled_sd = np.sqrt(pooled_variance)

    if pooled_sd == 0:
        if mean_a == mean_b:
            return 0.0
        return float(np.sign(mean_a - mean_b) * np.inf)

    return float((mean_a - mean_b) / pooled_sd)