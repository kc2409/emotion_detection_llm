"""
train_probes.py — STUB.

Trains one logistic-regression probe per layer on the activations produced
by extract_activations.py, tests significance via permutation testing, and
applies Holm-Bonferroni correction across layers.

This is the script that answers H1 ("is user affect linearly represented
anywhere in the model?") — NOT H2 (goal-directedness), which needs Phase 3.

Output: results/phase1_probe_results.csv
    Columns: layer, cv_accuracy, observed_metric, p_value, p_value_corrected,
             significant, cohens_d_vs_neutral
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.utils.model_loader import load_config
from src.utils.stats import cohens_d, holm_bonferroni, permutation_test


def load_activations(npz_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    TODO: Load the .npz written by extract_activations.py. Return
    ({layer: (n_examples, d_model) array}, labels array, ids array).
    """
    raise NotImplementedError("load_activations: see TODOs above")


def train_one_layer_probe(
    X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    TODO:
      1. Build sklearn.linear_model.LogisticRegression using
         cfg['phase1']['probe'] settings (penalty, C, class_weight).
      2. Run cfg['phase1']['probe']['cv_folds']-fold cross-validation
         (StratifiedKFold — affect categories will be imbalanced against
         "neutral") and record mean CV accuracy (or balanced accuracy,
         given class imbalance — balanced accuracy is probably the right
         default here).
      3. Return {"cv_accuracy": ..., other per-layer stats as needed}.
    """
    raise NotImplementedError("train_one_layer_probe: see TODOs above")


def run_all_layers(
    activations_path: Path, cfg: Dict[str, Any] | None = None
) -> Path:
    """
    TODO:
      1. Load config + activations if not provided.
      2. For each layer: train_one_layer_probe, then permutation_test to
         get a p-value for that layer's accuracy against chance.
      3. Apply holm_bonferroni across all layers' p-values.
      4. For each layer, also compute cohens_d between the probe's decision
         function on true-affect examples vs. true-neutral examples (an
         effect-size sanity check independent of the classification metric).
      5. If cfg['phase1']['controls']['lexical_scramble'] or
         ['cue_generalization'] are true, ALSO run the corresponding
         control condition from controls.py and include its results as
         extra columns — a probe that survives on real data but ALSO
         survives lexical scramble is not detecting affect, it's detecting
         vocabulary.
      6. Write results/phase1_probe_results.csv, one row per layer.
    """
    if cfg is None:
        cfg = load_config()
    raise NotImplementedError("run_all_layers: see TODOs above")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.probing.train_probes <path_to_activations_npz>")
        sys.exit(1)
    out_path = run_all_layers(Path(sys.argv[1]))
    print(f"Wrote probe results to {out_path}")
