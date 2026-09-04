"""
train_probes.py

Trains one logistic-regression probe per layer on Phase 1 activations.

Output:
    results/phase1_probe_results.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.model_loader import load_config
from src.utils.stats import cohens_d, holm_bonferroni, permutation_test


def load_activations(
    npz_path: Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Load layer activations, labels, and IDs."""

    data = np.load(npz_path, allow_pickle=False)

    activations = {}

    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_")[1])
            activations[layer] = data[key].astype(np.float32)

    if not activations:
        raise ValueError(f"No layer activations found in {npz_path}")

    labels = data["labels"]
    ids = data["ids"]

    return activations, labels, ids


def train_one_layer_probe(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Train and cross-validate one logistic-regression probe."""

    probe_cfg = cfg["phase1"]["probe"]

    requested_folds = int(probe_cfg.get("cv_folds", 5))
    C = float(probe_cfg.get("C", 1.0))
    penalty = probe_cfg.get("penalty", "l2")
    class_weight = probe_cfg.get("class_weight", "balanced")
    seed = int(cfg["model"].get("seed", 42))

    _, class_counts = np.unique(y, return_counts=True)

    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(
            "At least 2 examples per class are required."
        )

    actual_folds = min(requested_folds, min_class_count)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
        ),
    )

    cv = StratifiedKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=seed,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    # Fit on all examples for the downstream effect-size calculation.
    model.fit(X, y)

    return {
        "cv_accuracy": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "cv_folds": actual_folds,
        "model": model,
    }


def compute_neutral_effect_size(
    model,
    X: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Cohen's d between affect and neutral model scores."""

    probabilities = model.predict_proba(X)
    classes = model.classes_

    neutral_indices = np.where(classes == "neutral")[0]

    if len(neutral_indices) == 0:
        return float("nan")

    neutral_idx = neutral_indices[0]

    affect_indices = [
        i for i, cls in enumerate(classes)
        if cls != "neutral"
    ]

    affect_score = probabilities[:, affect_indices].max(axis=1)
    neutral_score = probabilities[:, neutral_idx]

    affect_mask = labels != "neutral"
    neutral_mask = labels == "neutral"

    if affect_mask.sum() < 2 or neutral_mask.sum() < 2:
        return float("nan")

    return float(
        cohens_d(
            affect_score[affect_mask],
            neutral_score[neutral_mask],
        )
    )


def run_all_layers(
    activations_path: Path,
    cfg: Dict[str, Any] | None = None,
) -> Path:
    """Train probes for all layers and save results."""

    if cfg is None:
        cfg = load_config()

    activations, labels, ids = load_activations(
        activations_path
    )

    print(f"Loaded {len(labels)} examples.")
    print(f"Found {len(activations)} layers.")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    print("\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")

    results = []
    raw_p_values = []

    n_permutations = int(
        cfg["phase1"]["significance"].get(
            "n_permutations",
            1000,
        )
    )

    seed = int(cfg["model"].get("seed", 42))

    for layer in sorted(activations):

        print(f"\n=== Layer {layer} ===")

        X = activations[layer]

        probe_result = train_one_layer_probe(
            X,
            labels,
            cfg,
        )

        model = probe_result["model"]

        cv_accuracy = probe_result["cv_accuracy"]

        print(
            f"Balanced CV accuracy: "
            f"{cv_accuracy:.4f} "
            f"+/- {probe_result['cv_accuracy_std']:.4f}"
        )

        # Predictions from the model fitted on all examples.
        predictions = model.predict(X)

        def metric_fn(
            y_true,
            y_pred,
        ):
            return balanced_accuracy_score(
                y_true,
                y_pred,
            )

        observed_metric, p_value = permutation_test(
            labels,
            predictions,
            metric_fn=metric_fn,
            n_permutations=n_permutations,
            seed=seed,
        )

        effect_size = compute_neutral_effect_size(
            model,
            X,
            labels,
        )

        print(
            f"Observed balanced accuracy: "
            f"{observed_metric:.4f}"
        )
        print(
            f"Permutation p-value: "
            f"{p_value:.4f}"
        )

        if np.isnan(effect_size):
            print("Cohen's d vs neutral: NaN")
        else:
            print(
                f"Cohen's d vs neutral: "
                f"{effect_size:.4f}"
            )

        raw_p_values.append(p_value)

        results.append(
            {
                "layer": layer,
                "cv_accuracy": cv_accuracy,
                "cv_accuracy_std": probe_result[
                    "cv_accuracy_std"
                ],
                "observed_metric": observed_metric,
                "p_value": p_value,
                "cohens_d_vs_neutral": effect_size,
            }
        )

    # Holm-Bonferroni returns rejection flags.
    alpha = float(
        cfg["phase1"]["significance"].get(
            "alpha",
            0.05,
        )
    )

    significant = holm_bonferroni(
        raw_p_values,
        alpha=alpha,
    )

    # There is no corrected p-value returned by the existing helper.
    # We therefore preserve the raw p-value and record the corrected
    # significance decision.
    for result, is_significant in zip(
        results,
        significant,
    ):
        result["p_value_corrected"] = np.nan
        result["significant"] = bool(is_significant)

    output_path = Path(
        "results/phase1_probe_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Probe training complete.")
    print("=" * 60)

    print(df.to_string(index=False))

    print(
        f"\nWrote: {output_path}"
    )

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: "
            "python -m src.probing.train_probes "
            "<path_to_activations_npz>"
        )
        sys.exit(1)

    out_path = run_all_layers(
        Path(sys.argv[1])
    )

    print(f"Wrote probe results to {out_path}")
