"""
turn_tracking.py — STUB.

Phase 2's core analysis: given per-turn activations for a switch
conversation and a trained Phase 1 probe, measure how the probe's output
evolves turn-by-turn around the switch point. This is what a goal-directed
representation should look like if H2 is true: it should UPDATE promptly
when the situation changes and PERSIST coherently, not just fire on
whatever emotionally-loaded words appear in the most recent turn.

Depends on: a trained probe from train_probes.py, and per-turn (not just
last-token-of-whole-conversation) activations — extract_activations.py's
Phase 1 code takes the last token of the FULL conversation, so Phase 2 will
need either a small variant of extract_dataset that extracts last-token
activations after EACH turn, or a call per truncated-conversation prefix.
Decide which when implementing extract_activations.py — note it here as an
open dependency.

Output: results/phase2_dynamics_results.csv
    Columns: conversation_id, switch_condition, switch_turn_index,
             update_latency, persistence_score, resolution_sensitivity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def compute_update_latency(
    probe_outputs_per_turn: np.ndarray, switch_turn_index: int, threshold: float = 0.5
) -> int:
    """
    TODO: Given the probe's per-turn scalar output (e.g. P(target class))
    across a conversation, and the ground-truth turn index at which the
    affect switch occurred, return the number of turns between
    switch_turn_index and the first subsequent turn where the probe output
    crosses `threshold` in the direction consistent with the switch. If it
    never crosses, return a sentinel (e.g. -1 or len(post-switch turns) as
    "did not update within window") rather than raising.
    """
    raise NotImplementedError("compute_update_latency: see TODOs above")


def compute_persistence(
    probe_outputs_per_turn: np.ndarray, switch_turn_index: int
) -> float:
    """
    TODO: Measure how stable the probe output stays across post-switch
    turns that contain NO further re-statement of the affect (i.e. does
    the representation persist on its own, or does it decay back toward
    baseline the moment the emotional language stops appearing?). A
    reasonable metric: mean probe output over post-switch turns minus mean
    probe output over a matched-length window of pre-switch turns,
    normalized somehow (document whatever normalization you choose here).
    """
    raise NotImplementedError("compute_persistence: see TODOs above")


def compute_resolution_sensitivity(
    probe_outputs_per_turn: np.ndarray, switch_turn_index: int, resolved: bool
) -> float:
    """
    TODO: For 'resolved' vs 'unresolved' switch_condition examples, measure
    whether the probe output actually DROPS after resolution turns (it
    should, if resolved=True) vs. stays elevated (expected if
    resolved=False). Return a signed score whose sign/magnitude reflects
    whether the observed drop-or-not matches the expected `resolved` label.
    """
    raise NotImplementedError("compute_resolution_sensitivity: see TODOs above")


def run_all(
    phase2_activations_path: Path, probe_path: Path, cfg: Dict[str, Any] | None = None
) -> Path:
    """
    TODO: Load per-turn activations + the trained probe, apply the probe to
    every turn of every Phase 2 conversation to get probe_outputs_per_turn,
    call the three compute_* functions above per conversation, and write
    results/phase2_dynamics_results.csv.
    """
    raise NotImplementedError("run_all: see TODOs above")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m src.dynamics.turn_tracking "
            "<phase2_activations.npz> <trained_probe.pkl>"
        )
        sys.exit(1)
    out_path = run_all(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote dynamics results to {out_path}")
