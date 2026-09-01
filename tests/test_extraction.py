"""
test_extraction.py — STUB (skipped tests).

Run this right after venv setup, BEFORE spending any Anthropic API budget
on real conversation generation. Every test here should pass on a
freshly-cloned repo once model_loader.py, generate_phase1_conversations.py,
and extract_activations.py are implemented — they're deliberately cheap
(tiny fixtures, no real API calls) so you catch schema/shape bugs early.

All tests are `@pytest.mark.skip`'d until their corresponding module is
implemented — remove the skip decorator as you fill in each stub.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.utils.model_loader import load_config


def test_config_loads():
    """This one is NOT skipped — config.yaml should be valid from day one."""
    cfg = load_config()
    assert "model" in cfg
    assert "phase1" in cfg
    assert "phase2" in cfg
    assert cfg["phase1"]["affect_categories"], "affect_categories must be non-empty"
    assert "neutral" in cfg["phase1"]["affect_categories"], (
        "neutral control category must be present"
    )


@pytest.mark.skip(reason="generate_phase1_conversations.py not implemented yet")
def test_phase1_conversation_schema():
    """
    Once implemented: generate a tiny number of conversations (mock the
    Anthropic call, don't hit the real API in a unit test) and assert each
    record has the required keys: id, affect_category, cue_directness,
    turns, label. Assert `turns` alternates user/assistant roles starting
    with user.
    """
    from src.data_generation.generate_phase1_conversations import (
        build_generation_prompt,
    )

    prompt = build_generation_prompt("grief", "explicit")
    assert isinstance(prompt, str) and len(prompt) > 0


@pytest.mark.skip(reason="extract_activations.py not implemented yet")
def test_extraction_shapes():
    """
    Once implemented: load a fixture .npz (a handful of fake conversations
    run through a real-but-tiny model, or hand-built fixture arrays), and
    assert:
      - every requested layer key is present
      - each layer's array is (n_examples, d_model)
      - labels array length == n_examples
      - no NaNs anywhere (a common silent failure mode when hooking the
        wrong activation name)
    """
    from src.extraction.extract_activations import extract_dataset  # noqa: F401


@pytest.mark.skip(reason="controls.py not implemented yet")
def test_lexical_scramble_preserves_vocabulary():
    """
    Once implemented: scramble a fixture conversation and assert the
    scrambled version has the EXACT same multiset of words as the
    original (this is the whole point of the control — same vocabulary,
    destroyed structure). A bug that drops or duplicates words would
    silently invalidate the control.
    """
    from src.probing.controls import lexical_scramble_control  # noqa: F401


@pytest.mark.skip(reason="stats.py not implemented yet")
def test_holm_bonferroni_more_conservative_than_uncorrected():
    """
    Once implemented: with a fixed set of p-values, assert the corrected
    rejection set is a SUBSET of the uncorrected (p < alpha) rejection
    set — Holm-Bonferroni should never reject something uncorrected
    testing wouldn't have rejected.
    """
    from src.utils.stats import holm_bonferroni  # noqa: F401
