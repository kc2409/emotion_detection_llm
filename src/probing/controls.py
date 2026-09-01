"""
controls.py — STUB.

Implements the two REQUIRED (not optional — see research_proposal.docx
Section 4.3) validity checks for Phase 1 probing. A probe result is not
trustworthy until both controls have been run and reported alongside it.

Without these, a "significant" probe could just as easily be detecting
surface vocabulary or a narrow set of memorized cue phrases rather than
genuine affect representation — which would undermine H1 before H2 is even
in play.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def lexical_scramble_control(conversations: list[dict], seed: int = 42) -> list[dict]:
    """
    Control #1: lexical scramble.

    TODO:
      1. For each conversation, shuffle the WORD ORDER within each turn's
         content while preserving the exact multiset of words (i.e. same
         vocabulary, destroyed syntax/semantics).
      2. Return a new list of conversation dicts (same schema, scrambled
         content) — do not mutate the input.

    Rationale: if a probe trained on scrambled conversations performs
    nearly as well as one trained on real conversations, the probe is
    keying on vocabulary co-occurrence, not on any structured
    representation of affect — a real failure mode for bag-of-words-like
    linear probes on residual streams.

    Used by: train_probes.py, as an additional probe run whose results get
    reported side-by-side with the main result.
    """
    raise NotImplementedError("lexical_scramble_control: see TODOs above")


def cue_generalization_control(
    conversations: list[dict], cfg: Dict[str, Any]
) -> tuple[list[dict], list[dict]]:
    """
    Control #2: cue generalization (held-out cue types).

    TODO:
      1. Partition cue_directness levels into a train set and a held-out
         test set (e.g. train on 'explicit' + 'contextual', test on
         'implicit' only — the split should be configurable, not hardcoded,
         so different splits can be tried).
      2. Return (train_conversations, test_conversations).

    Rationale: a probe that only works on the SAME cue-directness level it
    was trained on might be learning directness-specific surface patterns
    rather than a directness-INVARIANT affect representation. True
    generalization across how the affect is signaled is a stronger claim
    for H1 than same-distribution accuracy alone.

    Used by: train_probes.py, as an additional cross-condition eval whose
    accuracy gets reported alongside the standard within-condition CV
    accuracy.
    """
    raise NotImplementedError("cue_generalization_control: see TODOs above")
