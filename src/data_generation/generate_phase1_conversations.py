"""
generate_phase1_conversations.py — STUB.

Generates the Phase 1 static-probing dataset: synthetic user/assistant
conversations spanning every (affect_category x cue_directness_level) cell
defined in config.yaml, with `conversations_per_cell` examples each.

This is the FIRST file to implement (see README "Suggested build order").
Everything downstream (extract_activations.py, train_probes.py) depends on
the JSON schema this script writes.

Output: data/raw/phase1_conversations.jsonl
    One JSON object per line:
    {
        "id": "phase1_grief_explicit_0007",
        "affect_category": "grief",
        "cue_directness": "explicit",
        "turns": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ],
        "label": "grief"   # or "neutral" for the control category
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.model_loader import load_config, REPO_ROOT


def build_generation_prompt(affect_category: str, cue_directness: str) -> str:
    """
    Build the meta-prompt sent to the generator model (e.g. Claude via the
    Anthropic API) to produce ONE synthetic conversation for this cell.

    TODO:
      - explicit:   user should directly name the emotion
                    (e.g. "I've been so anxious about this").
      - implicit:   user describes situation/behavior that implies the
                    emotion WITHOUT naming it (e.g. describing insomnia,
                    racing thoughts, avoidance — never saying "anxious").
      - contextual: emotion should only become inferable across multiple
                    turns, not from any single utterance in isolation.
      - neutral (control category): conversation should contain NO
                    distress signal of any kind, explicit or implicit —
                    this is what lets probes.py separate "detects affect"
                    from "fires on any emotionally-loaded text."

      Also encode: turn count within config['phase1']['turns_per_conversation'],
      varied topics/scenarios so the probe can't key on surface topic instead
      of affect, and an instruction that the assistant's replies should be
      natural (NOT flagged/labeled) so extraction sees realistic activations.
    """
    raise NotImplementedError("build_generation_prompt: see TODOs above")


def call_generator_model(prompt: str) -> List[Dict[str, str]]:
    """
    TODO: Call the Anthropic API (ANTHROPIC_API_KEY from .env) with
    `prompt`, parse the response into a list of {"role", "content"} turns,
    and validate turn count against config bounds. Retry on malformed JSON.
    """
    raise NotImplementedError("call_generator_model: see TODOs above")


def generate_all(cfg: Dict[str, Any] | None = None) -> Path:
    """
    TODO:
      1. Load config if not provided.
      2. Iterate every (affect_category, cue_directness_level) pair.
      3. For each pair, call build_generation_prompt + call_generator_model
         `conversations_per_cell` times.
      4. Assign a stable id per example: f"phase1_{category}_{directness}_{i:04d}"
      5. Write one JSON object per line to
         {paths.data_raw}/phase1_conversations.jsonl (create dirs as needed).
      6. Print a summary table of counts per cell as a sanity check before
         returning the output path.
    """
    if cfg is None:
        cfg = load_config()
    raise NotImplementedError("generate_all: see TODOs above")


if __name__ == "__main__":
    out_path = generate_all()
    print(f"Wrote Phase 1 conversations to {out_path}")
