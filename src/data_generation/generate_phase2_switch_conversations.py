"""
generate_phase2_switch_conversations.py — STUB.

Generates the Phase 2 dynamic-tracking dataset: conversations that start in
one affect state and then SWITCH (or resolve, or fail to resolve) partway
through, per config['phase2']['switch_conditions'].

Depends on generate_phase1_conversations.py being implemented first (reuses
its generator-call helper and conversation schema).

Output: data/raw/phase2_switch_conversations.jsonl
    {
        "id": "phase2_explicit_switch_0012",
        "switch_condition": "explicit_switch",
        "pre_switch_turns": [...],
        "switch_turn_index": 4,          # index into the full `turns` list
        "post_switch_turns": [...],
        "turns": [...],                   # pre + post, flattened, in order
        "pre_affect": "anxiety",
        "post_affect": "relief"           # or same affect for "unresolved"
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.utils.model_loader import load_config


def build_switch_prompt(switch_condition: str, pre_turns: int, post_turns: int) -> str:
    """
    TODO: Build the meta-prompt for one switch conversation.

      - explicit_switch:   user directly states the affect change
                            ("actually, I feel a lot calmer now").
      - implicit_switch:   change only inferable from content/tone shift,
                            never named.
      - resolved:           distress is explicitly resolved by the end.
      - unresolved:         distress persists or is deflected/ignored,
                            NOT resolved by the end.

      Must also instruct the generator to mark (out-of-band, e.g. via a
      separate metadata field, NOT inline in the conversation text) exactly
      which turn index the switch/resolution occurs at — this becomes
      `switch_turn_index`, which turn_tracking.py needs to compute
      update_latency.
    """
    raise NotImplementedError("build_switch_prompt: see TODOs above")


def generate_all(cfg: Dict[str, Any] | None = None) -> Path:
    """
    TODO: Mirror generate_phase1_conversations.generate_all(), iterating
    config['phase2']['switch_conditions'] x conversations_per_condition,
    writing to {paths.data_raw}/phase2_switch_conversations.jsonl.
    """
    if cfg is None:
        cfg = load_config()
    raise NotImplementedError("generate_all: see TODOs above")


if __name__ == "__main__":
    out_path = generate_all()
    print(f"Wrote Phase 2 switch conversations to {out_path}")
