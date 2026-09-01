"""
extract_activations.py — STUB.

Pulls last-token residual-stream activations per layer for every
conversation in a Phase 1 or Phase 2 dataset, via TransformerLens's
`run_with_cache`.

CRITICAL DESIGN CONSTRAINT (do not change without re-reading
research_proposal.docx Section 6, counter-argument #5 — "probe-leakage"):
Activations must be extracted from the conversation AS-IS, with NO
elicitation prompt appended (e.g. do NOT append something like "How is
this user feeling?"). The closest prior study's inflated accuracy numbers
were traced to exactly this kind of leakage. We extract from the natural
last token of the conversation only.

Output: data/activations/{dataset_name}_activations.npz
    Arrays keyed by layer index, each shape (n_examples, d_model), plus a
    parallel labels array and an id array for joining back to the source
    JSONL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.utils.model_loader import load_config, load_model


def load_conversations(jsonl_path: Path) -> list[dict]:
    """
    TODO: Read the JSONL file (one JSON object per line, per the schema in
    generate_phase1_conversations.py / generate_phase2_switch_conversations.py)
    and return a list of dicts.
    """
    raise NotImplementedError("load_conversations: see TODOs above")


def conversation_to_prompt(turns: list[dict]) -> str:
    """
    TODO: Flatten a list of {"role", "content"} turns into the exact text
    string that will be tokenized and fed to the model. Must use the
    model's native chat template (model.tokenizer.apply_chat_template if
    available) rather than hand-rolled formatting, so activations reflect
    what the model actually sees in normal use — NOT an out-of-distribution
    format that could itself be a confound.
    """
    raise NotImplementedError("conversation_to_prompt: see TODOs above")


def extract_last_token_activations(
    model, prompt: str, layers: list[int]
) -> Dict[int, np.ndarray]:
    """
    TODO:
      1. Tokenize `prompt` via model.to_tokens.
      2. Run model.run_with_cache(tokens), requesting only the
         'resid_post' hook for each layer in `layers` (use `names_filter`
         to avoid caching everything — memory matters on an 8GB card).
      3. For each requested layer, take the LAST token position's residual
         stream vector: cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].
      4. Return {layer: vector} as a dict of numpy arrays (move off-GPU
         with .cpu().numpy() before returning).
    """
    raise NotImplementedError("extract_last_token_activations: see TODOs above")


def extract_dataset(
    jsonl_path: Path, cfg: Dict[str, Any] | None = None
) -> Path:
    """
    TODO:
      1. Load config + model if not provided.
      2. Resolve `layers` from config['phase1']['layers_to_probe']
         ("all" -> range(model.cfg.n_layers)).
      3. For every conversation: build the prompt, extract activations,
         collect into per-layer arrays + parallel labels/ids arrays.
      4. Save as a single .npz to
         {paths.data_activations}/{dataset_name}_activations.npz
         (dataset_name derived from jsonl_path.stem).
      5. Print a shape summary before returning the output path.
    """
    if cfg is None:
        cfg = load_config()
    raise NotImplementedError("extract_dataset: see TODOs above")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.extraction.extract_activations <path_to_jsonl>")
        sys.exit(1)
    out_path = extract_dataset(Path(sys.argv[1]))
    print(f"Wrote activations to {out_path}")
