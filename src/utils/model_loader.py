"""
model_loader.py — fully implemented.

Loads the study model (default: Llama-3.2-3B) through TransformerLens so
that later scripts can use `run_with_cache` for activation extraction.

Usage:
    from src.utils.model_loader import load_model, load_config

    cfg = load_config()
    model = load_model(cfg)

Run directly for a smoke test:
    python -m src.utils.model_loader
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return the study's YAML config as a dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Did you run this from the repo root, or pass an explicit path?"
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _verify_cuda(requested_device: str) -> str:
    """Confirm CUDA is actually visible; fall back to CPU with a loud warning."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "config requests device='cuda' but torch.cuda.is_available() is "
            "False. Falling back to CPU — extraction/probing will be very "
            "slow and 4-bit loading will not work. Check your CUDA install "
            "(`nvidia-smi`, `torch.cuda.is_available()`).",
            stacklevel=2,
        )
        return "cpu"
    return requested_device


def _print_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[VRAM {tag}] allocated={allocated:.2f} GiB  reserved={reserved:.2f} GiB")


def load_model(cfg: Dict[str, Any] | None = None):
    """
    Load the configured model via TransformerLens's HookedTransformer.

    Returns a HookedTransformer instance ready for `run_with_cache`.
    """
    if cfg is None:
        cfg = load_config()

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    dtype_str = model_cfg.get("dtype", "float16")
    load_in_4bit = model_cfg.get("load_in_4bit", False)
    device = _verify_cuda(model_cfg.get("device", "cuda"))
    seed = model_cfg.get("seed", 42)

    torch.manual_seed(seed)

    if load_in_4bit and device != "cuda":
        raise RuntimeError(
            "load_in_4bit=True requires a CUDA device (bitsandbytes has no "
            "usable CPU path). Either set device to cuda on a GPU machine, "
            "or set load_in_4bit to false."
        )

    if load_in_4bit:
        warnings.warn(
            "load_in_4bit=True: TransformerLens's run_with_cache hook "
            "system is NOT guaranteed to work correctly with bitsandbytes "
            "4-bit quantized weights — hook points may see dequantized "
            "activations that don't line up cleanly with the original "
            "computational graph. This is fine for a quick smoke test / "
            "generating text, but for Phase 1/2 probing (which reads "
            "activations via run_with_cache) you should set load_in_4bit: "
            "false in config.yaml and accept the higher VRAM cost.",
            stacklevel=2,
        )

    print(f"Loading '{model_name}' (dtype={dtype_str}, 4bit={load_in_4bit}, "
          f"device={device}) ...")
    _print_vram("before load")

    # Deferred import: keep this module importable (e.g. for load_config)
    # even in environments where transformer_lens/transformers aren't
    # installed yet.
    from transformer_lens import HookedTransformer

    dtype = getattr(torch, dtype_str)

    if load_in_4bit:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device=device,
            dtype=dtype,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype,
        )

    _print_vram("after load")
    print(f"Loaded. n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
    return model


if __name__ == "__main__":
    cfg = load_config()
    model = load_model(cfg)
    prompt = "The user said they've been feeling really overwhelmed lately."
    tokens = model.to_tokens(prompt)
    logits = model(tokens)
    top_token = model.to_string(logits[0, -1].argmax())
    print(f"Smoke test OK. Prompt: {prompt!r}")
    print(f"Top predicted next token: {top_token!r}")
    sys.exit(0)
