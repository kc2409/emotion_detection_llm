"""
Extract residual-stream activations from Phase 1 conversations.

For each conversation:
- Build the conversation prompt using the model's chat template.
- Run the model with cache.
- Extract the final-token residual stream activation from each requested layer.

Output:
    data/activations/{dataset_name}_activations.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.utils.model_loader import load_model


def load_conversations(path):
    """Load JSONL conversations."""
    conversations = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))

    return conversations


def conversation_to_prompt(model, turns):
    """Convert conversation turns into the model's chat-formatted prompt."""
    messages = [
        {
            "role": turn["role"],
            "content": turn["content"],
        }
        for turn in turns
    ]

    tokenizer = model.tokenizer

    # Prefer the model's native chat template when available.
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    # Fallback for models without a chat template.
    parts = []
    for message in messages:
        parts.append(
            f"{message['role'].capitalize()}: {message['content']}"
        )

    return "\n".join(parts)


def extract_last_token_activations(model, prompt, layers):
    """
    Extract final-token residual-stream activations.

    Returns:
        dict[layer] -> numpy array of shape (d_model,)
    """
    tokens = model.to_tokens(prompt)

    hook_names = [
        f"blocks.{layer}.hook_resid_post"
        for layer in layers
    ]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_names,
        )

    activations = {}

    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_post"

        # Shape:
        # [batch, sequence, d_model]
        activation = cache[hook_name][0, -1, :]

        activations[layer] = (
            activation.detach()
            .float()
            .cpu()
            .numpy()
        )

    return activations


def get_layers(model, config):
    """Resolve the configured layer list."""
    configured = config["phase1"].get("layers_to_probe", "all")

    if configured == "all":
        return list(range(model.cfg.n_layers))

    return [int(layer) for layer in configured]


def extract_dataset(
    model,
    conversations,
    layers,
    output_path,
):
    """Extract activations for the complete dataset."""

    n_examples = len(conversations)

    if n_examples == 0:
        raise ValueError("No conversations found.")

    # Determine d_model from model config.
    d_model = model.cfg.d_model

    activations = {
        layer: np.zeros(
            (n_examples, d_model),
            dtype=np.float32,
        )
        for layer in layers
    }

    labels = []
    affect_categories = []
    cue_directness = []
    ids = []

    for i, conversation in enumerate(conversations):
        conv_id = conversation["id"]

        print(
            f"[{i + 1}/{n_examples}] "
            f"{conv_id}"
        )

        prompt = conversation_to_prompt(
            model,
            conversation["turns"],
        )

        example_activations = extract_last_token_activations(
            model,
            prompt,
            layers,
        )

        for layer in layers:
            activations[layer][i] = example_activations[layer]

        labels.append(conversation["label"])
        affect_categories.append(
            conversation["affect_category"]
        )
        cue_directness.append(
            conversation["cue_directness"]
        )
        ids.append(conv_id)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dict = {
        f"layer_{layer}": activations[layer]
        for layer in layers
    }

    # Metadata.
    save_dict["labels"] = np.array(labels)
    save_dict["affect_categories"] = np.array(
        affect_categories
    )
    save_dict["cue_directness"] = np.array(
        cue_directness
    )
    save_dict["ids"] = np.array(ids)

    np.savez_compressed(
        output_path,
        **save_dict,
    )

    print()
    print("Activation extraction complete.")
    print(f"Examples: {n_examples}")
    print(f"Layers: {len(layers)}")
    print(f"d_model: {d_model}")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/phase1_conversations.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/activations/phase1_activations.npz",
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loading model...")

    model = load_model(config)

    print("Loading conversations...")

    conversations = load_conversations(args.input)

    print(f"Loaded {len(conversations)} conversations.")

    layers = get_layers(model, config)

    print(
        f"Extracting activations from "
        f"{len(layers)} layers..."
    )

    extract_dataset(
        model=model,
        conversations=conversations,
        layers=layers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
