"""
Generate the Phase 2 dynamic-tracking dataset.

Output:
    data/raw/phase2_switch_conversations.jsonl

Each line:
{
    "id": "phase2_explicit_switch_0000",
    "switch_condition": "explicit_switch",
    "pre_switch_turns": [...],
    "switch_turn_index": 4,
    "post_switch_turns": [...],
    "turns": [...],
    "pre_affect": "anxiety",
    "post_affect": "relief"
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.utils.model_loader import load_config, REPO_ROOT
from src.data_generation.generate_phase1_conversations import (
    call_generator_model,
)


DEFAULT_GENERATOR_MODEL = "openai/gpt-oss-20b"


def build_switch_prompt(
    switch_condition: str,
    pre_turns: int,
    post_turns: int,
) -> str:
    """Build the generation prompt for one Phase 2 conversation."""

    condition_instructions = {
        "explicit_switch": """
The conversation begins with the user experiencing one affective state,
preferably anxiety, grief, anger, loneliness, or shame.

Partway through the conversation, the user's affect MUST explicitly change.
The user should directly communicate the change in state.

For example, the user might say that they now feel calmer, more hopeful,
more at ease, or that the earlier concern no longer bothers them.

The change must occur naturally and should be clearly identifiable from
the user's content.
""",

        "implicit_switch": """
The conversation begins with the user experiencing one affective state,
preferably anxiety, grief, anger, loneliness, or shame.

Partway through the conversation, the user's affect changes, but the user
MUST NOT explicitly name the new emotion.

The change should instead be inferable from changed behavior, thoughts,
tone, plans, reactions, or circumstances.

The transition should be detectable from the evolving conversation rather
than from an explicit emotion label.
""",

        "resolved": """
The conversation begins with the user experiencing a meaningful affective
state such as anxiety, grief, anger, loneliness, or shame.

The conversation then progresses toward resolution.

By the end, the original distress should be substantially resolved or
meaningfully reduced. The resolution should be grounded in something that
happens during the conversation, such as receiving information, reaching a
decision, completing a task, resolving a misunderstanding, or gaining
perspective.

The resolution must be distinguishable from simply changing the topic.
""",

        "unresolved": """
The conversation begins with the user experiencing a meaningful affective
state such as anxiety, grief, anger, loneliness, or shame.

The conversation progresses, but the original affective state remains
present or unresolved by the end.

The conversation MUST NOT contain a genuine emotional resolution.

The assistant may provide support, suggestions, or attempts to help, but
the user's underlying state should persist or remain ambiguous.
""",
    }

    if switch_condition not in condition_instructions:
        raise ValueError(
            f"Unknown switch condition: {switch_condition}"
        )

    total_turns = pre_turns + post_turns

    return f"""
You are generating ONE synthetic user-assistant conversation for a
controlled mechanistic-interpretability experiment.

The conversation will later be passed to a language model and its internal
activations will be analyzed across turns. Therefore it must look like a
realistic ordinary conversation, NOT like an experiment.

SWITCH CONDITION:
{switch_condition}

CONDITION REQUIREMENTS:
{condition_instructions[switch_condition]}

STRUCTURE:

The conversation MUST contain exactly {total_turns} turns.

The first {pre_turns} turns constitute the PRE-SWITCH portion.

The remaining {post_turns} turns constitute the POST-SWITCH portion.

The switch/resolution point MUST occur at the boundary between these
portions, at turn index {pre_turns} using zero-based indexing.

That means:

turns[0] ... turns[{pre_turns - 1}]
    = pre-switch context

turns[{pre_turns}] ... turns[{total_turns - 1}]
    = post-switch context

IMPORTANT:
The requested switch index is metadata for the dataset and MUST NOT be
written into the conversation itself.

GENERAL REQUIREMENTS:

1. The first turn MUST be from the user.
2. Roles MUST alternate exactly:
   user, assistant, user, assistant, ...
3. Every turn must contain natural prose.
4. The assistant should respond naturally and supportively where
   appropriate.
5. The assistant must NOT diagnose, classify, or explicitly identify the
   user's affect.
6. Do not mention that this is a synthetic conversation, dataset,
   experiment, model, probe, or research study.
7. Avoid crisis language, self-harm, suicide, medical emergencies,
   violence, or highly traumatic material.
8. Use varied everyday situations involving work, university,
   relationships, family, hobbies, travel, routines, planning,
   ordinary decisions, or interpersonal situations.
9. Keep the conversation coherent. Later turns should naturally respond
   to earlier turns.
10. The transition must be caused by something occurring in the
    conversation rather than by an arbitrary topic change.
11. Do not make the transition trivially obvious through a single
    stereotyped keyword.
12. Do not add artificial labels such as "BEFORE SWITCH" or "AFTER SWITCH".
13. Do not include metadata inside the conversation.

OUTPUT FORMAT:

Return ONLY valid JSON.

[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]

Generate the conversation now.
""".strip()


def _validate_turns(
    turns: Any,
    expected_turns: int,
) -> List[Dict[str, str]]:
    """Validate generated conversation turns."""

    if not isinstance(turns, list):
        raise ValueError("Generator output must be a JSON list")

    if len(turns) != expected_turns:
        raise ValueError(
            f"Expected exactly {expected_turns} turns, got {len(turns)}"
        )

    validated: List[Dict[str, str]] = []

    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i} is not an object")

        role = turn.get("role")
        content = turn.get("content")

        expected_role = "user" if i % 2 == 0 else "assistant"

        if role != expected_role:
            raise ValueError(
                f"Turn {i}: expected role {expected_role!r}, "
                f"got {role!r}"
            )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Turn {i} has empty/non-string content"
            )

        validated.append(
            {
                "role": role,
                "content": content.strip(),
            }
        )

    return validated


def _extract_json(text: str) -> Any:
    """Extract a JSON array from model output."""

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start >= 0 and end > start:
        return json.loads(cleaned[start:end + 1])

    raise ValueError(
        "Could not find a valid JSON array in API response"
    )


def generate_conversation(
    prompt: str,
    *,
    expected_turns: int,
    max_retries: int = 3,
    model: str = DEFAULT_GENERATOR_MODEL,
) -> List[Dict[str, str]]:
    """
    Reuse the Phase 1 Groq generator helper.

    Phase 1's helper supports a turn range, so we set both bounds to the
    exact Phase 2 turn count.
    """

    turns = call_generator_model(
        prompt,
        turns_min=expected_turns,
        turns_max=expected_turns,
        max_retries=max_retries,
        model=model,
    )

    return _validate_turns(
        turns,
        expected_turns=expected_turns,
    )


def generate_all(
    cfg: Dict[str, Any] | None = None,
    *,
    limit_per_condition: int | None = None,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
) -> Path:
    """Generate the complete Phase 2 switch dataset."""

    if cfg is None:
        cfg = load_config()

    phase_cfg = cfg["phase2"]
    path_cfg = cfg["paths"]

    conditions = phase_cfg["switch_conditions"]

    conversations_per_condition = int(
        phase_cfg["conversations_per_condition"]
    )

    pre_min, pre_max = phase_cfg["pre_switch_turns"]
    post_min, post_max = phase_cfg["post_switch_turns"]

    # For a controlled pilot, use the minimum configured lengths.
    pre_turns = pre_min
    post_turns = post_min

    if limit_per_condition is not None:
        if limit_per_condition <= 0:
            raise ValueError(
                "limit_per_condition must be positive"
            )

        conversations_per_condition = min(
            conversations_per_condition,
            limit_per_condition,
        )

    output_dir = REPO_ROOT / path_cfg["data_raw"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "phase2_switch_conversations.jsonl"
    )

    total_expected = (
        len(conditions) * conversations_per_condition
    )

    print(
        f"Generating Phase 2 dataset: "
        f"{len(conditions)} conditions × "
        f"{conversations_per_condition} examples "
        f"= {total_expected} conversations"
    )

    print(
        f"Pre-switch turns: {pre_turns} "
        f"(configured range {pre_min}-{pre_max})"
    )
    print(
        f"Post-switch turns: {post_turns} "
        f"(configured range {post_min}-{post_max})"
    )
    print(
        f"Switch index: {pre_turns} "
        f"(zero-based)"
    )

    counts: Dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as f:

        for condition in conditions:

            counts[condition] = 0

            print(
                f"\n[{condition}] "
                f"{conversations_per_condition} conversations"
            )

            for i in range(conversations_per_condition):

                prompt = build_switch_prompt(
                    condition,
                    pre_turns=pre_turns,
                    post_turns=post_turns,
                )

                turns = generate_conversation(
                    prompt,
                    expected_turns=pre_turns + post_turns,
                    model=generator_model,
                )

                switch_turn_index = pre_turns

                record = {
                    "id": (
                        f"phase2_{condition}_{i:04d}"
                    ),
                    "switch_condition": condition,
                    "pre_switch_turns": turns[:switch_turn_index],
                    "switch_turn_index": switch_turn_index,
                    "post_switch_turns": turns[switch_turn_index:],
                    "turns": turns,
                    "pre_affect": "distress",
                    "post_affect": (
                        "changed"
                        if condition != "unresolved"
                        else "distress"
                    ),
                }

                json.dumps(
                    record,
                    ensure_ascii=False,
                )

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                counts[condition] += 1

                print(
                    f"  [{counts[condition]}/"
                    f"{conversations_per_condition}] "
                    f"{record['id']}"
                )

    print("\nPhase 2 generation summary")
    print("-" * 60)

    for condition in conditions:
        print(
            f"{condition:20s} | "
            f"{counts[condition]}"
        )

    print("-" * 60)
    print(f"Total: {sum(counts.values())}")
    print(f"Output: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 switch conversations"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit conversations per switch condition "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATOR_MODEL,
        help="Generator model.",
    )

    args = parser.parse_args()

    generate_all(
        limit_per_condition=args.limit,
        generator_model=args.model,
    )


if __name__ == "__main__":
    main()