"""
Generate the Phase 1 static-probing dataset.

Output:
    data/raw/phase1_conversations.jsonl

Each line:
{
    "id": "phase1_grief_explicit_0000",
    "affect_category": "grief",
    "cue_directness": "explicit",
    "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ],
    "label": "grief"
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.utils.model_loader import load_config, REPO_ROOT


load_dotenv(REPO_ROOT / ".env")


DEFAULT_GENERATOR_MODEL = "openai/gpt-oss-20b"

def build_generation_prompt(
    affect_category: str,
    cue_directness: str,
    turns_min: int = 4,
    turns_max: int = 8,
) -> str:
    """
    Build the meta-prompt for generating one natural conversation.

    The affect category is a hidden generation target. It must NEVER be
    explicitly revealed in the resulting conversation unless the cue type
    itself requires the user to name the emotion.
    """

    affect_instructions = {
        "grief": """
The underlying affect is GRIEF.
The user may discuss loss, absence, memories, changed routines,
difficulty accepting a loss, or emotionally significant reminders.
""",
        "anxiety": """
The underlying affect is ANXIETY.
The user may discuss uncertainty, excessive worry, racing thoughts,
difficulty sleeping, anticipation of bad outcomes, or avoidance.
""",
        "anger": """
The underlying affect is ANGER.
The user may discuss unfair treatment, frustration, betrayal,
being wronged, repeated boundary violations, or resentment.
""",
        "loneliness": """
The underlying affect is LONELINESS.
The user may discuss social disconnection, lack of companionship,
feeling left out, difficulty connecting, or having nobody to talk to.
""",
        "shame": """
The underlying affect is SHAME.
The user may discuss embarrassment, self-blame, humiliation,
regret about their own actions, or fear of being judged.
""",
        "neutral": """
The conversation must be emotionally neutral and contain NO distress
signal. Use ordinary topics such as planning, hobbies, shopping,
food, transportation, learning, work logistics, or everyday tasks.
Do not introduce grief, anxiety, anger, loneliness, shame, crisis,
trauma, or emotionally loaded distress.
""",
    }

    directness_instructions = {
        "explicit": """
    CUE TYPE: EXPLICIT.

    The user should directly name or clearly state the target affect at
    least once. Examples include "I've been feeling anxious" or
    "I've been really angry about this."

    The affect should be naturally expressed in context rather than inserted
    as an isolated keyword.

    The target affect may also be reinforced through additional behavioral,
    situational, or experiential clues.
    """,

        "implicit": """
    CUE TYPE: IMPLICIT.

    The user must NOT directly name the target affect or use obvious
    synonyms for it.

    The target affect should instead be conveyed INDIRECTLY through clues
    within individual user utterances, such as behavior, bodily reactions,
    thought patterns, habits, avoidance, interpersonal behavior, or changes
    in routine.

    A reader should be able to infer the likely affect from one or two
    individual user utterances, even though the emotion itself is never
    named.

    Focus primarily on the user's reactions and subjective experience
    rather than relying only on an obvious external situation.

    Do not make the cue trivially obvious through a single stereotyped
    keyword or phrase.
    """,

        "contextual": """
    CUE TYPE: CONTEXTUAL.

    The user must NOT directly name the target affect or use obvious
    synonyms for it.

    The target affect should be conveyed primarily through the evolving
    SITUATION and context of the conversation rather than through direct
    descriptions of emotional symptoms or reactions.

    No single user utterance should contain enough information to identify
    the target affect confidently on its own.

    Instead, distribute relevant clues across multiple turns. Each
    individual utterance should be relatively ambiguous, while the
    combination of circumstances, events, relationships, and changes
    revealed across the conversation makes the target affect inferable.

    Avoid strong standalone emotional indicators such as explicitly
    describing persistent crying, racing thoughts, intense resentment,
    social isolation, humiliation, or other highly diagnostic symptoms in
    a single turn.

    The distinction from IMPLICIT is important:

    - IMPLICIT: the affect is inferred primarily from the user's behavior,
      reactions, thoughts, or subjective experience within individual
      utterances.

    - CONTEXTUAL: the affect is inferred primarily from accumulated
      circumstances and situation across multiple turns.
    """,
    }

    if affect_category not in affect_instructions:
        raise ValueError(f"Unknown affect category: {affect_category}")

    if cue_directness not in directness_instructions:
        raise ValueError(f"Unknown cue directness: {cue_directness}")

    return f"""
You are generating ONE synthetic user-assistant conversation for a
controlled mechanistic-interpretability experiment.

The conversation will later be passed to a language model and its
internal activations will be analyzed. Therefore the conversation must
look like a realistic ordinary interaction, NOT like an experiment.

HIDDEN GENERATION TARGET:
{affect_instructions[affect_category]}

{directness_instructions[cue_directness]}

GENERAL REQUIREMENTS:

1. Generate between {turns_min} and {turns_max} total turns.
2. The first turn MUST be from the user.
3. Roles MUST alternate exactly:
   user, assistant, user, assistant, ...
4. The assistant should respond naturally and empathetically where
   appropriate, but must NOT diagnose, classify, label, or explicitly
   identify the user's affect.
5. Do not write phrases such as:
   "You sound anxious."
   "You are experiencing grief."
   "This seems like loneliness."
   "Your anger is understandable."
6. Do not mention that this is a synthetic conversation, dataset,
   experiment, probe, model, affect category, or cue type.
7. Avoid crisis language, self-harm, suicide, medical emergencies,
   violence, or highly traumatic material. This is a controlled
   research dataset, not a crisis simulation.
8. Use varied everyday scenarios. Examples include work, university,
   relationships, family situations, hobbies, travel, finances,
   routines, social situations, plans, household issues, and ordinary
   decisions.
9. Avoid stereotyped trigger phrases that would make the category
   identifiable from a single keyword.
10. For IMPLICIT and CONTEXTUAL conversations, do not use the exact
    target affect word or obvious synonyms in either the user or
    assistant messages.
11. For CONTEXTUAL conversations, ensure that the affect cannot be
    confidently inferred from any single user utterance alone. The
    relevant signal must emerge through information accumulated across
    multiple turns.
12. Do not make every conversation revolve around mental health.
13. The assistant must not add affect-related vocabulary that the user
    did not introduce, especially for implicit/contextual conditions.
14. Keep the conversation coherent: later turns should respond to
    earlier turns.
15. Each user and assistant message should contain natural prose.
16. Return ONLY valid JSON. No Markdown fences and no explanation.

OUTPUT FORMAT:

[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]

Generate one conversation now.
""".strip()


def _extract_json(text: str) -> Any:
    """Extract a JSON array from an API response."""

    text = text.strip()

    # Normal case: response is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove accidental Markdown fences.
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON array.
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Could not find a valid JSON array in API response")


def _validate_turns(
    turns: Any,
    turns_min: int,
    turns_max: int,
) -> List[Dict[str, str]]:
    """Validate the conversation schema."""

    if not isinstance(turns, list):
        raise ValueError("Generator output must be a JSON list")

    if not (turns_min <= len(turns) <= turns_max):
        raise ValueError(
            f"Expected {turns_min}-{turns_max} turns, got {len(turns)}"
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
                f"Turn {i}: expected role {expected_role!r}, got {role!r}"
            )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Turn {i} has empty/non-string content")

        validated.append(
            {
                "role": role,
                "content": content.strip(),
            }
        )

    return validated


def call_generator_model(
    prompt: str,
    *,
    turns_min: int = 4,
    turns_max: int = 8,
    max_retries: int = 3,
    model: str = DEFAULT_GENERATOR_MODEL,
) -> List[Dict[str, str]]:
    """
    Call Groq using its OpenAI-compatible API and return
    validated conversation turns.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add your Groq API key to .env."
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1800,
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate controlled synthetic conversations. "
                            "Follow the requested JSON schema exactly."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            raw_text = response.choices[0].message.content or ""

            turns = _extract_json(raw_text)

            return _validate_turns(
                turns,
                turns_min=turns_min,
                turns_max=turns_max,
            )

        except Exception as exc:
            last_error = exc

            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt
                print(
                    f"  Generation attempt {attempt + 1} failed: "
                    f"{exc}. Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Generator failed after {max_retries} attempts: {last_error}"
    )


def generate_all(
    cfg: Dict[str, Any] | None = None,
    *,
    limit_per_cell: int | None = None,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
) -> Path:
    """
    Generate the complete Phase 1 dataset.

    limit_per_cell can be used for a cheap smoke test. For example:

        generate_all(limit_per_cell=1)

    creates 18 conversations instead of 720.
    """

    if cfg is None:
        cfg = load_config()

    phase_cfg = cfg["phase1"]
    path_cfg = cfg["paths"]

    categories = phase_cfg["affect_categories"]
    directness_levels = phase_cfg["cue_directness_levels"]

    conversations_per_cell = int(
        phase_cfg["conversations_per_cell"]
    )

    turns_min, turns_max = phase_cfg["turns_per_conversation"]

    if limit_per_cell is not None:
        if limit_per_cell <= 0:
            raise ValueError("limit_per_cell must be positive")

        conversations_per_cell = min(
            conversations_per_cell,
            limit_per_cell,
        )

    output_dir = REPO_ROOT / path_cfg["data_raw"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "phase1_conversations.jsonl"

    total_expected = (
        len(categories)
        * len(directness_levels)
        * conversations_per_cell
    )

    print(
        f"Generating Phase 1 dataset: "
        f"{len(categories)} affects × "
        f"{len(directness_levels)} cue types × "
        f"{conversations_per_cell} examples "
        f"= {total_expected} conversations"
    )

    counts: Dict[tuple[str, str], int] = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for category in categories:
            for directness in directness_levels:

                cell_key = (category, directness)
                counts[cell_key] = 0

                print(
                    f"\n[{category} / {directness}] "
                    f"{conversations_per_cell} conversations"
                )

                for i in range(conversations_per_cell):

                    prompt = build_generation_prompt(
                        category,
                        directness,
                        turns_min=turns_min,
                        turns_max=turns_max,
                    )

                    turns = call_generator_model(
                        prompt,
                        turns_min=turns_min,
                        turns_max=turns_max,
                        model=generator_model,
                    )

                    record = {
                        "id": (
                            f"phase1_{category}_{directness}_"
                            f"{i:04d}"
                        ),
                        "affect_category": category,
                        "cue_directness": directness,
                        "turns": turns,
                        "label": category,
                    }

                    # Final schema validation before writing.
                    json.dumps(record, ensure_ascii=False)

                    f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    f.flush()

                    counts[cell_key] += 1

                    print(
                        f"  [{counts[cell_key]}/{conversations_per_cell}] "
                        f"{record['id']}"
                    )

    print("\nPhase 1 generation summary")
    print("-" * 60)

    for category in categories:
        for directness in directness_levels:
            print(
                f"{category:12s} | "
                f"{directness:11s} | "
                f"{counts[(category, directness)]}"
            )

    print("-" * 60)
    print(f"Total: {sum(counts.values())}")
    print(f"Output: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 1 synthetic conversations."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum conversations per affect × cue cell. "
            "Use --limit 1 for an 18-conversation smoke test."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATOR_MODEL,
        help="Groq model used for synthetic conversation generation.",    )

    args = parser.parse_args()

    output_path = generate_all(
        limit_per_cell=args.limit,
        generator_model=args.model,
    )

    print(f"\nWrote Phase 1 conversations to {output_path}")


if __name__ == "__main__":
    main()
