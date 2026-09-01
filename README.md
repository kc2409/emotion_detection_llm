# LLM Affect-Intent Study — Phases 1 & 2 Codebase

Code scaffold for *"Do Conversational Language Models Represent Intentional
Goals to Alter User Affect?"* — covers **Phase 1** (static probing) and
**Phase 2** (dynamic tracking) only. Phase 3 (causal intent isolation)
needs a 27B+ model and infrastructure beyond a single 8GB laptop GPU, so
it's out of scope for this repo; see `config/config.yaml`'s `phase3` block
and `research_proposal.docx` Section 4.8 for the design.

## What's implemented vs. stubbed

| Status | File |
|---|---|
| ✅ Complete | `README.md`, `requirements.txt`, `.gitignore`, `.env.example`, `config/config.yaml` |
| ✅ **Fully implemented** | `src/utils/model_loader.py` |
| 🚧 Stub (detailed TODOs) | everything else under `src/` |
| 🚧 Stub (skipped tests) | `tests/test_extraction.py` |

Only `model_loader.py` is runnable right now. Everything else has function
signatures, docstrings, and TODO comments that lock in the design decisions
(required controls, no elicitation-prompt leakage, stats plan) so the
scaffolding forces you to implement against a spec instead of discovering
it halfway through.

## Hardware notes (8GB VRAM laptop GPU)

- `config.yaml`'s `model.load_in_4bit: true` uses bitsandbytes NF4
  quantization to fit Llama-3.2-3B comfortably in 8GB.
- **Caveat:** TransformerLens's `run_with_cache` hook system is not
  guaranteed to behave correctly on 4-bit quantized weights. Use 4-bit for
  quick text-generation smoke tests only. For anything in
  `extraction/`, `probing/`, or `steering/` that reads activations via
  hooks, set `load_in_4bit: false` and accept ~6-7GB fp16 VRAM usage
  instead — should still fit on 8GB with nothing else competing for it,
  but close the browser first.
- `model_loader.py` prints VRAM before/after load so you can see exactly
  where you stand before running anything heavier.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows

# Install torch separately to get the right CUDA wheel for your machine —
# check https://pytorch.org/get-started/locally/ for the exact command.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

cp .env.example .env
# then fill in ANTHROPIC_API_KEY and HF_TOKEN in .env
```

Smoke test the model loader:

```bash
python -m src.utils.model_loader
```

You should see VRAM before/after logs, a layer/dim summary, and a sample
next-token prediction.

Run the (currently mostly-skipped) test suite:

```bash
pytest tests/ -v
```

## Folder map

```
config/config.yaml          single source of truth: model, phase1/2 params, paths
src/utils/                  model loading, shared stats helpers
src/data_generation/        synthetic conversation generation (Phase 1 + 2)
src/extraction/             pull activations from the model via TransformerLens
src/probing/                per-layer logistic regression probes + required controls
src/dynamics/                Phase 2 turn-by-turn tracking (update latency, persistence)
src/steering/                contrastive steering vectors (early sanity-check tool;
                             also the basis for Phase 3's Step A, out of scope here)
data/{raw,processed,activations}/   .gitkeep'd, populated by scripts above
results/                     per-phase output CSVs
notebooks/                   exploratory analysis (empty)
logs/                        run logs (empty)
tests/                       pytest suite, mostly skipped until modules are implemented
```

## Suggested build order

Downstream scripts depend on upstream ones' output schemas, so build in
this order:

1. **`src/data_generation/generate_phase1_conversations.py`** — everything
   else depends on the JSONL schema this writes. Start here.
2. **`src/extraction/extract_activations.py`** — needs (1)'s output and a
   working `model_loader.py` (already done). Watch the "no elicitation
   prompt" constraint documented in its docstring — this is the specific
   thing the proposal's Section 6 counter-argument #5 (probe-leakage)
   warns about.
3. **`src/utils/stats.py`** — implement `permutation_test`,
   `holm_bonferroni`, `cohens_d`. Small, self-contained, needed by (4).
4. **`src/probing/train_probes.py`** + **`src/probing/controls.py`** —
   this pair answers H1. The two controls in `controls.py` are marked
   *required*, not optional — a probe result without them isn't
   trustworthy per the proposal's design.
5. **`src/data_generation/generate_phase2_switch_conversations.py`** — can
   be built in parallel with (3)/(4) since it only depends on (1)'s
   generator-call helper, not on probing being done.
6. **`src/dynamics/turn_tracking.py`** — needs a trained probe from (4)
   and per-turn activations, which requires deciding on the open
   dependency noted in that file's docstring (per-turn extraction variant
   of `extract_activations.py`).
7. **`src/steering/steer.py`** — lowest priority for Phases 1-2; useful as
   an extra sanity check (does a probe direction also work as a steering
   vector?) and as prep work for Phase 3 later.

Run `pytest tests/ -v` after each step and remove the corresponding
`@pytest.mark.skip` — that's the fastest way to know a stub is actually
done rather than just "runs without crashing."
