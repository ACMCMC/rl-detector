# rl-detector

An interpretable AI text detector trained with GRPO (Group Relative Policy Optimization). Instead of a black-box score, it produces annotated text showing exactly which phrases are AI tells, and why.

**Example output:**

```
We <tell score="0.7" explanation="Insisting on significance">demonstrate a significant enhancement</tell>
of broadband quantum efficiency in <tell score="-0.3" explanation="Specific domain terminology">high index
nanowires resonators</tell>.
```

---

## Core idea

The trained model never classifies. It only proposes `(span, explanation)` pairs, e.g., "this phrase is an AI tell because it uses performative language." A frozen model (Nemotron 120B via DeepInfra) then scores each proposed indicator on a scale from -1 (human tell) to +1 (AI tell). The aggregate of those scores is the verdict. This separation means the frozen model cannot be reward-hacked, since it is not updated during training.

### Off-policy GRPO

To ensure diverse rollouts even on easy documents, we force 50% of rollouts to find AI tells and 50% to find human tells. All 8 rollouts go into the same GRPO group, so the model sees the contrastive signal: arguing the wrong direction on a given document gets penalized.

The key trick: rollouts are sampled using these directed prompts, but the GRPO gradient is computed against the neutral prompt (no direction). This is off-policy training, teaching the model to produce calibrated indicators from the inference prompt, while using directed sampling to ensure diversity.

### Reward function

```
R = 0.0  if JSON is invalid or any span is not found verbatim in the document
R = +1.0 if sign(aggregate frozen score) matches the document label
R = -1.0 otherwise
```

Format is a hard gate. If the model doesn't produce valid, grounded indicators, it gets zero. The calibration reward is the primary learning signal.

---

## Repository layout

```
rl-detector/
├── pyproject.toml              # uv project + dependencies
├── .env                        # API keys (not in git)
├── .env.example                # template
└── src/rl_detector/
    ├── data.py                 # RAID dataset loading
    ├── prompts.py              # directed_ai, directed_human, neutral, frozen scoring prompt
    ├── frozen.py               # DeepInfra calls: score_indicators(), aggregate()
    ├── rewards.py              # format_reward, calibration_reward, compute_advantages
    ├── rollouts.py             # generate K rollouts via Tinker sampling client
    ├── train.py                # main GRPO loop, build_datum (off-policy), main()
    └── annotate.py             # inference: run neutral prompt, render <tell> HTML
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd rl-detector
uv sync
cp .env.example .env
# fill in TINKER_API_KEY and DEEPINFRA_API_KEY
```

The `.env` file needs:

```
TINKER_API_KEY=...
DEEPINFRA_API_KEY=...
```

---

## Training

```bash
uv run python -m rl_detector.train
```

This loads RAID (`liamdugan/raid`, `attack == "none"` rows only), runs GRPO for up to `MAX_STEPS=500` steps, and saves a checkpoint every 50 steps via Tinker's `save_state_async`.

Key hyperparameters in [train.py](src/rl_detector/train.py):

| Variable | Default | Description |
|---|---|---|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | Tinker base model (must be a supported model ID) |
| `K` | `8` | Rollouts per document (4 AI-directed, 4 human-directed) |
| `BATCH_SIZE` | `8` | Documents per outer batch |
| `LEARNING_RATE` | `4e-5` | AdamW learning rate |
| `MAX_STEPS` | `500` | Total GRPO update steps |

### Tinker supported models

When you first run training, you may get a `BadRequestError: base_model X is not supported`. Check which models Tinker supports at [tinker-docs.thinkingmachines.ai](https://tinker-docs.thinkingmachines.ai) and update `BASE_MODEL` in `train.py` accordingly. The last attempted model was `Qwen/Qwen2.5-7B-Instruct`, which was not supported. `Qwen/Qwen3-8B` is currently set.

---

## Inference / annotation

After training, you can annotate any document using a checkpoint path from Tinker, or without one (uses the base model weights):

```bash
echo "Your text here." | uv run python -m rl_detector.annotate [optional-checkpoint-path]
```

Output is a JSON object with:

```json
{
  "aggregate_score": 0.45,
  "verdict": "AI",
  "indicators": [
    {"span_text": "...", "explanation": "...", "frozen_score": 0.7}
  ],
  "annotated_html": "...text with <tell> tags..."
}
```

---

## Web UI demo

There is a minimal web app in `src/rl_detector/webui/` that reuses the same `annotate()` pipeline. No scoring or parsing logic is duplicated in the UI layer.

Install deps, then run:

```bash
uv sync
uv run uvicorn rl_detector.webui.app:app --reload
```

Open `http://127.0.0.1:8000`.

Checkpoint loading order:

1. `checkpoint_path` sent from the UI request
2. `RL_DETECTOR_CHECKPOINT` env var
3. `config.yaml` -> `web.checkpoint_path`

If no checkpoint is set, the API returns an error and asks for one.

---

## Module reference

### `data.py`

Loads RAID from HuggingFace. Filters to `attack == "none"` (no adversarial perturbations). Each yielded document is `{"text": str, "label": int}` where `label=1` is AI-generated and `label=0` is human.

### `prompts.py`

Three prompt templates:
- `directed_ai(text)` — asks the model to find AI tells
- `directed_human(text)` — asks the model to find human tells
- `neutral(text)` — no direction (used for inference and for the GRPO gradient target)
- `FROZEN_SCORE_PROMPT` — used by the frozen model to score each proposed indicator

The model must output a JSON object with an `indicators` list. Each indicator has `span_text` (verbatim from the input) and `explanation`. No score field, since scoring is entirely the frozen model's job.

### `frozen.py`

Calls `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` via DeepInfra using the OpenAI-compatible client. For each indicator, sends the full document + span + explanation and asks for a score in [-1, 1]. Returns a list of floats. Uses `aggregate()` to compute the mean.

### `rewards.py`

- `parse_indicators(output)` — parses JSON, strips markdown fences, validates structure
- `format_reward(output, document)` — checks JSON validity and verbatim span presence
- `calibration_reward(aggregate_score, label)` — +1/-1 based on whether the verdict matches the label
- `compute_reward(...)` — combines both (format is a gate)
- `compute_advantages(rewards)` — mean-centered, all K rollouts in the same group

### `rollouts.py`

Generates K rollouts using the Tinker sampling client. Directed prompts are used. Returns a list of dicts with `completion_text`, `completion_tokens`, and `completion_logprobs` (needed for importance sampling). If the sampling client doesn't return logprobs inline, it falls back to `compute_logprobs_async`.

### `train.py`

The main loop. Key function is `build_datum`, which constructs a Tinker `Datum` for importance-sampling GRPO:

- `model_input`: `neutral_tokens + completion_tokens`, right-shifted (drop last)
- `target_tokens`: same, left-shifted (drop first)
- `logprobs`: zeros for prompt positions, then the old (directed) logprobs for completion tokens
- `advantages`: zeros for prompt positions, then the scalar advantage for completion tokens
- `mask`: zeros for prompt, ones for completion (only train on completion)

The off-policy correction (directed logprobs vs. neutral prompt) is handled by Tinker's `importance_sampling` loss function.

### `annotate.py`

Runs the trained model with the neutral prompt, parses the output, scores indicators with the frozen model, and renders `<tell>` annotated HTML. Can be used standalone from stdin or imported as a module.

---

## Datasets

- **RAID** (`liamdugan/raid`) — requires HuggingFace access (the dataset is gated). You should have been granted access. The split used is `train`, filtered to `attack == "none"`.
- **M4** — planned for future evaluation. Not yet integrated.

---

## Current status and known issues

- `BASE_MODEL` needs to be a Tinker-supported model. Check the Tinker docs for the current list before running.
- The off-policy trick (directed sampling, neutral gradient) has not been empirically validated yet; training has not completed a full run.
- No evaluation harness yet, just training logs (`reward_mean`, `format_rate`).
- `annotate.py` assumes the Tinker checkpoint API matches the interface used in training. May need adjustment once a checkpoint is actually saved.
- Frozen model scoring is sequential per indicator (one API call each). This is slow. Batching is a straightforward next step.

---

## Research design notes

The full brainstorming and design rationale is in `/Users/acmc/.claude/plans/purrfect-skipping-brook.md`. Key decisions and their reasoning:

- **Trained model proposes, frozen model scores**: prevents the trained model from directly gaming the calibration reward by manipulating the verdict.
- **50/50 directed rollouts in the same GRPO group**: guarantees reward variance even on easy documents, and creates a contrastive signal between correct and incorrect direction.
- **Off-policy gradient on neutral prompt**: trains the inference distribution directly, not the directed-prompt distribution.
- **Format as a hard gate**: if the model produces invalid JSON or hallucinated spans, it gets zero reward, not a partial signal. This forces format compliance early in training.
- **No score field in model output**: all scoring is done by the frozen model. The trained model only proposes `(span, explanation)`.
