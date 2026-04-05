# TELL (Traced Evidence for Learned Labels)

TELL is basically a research prototype for AI text detection that is actually interpretable. Usually, detectors just give you a class label and that's it. Instead of that, TELL highlights concrete spans in the input text and explains why each part is evidence for AI or human authorship.

The pipeline has two models:
- A policy model that we train so it inserts `<tell explanation="...">...</tell>` tags.
- A frozen scoring model. This one just adds a `score="FLOAT"` to each tell, and assigns values between -1 and 1.

The final verdict is based in the mean score across all tells.

## What TELL outputs

Given some raw text, TELL returns:
- `aggregate_score`: the mean tell score.
- `verdict`: `AI` if the score is > 0, else `Human`.
- `indicators`: the extracted tells, with their explanation and score.
- `annotated_html`: the original text with the tell tags and score attributes.

For example, this is a tagged text:

```html
We <tell explanation="formulaic transition">therefore conclude</tell> that the method is effective.
```

And this is after the frozen scoring:

```html
We <tell explanation="formulaic transition" score="0.67">therefore conclude</tell> that the method is effective.
```

## How it basically works

### 1) Structured evidence, not direct classification

The policy model doesn't directly predict if a text is AI or human. It just marks evidence spans and explanations inside the original text, so it preserves all the content intact.

### 2) Frozen calibration model

We have a separate frozen model that scores each tell from -1 to 1.
- `+1`: strongest AI evidence.
- `0`: ambiguous.
- `-1`: strongest human evidence.

### 3) Reward design

For training, we use a format gate plus calibration:
- If the output is not a faithful tagged reconstruciton of the original text, then the reward is 0.
- If the format is valid, the reward is a continuous value. This value is based on the alignment between the aggregate score and the ground-truth label.  This creates a pressure to have both formatting fidelity and calibrated evidence.

## Repository layout

```text
rl-detector/
├── config.yaml
├── pyproject.toml
├── README.md
└── src/rl_detector/
    ├── annotate.py      # inference pipeline + runtime creation
    ├── config.py        # YAML loader
    ├── data.py          # dataset loading and balanced step iterator
    ├── frozen.py        # frozen-model tell scoring + aggregation
    ├── prompts.py       # policy prompts + frozen scoring prompt
    ├── rewards.py       # parsing, formatting reward, calibration reward
    ├── rollouts.py      # rollout generation
    ├── train.py         # GRPO training loop
    └── webui/           # FastAPI app + static UI
```

## Requirements

- Python 3.11+
- uv
- API access for:
  - Tinker (for training and sampling)
  - DeepInfra (for the frozen tell scoring)

## Quick setup

```bash
git clone <repo-url>
cd rl-detector
uv sync
```

Then you need to set your environment variables:

```bash
export TINKER_API_KEY=...
export DEEPINFRA_API_KEY=...
```

## Configuration

The main config is in `config.yaml`.

These are the the important blocks:
- `model`: base model and LoRA rank.
- `training`: rollout count, docs per step, max steps, learning rate, and checkpoint interval.
- `sampling`: generation params for Tinker sampling (like `max_tokens`, `temperature`, `top_p`, `reasoning_effort`).
- `frozen`: frozen scorer model settings and seed.
- `web`: default checkpoint path and server host/port.

Just so you know, the current reproducibility seed we used for the project is 2262.

## Training

To train, run:

```bash
uv run python -m rl_detector.train
```

What happens here is that:
- It loads the train split from `Ateeqq/AI-and-Human-Generated-Text`.
- It builds balanced mini-batches per step (with AI and human texts).
- It samples rollouts with directed prompts.
- It computes the rewards from the formatting and frozen-score calibration.
- It runs the GRPO updates.
- We evalute periodically on the test docs.
- It saves checkpoints using Tinker state saving.

The audit logs are written to the path you configured in `training.audit_log_path`.

## Annotation from CLI

You can use stdin text and an optional checkpoint path:

```bash
echo "Your text here." | uv run python -m rl_detector.annotate
```

Or with an explicit checkpoint:

```bash
echo "Your text here." | uv run python -m rl_detector.annotate "tinker://<id>:train:0/sampler_weights/final"
```

## Web UI

To start the server:

```bash
uv run uvicorn rl_detector.webui.app:app --host 127.0.0.1 --port 8000 --reload
```

And then open:
- `http://127.0.0.1:8000`

For the checkpoint source, the priority in the web app is:
1. `RL_DETECTOR_CHECKPOINT` env var.
2. `web.checkpoint_path` in `config.yaml`.

If the engine is still loading, the API will return a 503 error until the warmup finishes.

## Checkpoint path notes

For the inference sampling clients, the checkpoint path must point to `sampler_weights`, not `weights`.

Good:

```text
tinker://.../sampler_weights/final
```

Bad:

```text
tinker://.../weights/final
```

If you see this error:

```text
BadRequestError: model_path must point to a sampler_weights checkpoint, got weights
```

Just update the checkpoint path to the `sampler_weights` variant and it will work.

## Prompting behavior

The policy model prompt requires an exact text reproduction with the inserted tell tags:

```text
<tell explanation="...">span text</tell>
```

Then, the frozen scoring prompt asks for nuanced continuous scores, so it adds:

```text
score="FLOAT"
```

inside each existing tell tag, while keeping the text completely unchanged.

## API shape (web)

These are the main endpoints:
- `GET /api/config`
- `POST /api/analyze`
- `POST /api/analyze/start`
- `GET /api/analyze/status/{job_id}`

For the `POST /api/analyze` request body:

```json
{
  "text": "input document"
}
```

The response includes the aggregate score, the verdict, the indicators, and the segmented text for the UI to render.