# TELL (Traced Evidence for Learned Labels)

TELL is a research prototype for interpretable AI text detection. Instead of returning only a class label, it highlights concrete spans in the input text and explains why each span is evidence for AI or human authorship.

The pipeline has two models:
- A trainable policy model that inserts `<tell explanation="...">...</tell>` tags.
- A frozen scoring model that adds `score="FLOAT"` to each tell and assigns values in `[-1, 1]`.

Final verdict is based on the mean score across tells.

## What TELL outputs

Given raw text, TELL returns:
- `aggregate_score`: mean tell score.
- `verdict`: `AI` if score > 0, else `Human`.
- `indicators`: extracted tells with explanation and score.
- `annotated_html`: original text with tell tags and score attributes.

Example tagged text:

```html
We <tell explanation="formulaic transition">therefore conclude</tell> that the method is effective.
```

After frozen scoring:

```html
We <tell explanation="formulaic transition" score="0.67">therefore conclude</tell> that the method is effective.
```

## Core approach

### 1) Structured evidence, not direct classification

The policy model does not directly predict AI vs human. It marks evidence spans and explanations inside the original text, preserving the full content.

### 2) Frozen calibration model

A separate frozen model scores each tell from `-1` to `1`.
- `+1`: strongest AI evidence.
- `0`: ambiguous.
- `-1`: strongest human evidence.

### 3) Reward design

Training uses a format gate plus calibration:
- If the output is not a faithful tagged reconstruction of the original text, reward is `0`.
- If format is valid, reward is a continuous value based on alignment between aggregate score and ground-truth label.

This creates pressure for both formatting fidelity and calibrated evidence.

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
  - Tinker (training and sampling)
  - DeepInfra (frozen tell scoring)

## Quick setup

```bash
git clone <repo-url>
cd rl-detector
uv sync
```

Set environment variables:

```bash
export TINKER_API_KEY=...
export DEEPINFRA_API_KEY=...
```

## Configuration

Main config is in `config.yaml`.

Important blocks:
- `model`: base model and LoRA rank.
- `training`: rollout count, docs per step, max steps, learning rate, checkpoint interval.
- `sampling`: generation params for Tinker sampling (`max_tokens`, `temperature`, `top_p`, `reasoning_effort`).
- `frozen`: frozen scorer model settings and seed.
- `web`: default checkpoint path and server host/port.

Current reproducibility seed used by the project is `2262`.

## Training

Run:

```bash
uv run python -m rl_detector.train
```

What happens:
- Loads train split from `Ateeqq/AI-and-Human-Generated-Text`.
- Builds balanced mini-batches per step (AI and human).
- Samples rollouts with directed prompts.
- Computes rewards from formatting + frozen-score calibration.
- Runs GRPO updates.
- Evaluates periodically on test docs.
- Saves checkpoints with Tinker state saving.

Audit logs are written to the path configured at `training.audit_log_path`.

## Annotation from CLI

Use stdin text and optional checkpoint path:

```bash
echo "Your text here." | uv run python -m rl_detector.annotate
```

With explicit checkpoint:

```bash
echo "Your text here." | uv run python -m rl_detector.annotate "tinker://<id>:train:0/sampler_weights/final"
```

## Web UI

Start server:

```bash
uv run uvicorn rl_detector.webui.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000`

Checkpoint source priority in web app:
1. `RL_DETECTOR_CHECKPOINT` env var.
2. `web.checkpoint_path` in `config.yaml`.

If engine is still loading, API returns `503` until warmup finishes.

## Checkpoint path notes

For inference sampling clients, checkpoint path must point to `sampler_weights`, not `weights`.

Good:

```text
tinker://.../sampler_weights/final
```

Bad:

```text
tinker://.../weights/final
```

If you see:

```text
BadRequestError: model_path must point to a sampler_weights checkpoint, got weights
```

update the checkpoint path to the `sampler_weights` variant.

## Prompting behavior

Policy model prompt requires exact text reproduction with inserted tell tags:

```text
<tell explanation="...">span text</tell>
```

Frozen scoring prompt then asks for nuanced continuous scores, adding:

```text
score="FLOAT"
```

inside each existing tell tag, while keeping text unchanged.

## API shape (web)

Main endpoints:
- `GET /api/config`
- `POST /api/analyze`
- `POST /api/analyze/start`
- `GET /api/analyze/status/{job_id}`

`POST /api/analyze` request body:

```json
{
  "text": "input document"
}
```

Response includes aggregate score, verdict, indicators, and segmented text for UI rendering.

## Development notes

- This repo is prototyping-oriented, and it favors fast iteration.
- Scoring extraction in the frozen stage parses all `<tell>` tags and reads `score` from each tag.
- Sampling reasoning effort is configured via `sampling.reasoning_effort` in YAML.

## License

No license file is currently included in this repository.
