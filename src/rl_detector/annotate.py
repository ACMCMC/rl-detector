"""Inference: annotate a document with <tell> spans using a trained checkpoint."""

import asyncio
import json
import os
import re

import tinker
from dotenv import load_dotenv
from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.frozen import aggregate, get_client, rank_indicators
from rl_detector.prompts import neutral
from rl_detector.rewards import _extract_final_channel, parse_indicators

BASE_MODEL = CFG.model.base_model

load_dotenv()


def render_html(document: str, indicators: list[dict], frozen_scores: list[float]) -> str:
    """Add frozen score attribute to existing <tell> tags in the model output."""
    # indicators already came from the model's tagged output; re-inject frozen scores
    scored = sorted(
        zip(indicators, frozen_scores),
        key=lambda x: document.find(x[0]["span_text"]),
    )
    result = document
    offset = 0
    for ind, score in scored:
        span = ind["span_text"]
        explanation = ind["explanation"].replace('"', "&quot;")
        tag = f'<tell score="{score:.2f}" explanation="{explanation}">{span}</tell>'
        pos = result.find(span, offset)
        if pos == -1:
            continue
        result = result[:pos] + tag + result[pos + len(span):]
        offset = pos + len(tag)
    return result


async def annotate(document: str, checkpoint_path: str | None = None) -> dict:
    service_client = tinker.ServiceClient()
    if checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            checkpoint_path=checkpoint_path
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL, rank=32
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    prompt_tokens = tokenizer.encode(neutral(document))
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampled = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=512, temperature=0.0),
    )
    output = tokenizer.decode(sampled.sequences[0].tokens)

    indicators = parse_indicators(output) or []
    frozen_client = get_client()
    frozen_scored = await rank_indicators(frozen_client, output, indicators) if indicators else []
    agg = aggregate(frozen_scored)
    scores = [s["score"] for s in frozen_scored]
    html = render_html(document, indicators, scores)

    return {
        "aggregate_score": agg,
        "verdict": "AI" if agg > 0 else "Human",
        "indicators": [
            {**ind, "frozen_score": fs["score"]}
            for ind, fs in zip(indicators, frozen_scored)
        ],
        "annotated_html": html,
    }


if __name__ == "__main__":
    import sys
    text = sys.stdin.read()
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(annotate(text, checkpoint))
    print(json.dumps(result, indent=2))
