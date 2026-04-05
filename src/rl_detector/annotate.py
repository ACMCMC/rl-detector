"""Inference: annotate a document with <tell> spans using a trained checkpoint."""

import asyncio
import json
import inspect
import logging

import tinker
from dotenv import load_dotenv
from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.frozen import aggregate, get_client, rank_indicators
from rl_detector.prompts import neutral
from rl_detector.rewards import parse_indicators

BASE_MODEL = CFG.model.base_model
logger = logging.getLogger(__name__)

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


async def _emit_progress(progress_cb, pct: int, stage: str) -> None:
    if progress_cb is None:
        return
    maybe = progress_cb(pct, stage)
    if inspect.isawaitable(maybe):
        await maybe


async def _await_with_heartbeat(coro, phase: str, every_s: int = 10):
    """Log periodic liveness updates while awaiting potentially slow remote calls."""
    task = asyncio.create_task(coro)
    waited = 0
    while True:
        done, _ = await asyncio.wait({task}, timeout=every_s)
        if done:
            logger.info("runtime | %s complete (%ds)", phase, waited)
            return await task
        waited += every_s
        logger.info("runtime | still waiting on %s (%ds elapsed)", phase, waited)


async def create_runtime(checkpoint_path: str | None = None) -> dict:
    """Create reusable inference runtime objects for a checkpoint."""
    logger.info("runtime | create_runtime start")
    service_client = tinker.ServiceClient()
    if checkpoint_path:
        logger.info("runtime | creating sampling client directly from checkpoint")
        try:
            sampling_client = await _await_with_heartbeat(
                service_client.create_sampling_client_async(model_path=checkpoint_path),
                phase="create_sampling_client_async(model_path)",
            )
        except tinker.BadRequestError as e:
            msg = str(e)
            retry_path = checkpoint_path.replace("/weights/", "/sampler_weights/")
            if "sampler_weights" in msg and retry_path != checkpoint_path:
                logger.info("runtime | retrying sampling client with sampler_weights path")
                try:
                    sampling_client = await _await_with_heartbeat(
                        service_client.create_sampling_client_async(model_path=retry_path),
                        phase="create_sampling_client_async(model_path=sampler_weights)",
                    )
                except tinker.NotFoundError:
                    logger.info("runtime | sampler_weights missing after retry, generating from training weights")
                    weights_path = retry_path.replace("/sampler_weights/", "/weights/")
                    training_client = await _await_with_heartbeat(
                        service_client.create_training_client_from_state_async(path=weights_path),
                        phase="create_training_client_from_state_async(for_sampler_generation)",
                    )
                    ckpt_name = retry_path.rsplit("/", 1)[-1]
                    save_future = await training_client.save_weights_for_sampler_async(name=ckpt_name, ttl_seconds=None)
                    save_resp = await _await_with_heartbeat(
                        save_future.result_async(),
                        phase="save_weights_for_sampler_async.result_async",
                    )
                    generated_path = getattr(save_resp, "path", None) or retry_path
                    logger.info("runtime | generated sampler_weights path: %s", generated_path)
                    sampling_client = await _await_with_heartbeat(
                        service_client.create_sampling_client_async(model_path=generated_path),
                        phase="create_sampling_client_async(model_path=generated_sampler)",
                    )
            else:
                raise
        except tinker.NotFoundError as e:
            # If sampler_weights are missing, generate them from matching training weights.
            if "/sampler_weights/" not in checkpoint_path:
                raise
            logger.info("runtime | sampler_weights missing, attempting to generate from training weights")
            weights_path = checkpoint_path.replace("/sampler_weights/", "/weights/")
            training_client = await _await_with_heartbeat(
                service_client.create_training_client_from_state_async(path=weights_path),
                phase="create_training_client_from_state_async(for_sampler_generation)",
            )
            ckpt_name = checkpoint_path.rsplit("/", 1)[-1]
            save_future = await training_client.save_weights_for_sampler_async(name=ckpt_name, ttl_seconds=None)
            save_resp = await _await_with_heartbeat(
                save_future.result_async(),
                phase="save_weights_for_sampler_async.result_async",
            )
            generated_path = getattr(save_resp, "path", None) or checkpoint_path
            logger.info("runtime | generated sampler_weights path: %s", generated_path)
            sampling_client = await _await_with_heartbeat(
                service_client.create_sampling_client_async(model_path=generated_path),
                phase="create_sampling_client_async(model_path=generated_sampler)",
            )
    else:
        logger.info("runtime | creating base sampling client")
        sampling_client = await _await_with_heartbeat(
            service_client.create_sampling_client_async(base_model=BASE_MODEL),
            phase="create_sampling_client_async(base_model)",
        )

    logger.info("runtime | loading tokenizer for base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    logger.info("runtime | tokenizer ready")
    logger.info("runtime | sampling client ready")
    return {
        "checkpoint_path": checkpoint_path,
        "tokenizer": tokenizer,
        "sampling_client": sampling_client,
    }


async def annotate_with_runtime(document: str, runtime: dict, progress_cb=None) -> dict:
    """Run one-rollout annotation pipeline with a preloaded runtime."""
    await _emit_progress(progress_cb, 30, "Preparing prompt")
    tokenizer = runtime["tokenizer"]
    sampling_client = runtime["sampling_client"]

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": neutral(document)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(prompt_text)
    logger.info("annotate | sampling model output")
    await _emit_progress(progress_cb, 45, "Sampling one rollout")
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampled = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=CFG.frozen.seed,
            reasoning_effort=CFG.sampling.reasoning_effort,
        ),
    )
    output = tokenizer.decode(sampled.sequences[0].tokens)

    indicators = parse_indicators(output) or []
    logger.info("annotate | parsed %d indicators", len(indicators))
    await _emit_progress(progress_cb, 65, "Parsed indicators")
    frozen_client = get_client()
    logger.info("annotate | ranking indicators with frozen model")
    await _emit_progress(progress_cb, 80, "Scoring indicators")
    frozen_scored = await rank_indicators(frozen_client, output, indicators) if indicators else []
    agg = aggregate(frozen_scored)
    scores = [s["score"] for s in frozen_scored]
    html = render_html(document, indicators, scores)
    logger.info("annotate | complete, aggregate_score=%.3f", agg)
    await _emit_progress(progress_cb, 100, "Complete")

    return {
        "aggregate_score": agg,
        "verdict": "AI" if agg > 0 else "Human",
        "indicators": [
            {**ind, "frozen_score": fs["score"]}
            for ind, fs in zip(indicators, frozen_scored)
        ],
        "annotated_html": html,
    }


async def annotate(
    document: str,
    checkpoint_path: str | None = None,
    progress_cb=None,
) -> dict:
    logger.info("annotate | starting annotation run")
    await _emit_progress(progress_cb, 5, "Starting pipeline")
    if checkpoint_path:
        logger.info("annotate | loading checkpoint: %s", checkpoint_path)
        await _emit_progress(progress_cb, 15, "Loading checkpoint")
    else:
        logger.info("annotate | no checkpoint provided, using base LoRA client")
        await _emit_progress(progress_cb, 15, "Creating base client")

    runtime = await create_runtime(checkpoint_path=checkpoint_path)
    return await annotate_with_runtime(document, runtime, progress_cb=progress_cb)


if __name__ == "__main__":
    import sys
    text = sys.stdin.read()
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    result = asyncio.run(annotate(text, checkpoint))
    print(json.dumps(result, indent=2))
