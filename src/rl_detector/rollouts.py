"""Generate rollouts from the current policy using contrastive (teacher) prompts."""

import asyncio

import tinker

from rl_detector.config import CFG
from rl_detector.prompts import contrastive


async def generate_rollouts(
    sampling_client,
    tokenizer,
    document: str,
    contrast_docs: list[dict],
    K: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate K rollouts using contrastive (teacher) prompts.

    Each rollout uses a different contrast document (opposite label from the main document)
    so the model sees a diverse set of reference comparisons. This is the teacher side of
    self-distillation: the contrast is privileged context that will not be present at
    optimization time (which uses the neutral prompt).

    Args:
        contrast_docs: list of K dicts with {"text", "label"}, one per rollout.
                       Each must have the opposite label from the main document.
    """
    if K is None:
        K = CFG.training.k
    assert len(contrast_docs) == K, f"expected {K} contrast docs, got {len(contrast_docs)}"
    # enforce that all contrast docs have the same label (the caller picks opposite-label docs)
    contrast_labels = {c["label"] for c in contrast_docs}
    assert len(contrast_labels) == 1, f"contrast docs must all share one label, got {contrast_labels}"

    async def _sample_one(i: int, contrast: dict) -> dict:
        prompt_text = contrastive(document, contrast["text"], contrast["label"])
        prompt_text_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text_formatted)
        model_input = tinker.ModelInput.from_ints(prompt_tokens)

        # use the same fixed seed for every rollout — diversity comes from different contrast docs
        rollout_seed = seed
        sampling_params = tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            temperature=CFG.sampling.temperature,
            top_p=CFG.sampling.top_p,
            reasoning_effort=CFG.sampling.reasoning_effort,
            seed=rollout_seed,
        )

        sampled = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = sampled.sequences[0]
        completion_tokens = list(seq.tokens)
        if seq.logprobs is not None:
            completion_logprobs = list(seq.logprobs)
        else:
            completion_logprobs = [0.0] * len(completion_tokens)
        assert any(lp != 0.0 for lp in completion_logprobs), "completion_logprobs are all 0.0"

        return {
            "completion_text": tokenizer.decode(completion_tokens),
            "completion_tokens": completion_tokens,
            "completion_logprobs": completion_logprobs,
            "contrast_label": contrast["label"],
        }

    tasks = [_sample_one(i, contrast) for i, contrast in enumerate(contrast_docs)]
    return await asyncio.gather(*tasks)
