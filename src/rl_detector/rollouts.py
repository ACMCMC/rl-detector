"""Generate rollouts from the current policy using directed prompts."""

import tinker

from rl_detector.config import CFG
from rl_detector.prompts import directed_ai, directed_human


async def generate_rollouts(
    sampling_client,
    tokenizer,
    document: str,
    K: int | None = None,
) -> list[dict]:
    """
    Generate K rollouts: K/2 directed to find AI tells, K/2 for human tells.
    Returns list of dicts with completion_text, completion_tokens, completion_logprobs.
    """
    if K is None:
        K = CFG.training.k
    directed_prompts = (
        [directed_ai(document)] * (K // 2) +
        [directed_human(document)] * (K // 2)
    )
    sampling_params = tinker.SamplingParams(
        max_tokens=CFG.sampling.max_tokens,
        temperature=CFG.sampling.temperature,
        top_p=CFG.sampling.top_p,
    )
    results = []
    for prompt_text in directed_prompts:
        prompt_text_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text_formatted)
        model_input = tinker.ModelInput.from_ints(prompt_tokens)
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
            # compute logprobs separately if not returned inline
            full_tokens = prompt_tokens + completion_tokens
            full_input = tinker.ModelInput.from_ints(full_tokens)
            all_logprobs = await sampling_client.compute_logprobs_async(full_input)
            completion_logprobs = [lp or 0.0 for lp in all_logprobs[len(prompt_tokens):]]
        completion_text = tokenizer.decode(completion_tokens)
        results.append({
            "completion_text": completion_text,
            "completion_tokens": completion_tokens,
            "completion_logprobs": completion_logprobs,
        })
    return results
