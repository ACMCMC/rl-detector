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
    half_k = K // 2
    directed_prompts = [
        directed_ai(document),
        directed_human(document),
    ]
    sampling_params = tinker.SamplingParams(
        max_tokens=CFG.sampling.max_tokens,
        temperature=CFG.sampling.temperature,
        top_p=CFG.sampling.top_p,
        reasoning_effort=CFG.sampling.reasoning_effort,
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
            num_samples=half_k,
            sampling_params=sampling_params,
        )
        for seq in sampled.sequences:
            completion_tokens = list(seq.tokens)
            if seq.logprobs is not None:
                completion_logprobs = list(seq.logprobs)
            else:
                # avoid extra API call cost; zeros keep tensor shapes valid
                completion_logprobs = [0.0] * len(completion_tokens)
            assert any(lp != 0.0 for lp in completion_logprobs), "completion_logprobs are all 0.0"
            completion_text = tokenizer.decode(completion_tokens)
            results.append({
                "completion_text": completion_text,
                "completion_tokens": completion_tokens,
                "completion_logprobs": completion_logprobs,
            })
    return results
