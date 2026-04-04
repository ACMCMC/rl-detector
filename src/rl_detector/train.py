"""Main GRPO training loop."""

import asyncio
import json
import logging
import os

import tinker
import torch
import wandb
import weave
from dotenv import load_dotenv
from tinker import TensorData
from tqdm import tqdm
from transformers import AutoTokenizer

from rl_detector.config import CFG
from rl_detector.data import iter_balanced_steps, load_docs
from rl_detector.frozen import get_client, rank_indicators
from rl_detector.prompts import directed_ai, directed_human, neutral
from rl_detector.rewards import compute_advantages, compute_reward, parse_indicators
from rl_detector.rollouts import generate_rollouts

load_dotenv()
logger = logging.getLogger(__name__)

# expose BASE_MODEL for annotate.py
BASE_MODEL = CFG.model.base_model


def build_datum(
    neutral_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    advantage: float,
) -> tinker.Datum:
    """
    Build a Datum for importance-sampling GRPO.

    Full sequence: [neutral_prompt... | completion...]
    model_input:   full_seq[:-1]   (right-shifted)
    target_tokens: full_seq[1:]    (left-shifted)
    logprobs:      [0]*(N-1) + completion_logprobs    (old logprobs from directed sampling)
    advantages:    [0]*(N-1) + [advantage]*M
    mask:          [0.0]*(N-1) + [1.0]*M             (train only on completion)
    """
    N = len(neutral_tokens)
    M = len(completion_tokens)
    full_seq = neutral_tokens + completion_tokens

    input_tokens = full_seq[:-1]      # length N+M-1
    target_tokens = full_seq[1:]      # length N+M-1

    logprobs = [0.0] * (N - 1) + completion_logprobs
    advantages = [0.0] * (N - 1) + [advantage] * M

    assert len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
        },
    )


async def _process_doc(sampling_client, tokenizer, frozen_client, doc):
    """Process a single doc: generate rollouts, score, compute rewards/advantages, build datums."""
    document = doc["text"]
    label = doc["label"]
    label_str = "AI" if label == 1 else "human"
    snippet = document[:60].replace("\n", " ")

    logger.info("rollouts | generating K=%d for %s doc: %r...", CFG.training.k, label_str, snippet)
    rollouts = await generate_rollouts(sampling_client, tokenizer, document)
    n_tells_per_rollout = [len(parse_indicators(r["completion_text"]) or []) for r in rollouts]
    logger.info("rollouts | done — tells per rollout: %s", n_tells_per_rollout)

    prompt_directions = ["ai"] * (CFG.training.k // 2) + ["human"] * (CFG.training.k // 2)

    async def score_and_reward(i, r):
        indicators = parse_indicators(r["completion_text"]) or []
        logger.info("scoring  | rollout %d/%d (%s-directed): %d tells", i + 1, CFG.training.k, prompt_directions[i], len(indicators))
        frozen_scored = await rank_indicators(frozen_client, r["completion_text"], indicators) if indicators else []
        reward = compute_reward(r["completion_text"], document, label, frozen_scored)
        agg = sum(s["score"] for s in frozen_scored) / len(frozen_scored) if frozen_scored else 0.0
        logger.info("scoring  | rollout %d/%d done — agg=%.3f reward=%.1f", i + 1, CFG.training.k, agg, reward)
        return indicators, frozen_scored, reward

    logger.info("scoring  | sending %d rollouts to frozen model", len(rollouts))
    results = await asyncio.gather(*[score_and_reward(i, r) for i, r in enumerate(rollouts)])
    all_indicators = [res[0] for res in results]
    all_frozen_scored = [res[1] for res in results]
    rewards = [res[2] for res in results]

    advantages = compute_advantages(rewards)

    neutral_tokens = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": neutral(document)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    datums = []
    for r, adv in zip(rollouts, advantages):
        if not r["completion_tokens"]:
            continue
        datum = build_datum(
            neutral_tokens,
            r["completion_tokens"],
            r["completion_logprobs"],
            adv,
        )
        datums.append(datum)

    reward_mean = sum(rewards) / len(rewards)
    format_rate = sum(1 for rw in rewards if rw != 0.0) / len(rewards)

    doc_audit = {
        "document": document,
        "label": label,
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "rollouts": [
            {
                "index": i,
                "prompt_direction": prompt_directions[i],
                "completion_text": r["completion_text"],
                "completion_tokens_len": len(r["completion_tokens"]),
                "completion_logprobs_len": len(r["completion_logprobs"]),
                "indicators": [
                    {
                        "span_text": ind["span_text"],
                        "explanation": ind["explanation"],
                        "frozen_score": fs["score"],
                    }
                    for ind, fs in zip(all_indicators[i], all_frozen_scored[i])
                ],
                "reward": rewards[i],
                "advantage": advantages[i],
            }
            for i, r in enumerate(rollouts)
        ],
    }

    return datums, doc_audit


async def train_step(
    training_client,
    tokenizer,
    frozen_client,
    docs: list[dict],
    step: int,
    audit_log,
) -> dict:
    """One GRPO update for a batch of docs. Returns aggregate metrics."""
    logger.info("step %d | getting ephemeral sampling client", step)
    sampling_client = await training_client.get_ephemeral_sampling_client_async()

    logger.info("step %d | processing %d docs", step, len(docs))
    doc_results = await asyncio.gather(
        *[_process_doc(sampling_client, tokenizer, frozen_client, doc) for doc in docs]
    )

    all_datums = []
    docs_audit = []
    for datums, doc_audit in doc_results:
        all_datums.extend(datums)
        docs_audit.append(doc_audit)

    if not all_datums:
        logger.warning("step %d | no valid datums, skipping update", step)
        return {"reward_mean": 0.0, "format_rate": 0.0}

    logger.info("step %d | forward/backward on %d datums", step, len(all_datums))
    fb_future = await training_client.forward_backward_async(
        data=all_datums,
        loss_fn="importance_sampling",
    )
    await fb_future.result_async()
    logger.info("step %d | optimizer step", step)
    opt_future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=CFG.training.learning_rate)
    )
    await opt_future.result_async()

    all_rewards = [ro["reward"] for da in docs_audit for ro in da["rollouts"]]
    reward_mean = sum(all_rewards) / len(all_rewards)
    format_rate = sum(1 for rw in all_rewards if rw != 0.0) / len(all_rewards)

    # per-rollout reward breakdown for wandb
    n_positive = sum(1 for rw in all_rewards if rw > 0)
    n_negative = sum(1 for rw in all_rewards if rw < 0)
    n_zero = sum(1 for rw in all_rewards if rw == 0.0)

    audit_entry = {
        "step": step,
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "docs": docs_audit,
    }
    audit_log.write(json.dumps(audit_entry) + "\n")
    audit_log.flush()

    return {
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
    }


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if CFG.wandb.enabled:
        wandb.init(
            project=CFG.wandb.project,
            entity=CFG.wandb.entity,
            config={
                "base_model": CFG.model.base_model,
                "lora_rank": CFG.model.lora_rank,
                "k": CFG.training.k,
                "docs_per_step": CFG.training.docs_per_step,
                "learning_rate": CFG.training.learning_rate,
                "max_steps": CFG.training.max_steps,
                "max_tokens": CFG.sampling.max_tokens,
                "temperature": CFG.sampling.temperature,
                "top_p": CFG.sampling.top_p,
                "frozen_model": CFG.frozen.model,
            },
        )

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=CFG.model.base_model,
        rank=CFG.model.lora_rank,
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.base_model)
    frozen_client = get_client()

    all_docs = load_docs()
    step = 0
    with open(CFG.training.audit_log_path, "w") as audit_log:
        steps_iter = iter_balanced_steps(all_docs, docs_per_step=CFG.training.docs_per_step)
        pbar = tqdm(total=CFG.training.max_steps, desc="training", unit="step")
        for docs in steps_iter:
            metrics = await train_step(training_client, tokenizer, frozen_client, docs, step, audit_log)
            pbar.set_postfix(
                reward=f"{metrics['reward_mean']:.3f}",
                format=f"{metrics['format_rate']:.2f}",
            )
            pbar.update(1)

            if CFG.wandb.enabled:
                wandb.log({
                    "reward_mean": metrics["reward_mean"],
                    "format_rate": metrics["format_rate"],
                    "n_positive_rollouts": metrics["n_positive"],
                    "n_negative_rollouts": metrics["n_negative"],
                    "n_zero_rollouts": metrics["n_zero"],
                }, step=step)

            logger.info(
                f"step={step} reward={metrics['reward_mean']:.3f} "
                f"format={metrics['format_rate']:.2f} "
                f"+={metrics['n_positive']} -={metrics['n_negative']} 0={metrics['n_zero']}"
            )
            step += 1
            if step >= CFG.training.max_steps:
                logger.info("Reached max_steps, stopping.")
                break

            if CFG.training.checkpoint_every > 0 and step % CFG.training.checkpoint_every == 0:
                save_future = await training_client.save_state_async(name=f"step-{step}")
                await save_future.result_async()
                logger.info(f"Saved checkpoint at step {step}")
                if CFG.wandb.enabled:
                    wandb.log({"checkpoint": step}, step=step)

        pbar.close()

    if CFG.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
