"""Main GRPO training loop."""

import asyncio
import json
import logging
import os
import random
import time

import tinker
from sklearn.metrics import roc_auc_score, roc_curve
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
from rl_detector.rewards import compute_advantages, compute_reward, format_reward, parse_indicators
from rl_detector.rollouts import generate_rollouts

load_dotenv()
logger = logging.getLogger(__name__)

# expose BASE_MODEL for annotate.py
BASE_MODEL = CFG.model.base_model
EVAL_SAMPLE_SIZE = 25
EVAL_EVERY_STEPS = 5
EVAL_SEED = 2262
GLOBAL_SEED = 2262
SAVE_TTL_SECONDS = 2 * 24 * 60 * 60


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(0.95 * (len(ordered) - 1))))
    return ordered[idx]


async def _await_with_heartbeat(coro, step: int, phase: str, every_s: int = 20):
    # keep emitting liveness logs while waiting on remote async work
    task = asyncio.create_task(coro)
    waited_s = 0
    while True:
        done, _ = await asyncio.wait({task}, timeout=every_s)
        if done:
            return await task
        waited_s += every_s
        logger.info("step %d | still waiting on %s (%ds elapsed)", step, phase, waited_s)


async def _save_state_with_ttl(training_client, name: str) -> str:
    save_future = await training_client.save_state_async(name=name, ttl_seconds=SAVE_TTL_SECONDS)
    return await save_future.result_async()


def _select_eval_docs(docs: list[dict], sample_size: int = EVAL_SAMPLE_SIZE, seed: int = EVAL_SEED) -> list[dict]:
    rng = random.Random(seed)
    ai_docs = [d for d in docs if d["label"] == 1]
    human_docs = [d for d in docs if d["label"] == 0]
    rng.shuffle(ai_docs)
    rng.shuffle(human_docs)
    n_ai = min(sample_size // 2, len(ai_docs))
    n_human = min(sample_size - n_ai, len(human_docs))
    chosen = ai_docs[:n_ai] + human_docs[:n_human]
    if len(chosen) < sample_size:
        remaining = [d for d in docs if d not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: sample_size - len(chosen)])
    rng.shuffle(chosen)
    return chosen[:sample_size]


async def _sample_standard_rollout(sampling_client, tokenizer, document: str) -> dict:
    prompt_text_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": neutral(document)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_tokens = tokenizer.encode(prompt_text_formatted)
    model_input = tinker.ModelInput.from_ints(prompt_tokens)
    sampled = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            seed=EVAL_SEED,
            temperature=CFG.sampling.temperature,
            top_p=CFG.sampling.top_p,
            reasoning_effort=CFG.sampling.reasoning_effort,
        ),
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
    }


async def _evaluate_model(training_client, tokenizer, frozen_client, eval_docs: list[dict], step: int | str) -> dict:
    logger.info("eval | step %s | evaluating %d test docs with neutral prompt", step, len(eval_docs))
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    async def score_eval_doc(doc):
        rollout = await _sample_standard_rollout(sampling_client, tokenizer, doc["text"])
        indicators = parse_indicators(rollout["completion_text"]) or []
        format_ok = format_reward(rollout["completion_text"], doc["text"]) == 1.0
        if not format_ok:
            return 0.0, True, False, None, None, doc["label"]
        frozen_scored = await rank_indicators(frozen_client, rollout["completion_text"], indicators) if indicators else []
        if indicators and frozen_scored is None:
            return None, False, True, "frozen_parse_failed", None, doc["label"]
        reward = compute_reward(rollout["completion_text"], doc["text"], doc["label"], frozen_scored)
        agg_score = sum(s["score"] for s in frozen_scored) / len(frozen_scored) if frozen_scored else 0.0
        return reward, True, True, None, agg_score, doc["label"]

    results = await asyncio.gather(*[score_eval_doc(doc) for doc in eval_docs])
    rewards = [0.0 if res[0] is None else res[0] for res in results if res[1]]
    format_ok_flags = [res[2] for res in results if res[1]]
    n_excluded = sum(1 for res in results if not res[1])
    agg_scores = [res[4] for res in results if res[1] and res[4] is not None]
    true_labels = [res[5] for res in results if res[1] and res[4] is not None]

    eval_reward_mean = (sum(rewards) / len(rewards)) if rewards else 0.0
    eval_format_rate = (sum(1 for ok in format_ok_flags if ok) / len(format_ok_flags)) if format_ok_flags else 0.0

    eval_auroc = roc_auc_score(true_labels, agg_scores) if len(agg_scores) >= 2 else 0.0
    eval_tpr_at_fpr_001 = 0.0
    if len(agg_scores) >= 2:
        fpr, tpr, _ = roc_curve(true_labels, agg_scores)
        tpr_at_or_below = [tpr_i for fpr_i, tpr_i in zip(fpr, tpr) if fpr_i <= 0.01]
        eval_tpr_at_fpr_001 = max(tpr_at_or_below) if tpr_at_or_below else 0.0

    logger.info(
        "eval | step %s | reward=%.3f format=%.2f auroc=%.3f tpr@fpr01=%.3f excluded=%d",
        step,
        eval_reward_mean,
        eval_format_rate,
        eval_auroc,
        eval_tpr_at_fpr_001,
        n_excluded,
    )

    return {
        "eval_reward_mean": eval_reward_mean,
        "eval_format_rate": eval_format_rate,
        "eval_n_excluded_rollouts": n_excluded,
        "eval_auroc": eval_auroc,
        "eval_tpr_at_fpr_001": eval_tpr_at_fpr_001,
    }


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
        format_ok = format_reward(r["completion_text"], document) == 1.0
        logger.info("scoring  | rollout %d/%d (%s-directed): %d tells", i + 1, CFG.training.k, prompt_directions[i], len(indicators))
        if not format_ok:
            logger.info("scoring  | rollout %d/%d format invalid, reward=0 and skip frozen scoring", i + 1, CFG.training.k)
            return indicators, [], 0.0, True, None, False
        frozen_scored = await rank_indicators(frozen_client, r["completion_text"], indicators) if indicators else []
        if indicators and frozen_scored is None:
            logger.warning("scoring  | rollout %d/%d excluded: frozen score parse failed after retries", i + 1, CFG.training.k)
            return indicators, [], None, False, "frozen_parse_failed", True
        reward = compute_reward(r["completion_text"], document, label, frozen_scored)
        agg = sum(s["score"] for s in frozen_scored) / len(frozen_scored) if frozen_scored else 0.0
        logger.info("scoring  | rollout %d/%d done — agg=%.3f reward=%.1f", i + 1, CFG.training.k, agg, reward)
        return indicators, frozen_scored, reward, True, None, True

    logger.info("scoring  | sending %d rollouts to frozen model", len(rollouts))
    results = await asyncio.gather(*[score_and_reward(i, r) for i, r in enumerate(rollouts)])
    all_indicators = [res[0] for res in results]
    all_frozen_scored = [res[1] for res in results]
    rewards = [res[2] for res in results]
    used_for_optimization = [res[3] for res in results]
    exclude_reasons = [res[4] for res in results]
    format_ok_flags = [res[5] for res in results]

    rewards_for_optimization = [rw for rw, use in zip(rewards, used_for_optimization) if use and rw is not None]
    advantages = compute_advantages(rewards_for_optimization) if rewards_for_optimization else []

    neutral_tokens = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": neutral(document)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    datums = []
    adv_idx = 0
    for i, r in enumerate(rollouts):
        if not used_for_optimization[i]:
            continue
        adv = advantages[adv_idx]
        adv_idx += 1
        if not r["completion_tokens"]:
            continue
        datum = build_datum(
            neutral_tokens,
            r["completion_tokens"],
            r["completion_logprobs"],
            adv,
        )
        datums.append(datum)

    reward_mean = (sum(rewards_for_optimization) / len(rewards_for_optimization)) if rewards_for_optimization else 0.0
    format_used = [fmt for fmt, use in zip(format_ok_flags, used_for_optimization) if use]
    format_rate = (sum(1 for fmt in format_used if fmt) / len(format_used)) if format_used else 0.0

    doc_audit = {
        "document": document,
        "label": label,
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "n_excluded_rollouts": sum(1 for use in used_for_optimization if not use),
        "rollouts": [
            {
                "index": i,
                "prompt_direction": prompt_directions[i],
                "used_for_optimization": used_for_optimization[i],
                "exclude_reason": exclude_reasons[i],
                "format_ok": format_ok_flags[i],
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
                "advantage": (advantages[sum(1 for u in used_for_optimization[:i + 1] if u) - 1] if used_for_optimization[i] else None),
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
    logger.info("step %d | saving weights and getting sampling client", step)
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

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
        return {
            "train_reward_mean": 0.0,
            "train_format_rate": 0.0,
            "train_n_positive": 0,
            "train_n_negative": 0,
            "train_n_zero": 0,
            "train_n_excluded_rollouts": 0,
        }

    completion_lens = [ro["completion_tokens_len"] for da in docs_audit for ro in da["rollouts"]]
    completion_total = sum(completion_lens)
    completion_mean = (completion_total / len(completion_lens)) if completion_lens else 0.0
    logger.info(
        "step %d | datum stats: n=%d completion_total=%d completion_mean=%.1f completion_p95=%d completion_max=%d",
        step,
        len(all_datums),
        completion_total,
        completion_mean,
        _p95(completion_lens),
        max(completion_lens) if completion_lens else 0,
    )

    logger.info("step %d | forward/backward on %d datums", step, len(all_datums))
    fb_t0 = time.perf_counter()
    fb_future = await training_client.forward_backward_async(
        data=all_datums,
        loss_fn="importance_sampling",
    )
    logger.info("step %d | forward/backward submitted, waiting for result", step)
    fb_result = await _await_with_heartbeat(
        fb_future.result_async(),
        step,
        f"forward/backward result ({len(all_datums)} datums)",
    )
    fb_dt = time.perf_counter() - fb_t0
    if hasattr(fb_result, "loss"):
        logger.info("step %d | forward/backward done in %.1fs loss=%s", step, fb_dt, fb_result.loss)
    else:
        logger.info("step %d | forward/backward done in %.1fs", step, fb_dt)

    logger.info("step %d | optimizer step", step)
    opt_t0 = time.perf_counter()
    opt_future = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=CFG.training.learning_rate)
    )
    logger.info("step %d | optimizer submitted, waiting for result", step)
    await _await_with_heartbeat(
        opt_future.result_async(),
        step,
        "optimizer result",
    )
    logger.info("step %d | optimizer done in %.1fs", step, time.perf_counter() - opt_t0)

    all_rewards = [ro["reward"] for da in docs_audit for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    reward_mean = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
    all_format_ok = [ro.get("format_ok", False) for da in docs_audit for ro in da["rollouts"] if ro.get("used_for_optimization")]
    format_rate = (sum(1 for ok in all_format_ok if ok) / len(all_format_ok)) if all_format_ok else 0.0
    n_excluded_rollouts = sum(1 for da in docs_audit for ro in da["rollouts"] if not ro.get("used_for_optimization"))

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
        "train_reward_mean": reward_mean,
        "train_format_rate": format_rate,
        "train_n_positive": n_positive,
        "train_n_negative": n_negative,
        "train_n_zero": n_zero,
        "train_n_excluded_rollouts": n_excluded_rollouts,
    }


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # single seed for reproducible startup and training behavior
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)

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
    loop = asyncio.get_event_loop()

    def _load_tokenizer():
        logger.info("startup | loading tokenizer...")
        tok = AutoTokenizer.from_pretrained(CFG.model.base_model)
        logger.info("startup | tokenizer ready")
        return tok

    def _load_dataset():
        logger.info("startup | loading dataset (%s)...", CFG.data.dataset)
        docs = load_docs()
        logger.info("startup | dataset ready (%d docs)", len(docs))
        return docs

    logger.info("startup | creating LoRA training client (base_model=%s, rank=%d) + loading tokenizer and dataset in parallel...", CFG.model.base_model, CFG.model.lora_rank)
    training_client, tokenizer, all_docs, test_docs = await asyncio.gather(
        service_client.create_lora_training_client_async(
            base_model=CFG.model.base_model,
            rank=CFG.model.lora_rank,
            seed=GLOBAL_SEED,
        ),
        loop.run_in_executor(None, _load_tokenizer),
        loop.run_in_executor(None, _load_dataset),
        loop.run_in_executor(None, lambda: load_docs(split="test")),
    )
    logger.info("startup | all ready")

    eval_docs = _select_eval_docs(test_docs)

    frozen_client = get_client()
    step = 0
    best_eval_auroc = float("-inf")
    best_eval_path = None
    with open(CFG.training.audit_log_path, "w") as audit_log:
        if eval_docs:
            eval_metrics = await _evaluate_model(training_client, tokenizer, frozen_client, eval_docs, step)
            if CFG.wandb.enabled:
                wandb.log({
                    (f"eval/{k[len('eval_'):]}" if k.startswith("eval_") else k): v
                    for k, v in eval_metrics.items()
                }, step=step)
            if eval_metrics["eval_auroc"] > best_eval_auroc:
                best_eval_auroc = eval_metrics["eval_auroc"]
                best_eval_path = await _save_state_with_ttl(training_client, name=f"best-step-{step}")
                logger.info("Saved new best eval checkpoint at step %d: %s", step, best_eval_path)
                if CFG.wandb.enabled:
                    wandb.log({"eval/best_auroc": best_eval_auroc}, step=step)

        steps_iter = iter_balanced_steps(all_docs, docs_per_step=CFG.training.docs_per_step)
        pbar = tqdm(total=CFG.training.max_steps, desc="training", unit="step")
        for docs in steps_iter:
            metrics = await train_step(training_client, tokenizer, frozen_client, docs, step, audit_log)
            pbar.set_postfix(
                train_reward=f"{metrics['train_reward_mean']:.3f}",
                train_format=f"{metrics['train_format_rate']:.2f}",
            )
            pbar.update(1)

            if CFG.wandb.enabled:
                wandb.log({
                    "train/reward_mean": metrics["train_reward_mean"],
                    "train/format_rate": metrics["train_format_rate"],
                    "train/n_positive_rollouts": metrics["train_n_positive"],
                    "train/n_negative_rollouts": metrics["train_n_negative"],
                    "train/n_zero_rollouts": metrics["train_n_zero"],
                    "train/n_excluded_rollouts": metrics["train_n_excluded_rollouts"],
                }, step=step)

            logger.info(
                f"step={step} train_reward={metrics['train_reward_mean']:.3f} "
                f"train_format={metrics['train_format_rate']:.2f} "
                f"+={metrics['train_n_positive']} -={metrics['train_n_negative']} 0={metrics['train_n_zero']} "
                f"excluded={metrics['train_n_excluded_rollouts']}"
            )
            step += 1

            if step % EVAL_EVERY_STEPS == 0 and eval_docs:
                eval_metrics = await _evaluate_model(training_client, tokenizer, frozen_client, eval_docs, step)
                if CFG.wandb.enabled:
                    wandb.log({
                        (f"eval/{k[len('eval_'):]}" if k.startswith("eval_") else k): v
                        for k, v in eval_metrics.items()
                    }, step=step)
                if eval_metrics["eval_auroc"] > best_eval_auroc:
                    best_eval_auroc = eval_metrics["eval_auroc"]
                    best_eval_path = await _save_state_with_ttl(training_client, name=f"best-step-{step}")
                    logger.info("Saved new best eval checkpoint at step %d: %s", step, best_eval_path)
                    if CFG.wandb.enabled:
                        wandb.log({"eval/best_auroc": best_eval_auroc}, step=step)

            if step >= CFG.training.max_steps:
                logger.info("Reached max_steps, stopping.")
                break

            if CFG.training.checkpoint_every > 0 and step % CFG.training.checkpoint_every == 0:
                save_future = await training_client.save_state_async(name=f"step-{step}", ttl_seconds=SAVE_TTL_SECONDS)
                await save_future.result_async()
                logger.info(f"Saved checkpoint at step {step}")
                if CFG.wandb.enabled:
                    wandb.log({"checkpoint": step}, step=step)

        pbar.close()

    logger.info("training complete | saving final checkpoint")
    final_save = await training_client.save_state_async(name="final", ttl_seconds=SAVE_TTL_SECONDS)
    final_path = await final_save.result_async()
    logger.info(f"final checkpoint saved: {final_path}")
    if best_eval_path is not None:
        logger.info("best eval checkpoint kept at: %s", best_eval_path)

    if CFG.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    asyncio.run(main())
