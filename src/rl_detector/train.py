"""Main GRPO training loop."""

import asyncio
import json
import logging
import math
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
from rl_detector.prompts import neutral
from rl_detector.rewards import compute_advantages, compute_reward, format_diagnostics, format_reward, format_status, parse_indicators
from rl_detector.rollouts import extract_response_text, generate_rollouts

load_dotenv()
logger = logging.getLogger(__name__)

# expose BASE_MODEL for annotate.py
BASE_MODEL = CFG.model.base_model
EVAL_SAMPLE_SIZE = 50
EVAL_EVERY_STEPS = 5
EVAL_SEED = 2262
GLOBAL_SEED = 2262
SAVE_TTL_SECONDS = 2 * 24 * 60 * 60


def _sample_label_noise_mode(rng: random.Random) -> str:
    # small helper: decide which supervision hint to use per rollout
    p_correct = float(getattr(CFG.training, "label_noise_correct_prob", 1.0))
    p_flip = float(getattr(CFG.training, "label_noise_flip_prob", 0.0))
    p_unknown = float(getattr(CFG.training, "label_noise_unknown_prob", 0.0))
    tot = max(1e-12, p_correct + p_flip + p_unknown)
    p_correct /= tot
    p_flip /= tot
    u = rng.random()
    if u < p_correct:
        return "correct"
    if u < (p_correct + p_flip):
        return "flip"
    return "unknown"


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    qq = max(0.0, min(1.0, q))
    idx = int(qq * (len(ordered) - 1))
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
    completion_text = tokenizer.decode(completion_tokens)
    return {
        "completion_text": completion_text,
        "response_text": extract_response_text(completion_text),
        "completion_tokens": completion_tokens,
        "completion_logprobs": completion_logprobs,
    }


async def _evaluate_model(training_client, tokenizer, frozen_client, eval_docs: list[dict], step: int | str, eval_audit_path: str | None = None) -> dict:
    logger.info("eval | step %s | evaluating %d test docs with neutral prompt", step, len(eval_docs))
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    async def score_eval_doc(doc):
        rollout = await _sample_standard_rollout(sampling_client, tokenizer, doc["text"])
        completion_text = rollout["completion_text"]
        response_text = rollout["response_text"]
        indicators = parse_indicators(response_text) or []
        fmt = format_diagnostics(response_text, doc["text"])
        format_ok = bool(fmt["ok"])
        format_reason = str(fmt["reason"])
        format_char_diff = int(fmt["char_diff_count"])
        if not format_ok:
            return 0.0, True, False, f"format:{format_reason}", None, doc["label"], format_char_diff, completion_text, response_text, indicators, []
        frozen_scored = await rank_indicators(frozen_client, response_text, indicators) if indicators else []
        if indicators and frozen_scored is None:
            return None, False, True, "frozen_parse_failed", None, doc["label"], 0, completion_text, response_text, indicators, []
        reward = compute_reward(response_text, doc["text"], doc["label"], frozen_scored)
        agg_score = sum(s["score"] for s in frozen_scored) / len(frozen_scored) if frozen_scored else 0.0
        return reward, True, True, "ok", agg_score, doc["label"], format_char_diff, completion_text, response_text, indicators, frozen_scored

    results = await asyncio.gather(*[score_eval_doc(doc) for doc in eval_docs])

    if eval_audit_path:
        import pathlib
        pathlib.Path(eval_audit_path).parent.mkdir(parents=True, exist_ok=True)
        with open(eval_audit_path, "a") as f:
            doc_traces = []
            for doc, res in zip(eval_docs, results):
                reward, format_used, frozen_used, reason, agg_score, label, char_diff, completion_text, response_text, indicators, frozen_scored = res
                doc_traces.append({
                    "doc_id": doc.get("id"),
                    "label": label,
                    "reward": reward,
                    "agg_score": agg_score,
                    "format_reason": reason,
                    "format_char_diff": char_diff,
                    "completion_text": completion_text,
                    "response_text": response_text,
                    "indicators": [
                        {
                            "span_text": ind["span_text"],
                            "explanation": ind["explanation"],
                            "type": ind.get("type"),
                            "frozen_score": fs["score"],
                        }
                        for ind, fs in zip(indicators, frozen_scored)
                    ],
                })
            f.write(json.dumps({"step": step, "docs": doc_traces}, ensure_ascii=False) + "\n")
    rewards = [0.0 if res[0] is None else res[0] for res in results if res[1]]
    format_ok_flags = [res[2] for res in results if res[1]]
    n_excluded = sum(1 for res in results if not res[1])
    format_reasons = [res[3] for res in results if res[3] is not None]
    format_char_diffs = [res[6] for res in results]
    eval_format_no_tells = sum(1 for r in format_reasons if r == "format:no_tells")
    eval_format_invalid_type = sum(1 for r in format_reasons if r == "format:invalid_type")
    eval_format_text_mismatch = sum(1 for r in format_reasons if r == "format:text_mismatch")
    eval_text_mismatch_char_diffs = [res[6] for res in results if res[3] == "format:text_mismatch"]
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

    eval_ai_scores = [s for s, y in zip(agg_scores, true_labels) if y == 1]
    eval_human_scores = [s for s, y in zip(agg_scores, true_labels) if y == 0]
    eval_ai_score_mean = (sum(eval_ai_scores) / len(eval_ai_scores)) if eval_ai_scores else 0.0
    eval_human_score_mean = (sum(eval_human_scores) / len(eval_human_scores)) if eval_human_scores else 0.0
    eval_score_gap_ai_minus_human = eval_ai_score_mean - eval_human_score_mean
    eval_ai_positive_rate = (sum(1 for s in eval_ai_scores if s > 0.0) / len(eval_ai_scores)) if eval_ai_scores else 0.0
    eval_human_negative_rate = (sum(1 for s in eval_human_scores if s < 0.0) / len(eval_human_scores)) if eval_human_scores else 0.0
    eval_ambiguous_rate = (sum(1 for s in agg_scores if abs(s) < 0.2) / len(agg_scores)) if agg_scores else 0.0

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
        "eval_ai_score_mean": eval_ai_score_mean,
        "eval_human_score_mean": eval_human_score_mean,
        "eval_score_gap_ai_minus_human": eval_score_gap_ai_minus_human,
        "eval_ai_positive_rate": eval_ai_positive_rate,
        "eval_human_negative_rate": eval_human_negative_rate,
        "eval_ambiguous_rate_abs_lt_02": eval_ambiguous_rate,
        "eval_ai_score_p10": _quantile(eval_ai_scores, 0.10),
        "eval_ai_score_p50": _quantile(eval_ai_scores, 0.50),
        "eval_ai_score_p90": _quantile(eval_ai_scores, 0.90),
        "eval_human_score_p10": _quantile(eval_human_scores, 0.10),
        "eval_human_score_p50": _quantile(eval_human_scores, 0.50),
        "eval_human_score_p90": _quantile(eval_human_scores, 0.90),
        "eval_format_no_tells": eval_format_no_tells,
        "eval_format_invalid_type": eval_format_invalid_type,
        "eval_format_text_mismatch": eval_format_text_mismatch,
        "eval_format_char_diff_mean": (sum(format_char_diffs) / len(format_char_diffs)) if format_char_diffs else 0.0,
        "eval_text_mismatch_char_diff_mean": (sum(eval_text_mismatch_char_diffs) / len(eval_text_mismatch_char_diffs)) if eval_text_mismatch_char_diffs else 0.0,
        "eval_text_mismatch_char_diff_p95": _quantile(eval_text_mismatch_char_diffs, 0.95),
        "eval_text_mismatch_char_diff_max": max(eval_text_mismatch_char_diffs) if eval_text_mismatch_char_diffs else 0,
        "_eval_ai_scores": eval_ai_scores,
        "_eval_human_scores": eval_human_scores,
    }


def build_datum(
    neutral_tokens: list[int],
    completion_tokens: list[int],
    completion_logprobs: list[float],
    advantage: float,
    n_reasoning_tokens: int = 0,
) -> tinker.Datum:
    """
    Build a Datum for importance-sampling GRPO.

    Full sequence: [neutral_prompt... | completion...]
    model_input:   full_seq[:-1]   (right-shifted)
    target_tokens: full_seq[1:]    (left-shifted)
    logprobs:      [0]*(N-1) + [0]*R + response_logprobs
    advantages:    [0]*(N-1) + [0]*R + [advantage]*S
    mask:          [0.0]*(N-1) + [0.0]*R + [1.0]*S

    where R = n_reasoning_tokens (masked out — conditioned on directed prompt,
    incomparable under neutral prompt) and S = len(completion_tokens) - R.
    """
    N = len(neutral_tokens)
    M = len(completion_tokens)
    R = min(n_reasoning_tokens, M)   # reasoning tokens to mask
    S = M - R                        # response tokens to train on
    full_seq = neutral_tokens + completion_tokens

    input_tokens = full_seq[:-1]      # length N+M-1
    target_tokens = full_seq[1:]      # length N+M-1

    logprobs = [0.0] * (N - 1) + [0.0] * R + completion_logprobs[R:]
    advantages = [0.0] * (N - 1) + [0.0] * R + [advantage] * S

    assert len(input_tokens) == len(target_tokens) == len(logprobs) == len(advantages)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
        },
    )


async def _process_doc(sampling_client, tokenizer, frozen_client, doc, all_docs: list[dict], rng: random.Random, rollout_seed: int | None = None):
    """Process a single doc: generate rollouts, score, compute rewards/advantages, build datums.

    Teacher rollouts use a contrastive prompt (main doc + a labeled contrast doc of opposite label).
    Student optimization uses the neutral prompt (main doc only). The IS correction bridges the two.
    """
    document = doc["text"]
    label = doc["label"]
    label_str = "AI" if label == 1 else "human"
    snippet = document[:60].replace("\n", " ")

    # compute neutral tokens up front — needed for re-scoring and datum construction
    neutral_tokens = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": neutral(document)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    # sample a noisy supervision mode per rollout
    noise_modes = [_sample_label_noise_mode(rng) for _ in range(CFG.training.k)]
    main_label_hints: list[int] = []
    show_labels_flags: list[bool] = []
    for mode in noise_modes:
        if mode == "correct":
            main_label_hints.append(label)
            show_labels_flags.append(True)
        elif mode == "flip":
            main_label_hints.append(1 - label)
            show_labels_flags.append(True)
        else:
            # unknown means: keep label internally, hide it in prompt text
            main_label_hints.append(label)
            show_labels_flags.append(False)

    # choose one contrast doc per rollout, conditioned on noisy hint when known
    contrast_docs: list[dict] = []
    for i in range(CFG.training.k):
        hint = main_label_hints[i]
        if not show_labels_flags[i]:
            target_contrast_label = rng.choice([0, 1])
        else:
            target_contrast_label = 1 - hint
        contrast_pool = [d for d in all_docs if d["label"] == target_contrast_label and d is not doc]
        if not contrast_pool:
            contrast_pool = [d for d in all_docs if d is not doc]
        contrast_docs.append(rng.choice(contrast_pool))

    logger.info("rollouts | generating K=%d for %s doc: %r... (seed=%s)", CFG.training.k, label_str, snippet, rollout_seed)
    logger.info(
        "rollouts | noisy modes: correct=%d flip=%d unknown=%d",
        sum(1 for m in noise_modes if m == "correct"),
        sum(1 for m in noise_modes if m == "flip"),
        sum(1 for m in noise_modes if m == "unknown"),
    )
    t0_rollouts = time.perf_counter()
    rollouts = await generate_rollouts(
        sampling_client,
        tokenizer,
        document,
        contrast_docs=contrast_docs,
        main_label_hints=main_label_hints,
        show_labels_flags=show_labels_flags,
        seed=rollout_seed,
    )
    dt_rollouts = time.perf_counter() - t0_rollouts
    n_tells_per_rollout = [len(parse_indicators(r["response_text"]) or []) for r in rollouts]
    n_reasoning_tokens_per_rollout = [r.get("n_reasoning_tokens", 0) for r in rollouts]
    logger.info("rollouts | done in %.1fs — tells per rollout: %s", dt_rollouts, n_tells_per_rollout)
    logger.info("rollouts | reasoning tokens masked (per rollout): %s", n_reasoning_tokens_per_rollout)

    # Re-score each completion under the neutral prompt so that the old logprobs
    # used for importance sampling match the distribution we actually train on.
    # The directed-prompt logprobs returned by the sampler are p(tok | directed_ctx),
    # but the training forward pass computes p(tok | neutral_ctx), so they must agree.
    async def rescore_under_neutral(r: dict) -> list[float]:
        full_input = tinker.ModelInput.from_ints(neutral_tokens + r["completion_tokens"])
        all_lps: list[float | None] = await sampling_client.compute_logprobs_async(full_input)
        # compute_logprobs returns one value per token; take only the completion slice.
        # completion_tokens[0] sits at index N in the full sequence, so its logprob is all_lps[N].
        N = len(neutral_tokens)
        completion_lps = all_lps[N:]
        return [lp if lp is not None else 0.0 for lp in completion_lps]

    logger.info("rollouts | re-scoring %d completions under neutral prompt", len(rollouts))
    t0_rescore = time.perf_counter()
    neutral_logprobs_list = await asyncio.gather(*[rescore_under_neutral(r) for r in rollouts])
    dt_rescore = time.perf_counter() - t0_rescore
    # Compute IS ratios before overwriting logprobs: ratio = exp(sum(neutral - directed)).
    # Values << 1 mean the completion is much more likely under the directed prompt than neutral.
    is_ratios = []
    for r, neutral_lps in zip(rollouts, neutral_logprobs_list):
        directed_lps = r["completion_logprobs"]
        R = r.get("n_reasoning_tokens", 0)
        # IS ratio computed only over response tokens — reasoning tokens are
        # conditioned on the directed prompt and must not contribute.
        n = min(len(neutral_lps), len(directed_lps))
        log_ratio = sum(neutral_lps[j] - directed_lps[j] for j in range(R, n)) if n > R else 0.0
        is_ratios.append(math.exp(max(-20.0, min(20.0, log_ratio))))
        r["completion_logprobs"] = neutral_lps
    is_ratio_mean = sum(is_ratios) / len(is_ratios) if is_ratios else 1.0
    is_ratio_min = min(is_ratios) if is_ratios else 1.0
    is_ratio_max = max(is_ratios) if is_ratios else 1.0
    logger.info(
        "rollouts | re-scoring done in %.1fs — IS ratios: mean=%.4f min=%.4f max=%.4f",
        dt_rescore, is_ratio_mean, is_ratio_min, is_ratio_max,
    )

    import json
    import pathlib
    AUDIT_FORMAT_FAIL_PATH = getattr(CFG.training, "format_fail_audit_path", "format_fail_audit.jsonl")
    async def score_and_reward(i, r):
        response_text = r["response_text"]
        indicators = parse_indicators(response_text) or []
        fmt = format_diagnostics(response_text, document)
        format_ok = bool(fmt["ok"])
        format_reason = str(fmt["reason"])
        format_char_diff = int(fmt["char_diff_count"])
        contrast_label_str = "AI" if r["contrast_label"] == 1 else "human"
        logger.info("scoring  | rollout %d/%d (contrast=%s): %d tells", i + 1, CFG.training.k, contrast_label_str, len(indicators))
        if not format_ok:
            logger.info(
                "scoring  | rollout %d/%d format invalid (%s, char_diff=%d), reward=0 and skip frozen scoring",
                i + 1,
                CFG.training.k,
                format_reason,
                format_char_diff,
            )
            # Write failed format example to audit file
            fail_obj = {
                "doc_id": doc.get("id", None),
                "doc_label": label,
                "rollout_index": i,
                "contrast_label": r["contrast_label"],
                "noise_mode": r.get("noise_mode"),
                "main_label_hint": r.get("main_label_hint"),
                "show_labels": r.get("show_labels"),
                "format_reason": format_reason,
                "format_char_diff": format_char_diff,
                "input_text": document,
                "response_text": response_text,
                "was_text_fixed": r.get("was_text_fixed"),
                "wrong_response_text": r.get("wrong_response_text"),
                "indicators": indicators,
                "format_diag": fmt,
            }
            pathlib.Path(AUDIT_FORMAT_FAIL_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_FORMAT_FAIL_PATH, "a") as f:
                f.write(json.dumps(fail_obj, ensure_ascii=False) + "\n")
            return indicators, [], 0.0, True, f"format:{format_reason}", False, 0.0, format_char_diff
        t0_frozen = time.perf_counter()
        frozen_scored = await rank_indicators(frozen_client, response_text, indicators) if indicators else []
        dt_frozen = time.perf_counter() - t0_frozen
        if indicators and frozen_scored is None:
            logger.warning("scoring  | rollout %d/%d excluded: frozen score parse failed after retries", i + 1, CFG.training.k)
            return indicators, [], None, False, "frozen_parse_failed", True, dt_frozen, 0
        reward = compute_reward(response_text, document, label, frozen_scored)
        agg = sum(s["score"] for s in frozen_scored) / len(frozen_scored) if frozen_scored else 0.0
        logger.info("scoring  | rollout %d/%d done in %.1fs — agg=%.3f reward=%.1f", i + 1, CFG.training.k, dt_frozen, agg, reward)
        return indicators, frozen_scored, reward, True, "ok", True, dt_frozen, 0

    logger.info("scoring  | sending %d rollouts to frozen model", len(rollouts))
    t0_scoring = time.perf_counter()
    results = await asyncio.gather(*[score_and_reward(i, r) for i, r in enumerate(rollouts)])
    dt_scoring = time.perf_counter() - t0_scoring
    all_indicators = [res[0] for res in results]
    all_frozen_scored = [res[1] for res in results]
    rewards = [res[2] for res in results]
    used_for_optimization = [res[3] for res in results]
    exclude_reasons = [res[4] for res in results]
    format_ok_flags = [res[5] for res in results]
    frozen_times = [res[6] for res in results]
    format_char_diffs = [res[7] for res in results]
    dt_frozen_mean = sum(frozen_times) / len(frozen_times) if frozen_times else 0.0
    logger.info(
        "timing   | doc=%r rollouts=%.1fs rescore=%.1fs scoring=%.1fs (frozen mean/rollout=%.1fs)",
        snippet, dt_rollouts, dt_rescore, dt_scoring, dt_frozen_mean,
    )

    rewards_for_optimization = [rw for rw, use in zip(rewards, used_for_optimization) if use and rw is not None]
    advantages = compute_advantages(rewards_for_optimization) if rewards_for_optimization else []

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
            n_reasoning_tokens=r.get("n_reasoning_tokens", 0),
        )
        datums.append(datum)

    reward_mean = (sum(rewards_for_optimization) / len(rewards_for_optimization)) if rewards_for_optimization else 0.0
    format_used = [fmt for fmt, use in zip(format_ok_flags, used_for_optimization) if use]
    format_rate = (sum(1 for fmt in format_used if fmt) / len(format_used)) if format_used else 0.0

    contrast_labels_used = [r["contrast_label"] for r in rollouts]

    doc_audit = {
        "document": document,
        "label": label,
        "contrast_labels": contrast_labels_used,
        "noise_modes": noise_modes,
        "main_label_hints": main_label_hints,
        "show_labels_flags": show_labels_flags,
        "reward_mean": reward_mean,
        "format_rate": format_rate,
        "n_excluded_rollouts": sum(1 for use in used_for_optimization if not use),
        "rollouts": [
            {
                "index": i,
                "contrast_label": r["contrast_label"],
                "noise_mode": noise_modes[i],
                "main_label_hint": r.get("main_label_hint"),
                "show_labels": r.get("show_labels"),
                "used_for_optimization": used_for_optimization[i],
                "exclude_reason": exclude_reasons[i],
                "format_ok": format_ok_flags[i],
                "completion_text": r["completion_text"],
                "response_text": r["response_text"],
                "was_text_fixed": r.get("was_text_fixed", False),
                "wrong_response_text": r.get("wrong_response_text"),
                "completion_tokens_len": len(r["completion_tokens"]),
                "completion_logprobs_len": len(r["completion_logprobs"]),
                "format_char_diff_count": format_char_diffs[i],
                "indicators": [
                    {
                        "span_text": ind["span_text"],
                        "explanation": ind["explanation"],
                        "type": ind.get("type"),
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

    doc_timing = {
        "dt_rollouts": dt_rollouts,
        "dt_rescore": dt_rescore,
        "dt_scoring": dt_scoring,
        "dt_frozen_mean": dt_frozen_mean,
    }
    return datums, doc_audit, doc_timing


async def train_step(
    training_client,
    tokenizer,
    frozen_client,
    docs: list[dict],
    all_docs: list[dict],
    step: int,
    audit_log,
) -> dict:
    """One GRPO update for a batch of docs. Returns aggregate metrics."""
    logger.info("step %d | saving weights and getting sampling client", step)
    t0_save = time.perf_counter()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    dt_save = time.perf_counter() - t0_save
    logger.info("step %d | sampling client ready in %.1fs", step, dt_save)

    logger.info("step %d | processing %d docs", step, len(docs))
    doc_results = await asyncio.gather(
        *[
            _process_doc(
                sampling_client, tokenizer, frozen_client, doc,
                all_docs=all_docs,
                rng=random.Random(GLOBAL_SEED + step * 1000 + i),
                rollout_seed=GLOBAL_SEED,
            )
            for i, doc in enumerate(docs)
        ]
    )

    # Top-p group variance filtering: drop groups where reward variance is low,
    # since the model is already consistent there and contributes little gradient.
    top_p = getattr(CFG.training, "group_variance_top_p", 1.0)
    if top_p < 1.0 and doc_results:
        def _group_variance(doc_result):
            _, doc_audit, _dt = doc_result
            rewards = [ro["reward"] for ro in doc_audit["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
            if len(rewards) < 2:
                return 0.0
            mean = sum(rewards) / len(rewards)
            return sum((r - mean) ** 2 for r in rewards) / len(rewards)

        variances = [_group_variance(dr) for dr in doc_results]
        n_keep = max(1, round(top_p * len(doc_results)))
        sorted_indices = sorted(range(len(doc_results)), key=lambda i: variances[i], reverse=True)
        keep_set = set(sorted_indices[:n_keep])
        n_dropped = len(doc_results) - len(keep_set)
        if n_dropped > 0:
            logger.info(
                "step %d | group_variance_top_p=%.2f: dropping %d/%d low-variance docs (variances: kept min=%.4f, dropped max=%.4f)",
                step, top_p, n_dropped, len(doc_results),
                min(variances[i] for i in keep_set),
                max((variances[i] for i in range(len(doc_results)) if i not in keep_set), default=0.0),
            )
        doc_results = [dr for i, dr in enumerate(doc_results) if i in keep_set]

    all_datums = []
    docs_audit = []
    doc_timings = []
    for datums, doc_audit, doc_timing in doc_results:
        all_datums.extend(datums)
        docs_audit.append(doc_audit)
        doc_timings.append(doc_timing)

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
    opt_dt = time.perf_counter() - opt_t0
    logger.info("step %d | optimizer done in %.1fs", step, opt_dt)

    all_rewards = [ro["reward"] for da in docs_audit for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    reward_mean = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
    all_format_ok = [ro.get("format_ok", False) for da in docs_audit for ro in da["rollouts"] if ro.get("used_for_optimization")]
    format_rate = (sum(1 for ok in all_format_ok if ok) / len(all_format_ok)) if all_format_ok else 0.0
    n_excluded_rollouts = sum(1 for da in docs_audit for ro in da["rollouts"] if not ro.get("used_for_optimization"))

    all_tell_scores = [ind["frozen_score"] for da in docs_audit for ro in da["rollouts"] for ind in ro.get("indicators", [])]
    tell_score_mean = (sum(all_tell_scores) / len(all_tell_scores)) if all_tell_scores else 0.0
    tell_score_std = (
        (sum((s - tell_score_mean) ** 2 for s in all_tell_scores) / len(all_tell_scores)) ** 0.5
        if all_tell_scores else 0.0
    )

    # per-label reward breakdown: AI docs (label=1) vs human docs (label=0)
    ai_rewards = [ro["reward"] for da in docs_audit if da["label"] == 1 for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    human_rewards = [ro["reward"] for da in docs_audit if da["label"] == 0 for ro in da["rollouts"] if ro.get("used_for_optimization") and ro["reward"] is not None]
    ai_reward_mean = (sum(ai_rewards) / len(ai_rewards)) if ai_rewards else 0.0
    human_reward_mean = (sum(human_rewards) / len(human_rewards)) if human_rewards else 0.0

    ai_tell_scores = [ind["frozen_score"] for da in docs_audit if da["label"] == 1 for ro in da["rollouts"] for ind in ro.get("indicators", [])]
    human_tell_scores = [ind["frozen_score"] for da in docs_audit if da["label"] == 0 for ro in da["rollouts"] for ind in ro.get("indicators", [])]
    ai_tell_score_mean = (sum(ai_tell_scores) / len(ai_tell_scores)) if ai_tell_scores else 0.0
    human_tell_score_mean = (sum(human_tell_scores) / len(human_tell_scores)) if human_tell_scores else 0.0
    tell_score_gap_ai_minus_human = ai_tell_score_mean - human_tell_score_mean
    ai_positive_rate = (sum(1 for s in ai_tell_scores if s > 0.0) / len(ai_tell_scores)) if ai_tell_scores else 0.0
    human_negative_rate = (sum(1 for s in human_tell_scores if s < 0.0) / len(human_tell_scores)) if human_tell_scores else 0.0
    ambiguous_rate = (sum(1 for s in all_tell_scores if abs(s) < 0.2) / len(all_tell_scores)) if all_tell_scores else 0.0

    type_ai_scores = [ind["frozen_score"] for da in docs_audit for ro in da["rollouts"] for ind in ro.get("indicators", []) if ind.get("type") == "AI"]
    type_human_scores = [ind["frozen_score"] for da in docs_audit for ro in da["rollouts"] for ind in ro.get("indicators", []) if ind.get("type") == "human"]
    type_ai_score_mean = (sum(type_ai_scores) / len(type_ai_scores)) if type_ai_scores else 0.0
    type_human_score_mean = (sum(type_human_scores) / len(type_human_scores)) if type_human_scores else 0.0

    all_exclude_reasons = [ro.get("exclude_reason") for da in docs_audit for ro in da["rollouts"]]
    train_format_no_tells = sum(1 for r in all_exclude_reasons if r == "format:no_tells")
    train_format_invalid_type = sum(1 for r in all_exclude_reasons if r == "format:invalid_type")
    train_format_text_mismatch = sum(1 for r in all_exclude_reasons if r == "format:text_mismatch")
    train_text_mismatch_char_diffs = [ro.get("format_char_diff_count", 0) for da in docs_audit for ro in da["rollouts"] if ro.get("exclude_reason") == "format:text_mismatch"]
    train_all_format_char_diffs = [ro.get("format_char_diff_count", 0) for da in docs_audit for ro in da["rollouts"]]

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

    def _mean(vals): return sum(vals) / len(vals) if vals else 0.0
    dt_rollouts_mean = _mean([t["dt_rollouts"] for t in doc_timings])
    dt_rescore_mean = _mean([t["dt_rescore"] for t in doc_timings])
    dt_scoring_mean = _mean([t["dt_scoring"] for t in doc_timings])
    dt_frozen_mean_mean = _mean([t["dt_frozen_mean"] for t in doc_timings])

    step_total_dt = dt_save + fb_dt + opt_dt  # excludes doc processing (runs in parallel)
    logger.info(
        "timing   | step %d: save_weights=%.1fs fwd_bwd=%.1fs optim=%.1fs | step_total=%.1fs",
        step, dt_save, fb_dt, opt_dt, step_total_dt,
    )
    logger.info(
        "timing   | step %d (per-doc means): rollouts=%.1fs rescore=%.1fs scoring=%.1fs frozen/rollout=%.1fs",
        step, dt_rollouts_mean, dt_rescore_mean, dt_scoring_mean, dt_frozen_mean_mean,
    )

    return {
        "train_reward_mean": reward_mean,
        "train_format_rate": format_rate,
        "train_n_positive": n_positive,
        "train_n_negative": n_negative,
        "train_n_zero": n_zero,
        "train_n_excluded_rollouts": n_excluded_rollouts,
        "train_tell_score_mean": tell_score_mean,
        "train_tell_score_std": tell_score_std,
        "train_ai_reward_mean": ai_reward_mean,
        "train_human_reward_mean": human_reward_mean,
        "train_ai_tell_score_mean": ai_tell_score_mean,
        "train_human_tell_score_mean": human_tell_score_mean,
        "train_tell_score_gap_ai_minus_human": tell_score_gap_ai_minus_human,
        "train_ai_positive_rate": ai_positive_rate,
        "train_human_negative_rate": human_negative_rate,
        "train_ambiguous_rate_abs_lt_02": ambiguous_rate,
        "train_ai_tell_score_p10": _quantile(ai_tell_scores, 0.10),
        "train_ai_tell_score_p50": _quantile(ai_tell_scores, 0.50),
        "train_ai_tell_score_p90": _quantile(ai_tell_scores, 0.90),
        "train_human_tell_score_p10": _quantile(human_tell_scores, 0.10),
        "train_human_tell_score_p50": _quantile(human_tell_scores, 0.50),
        "train_human_tell_score_p90": _quantile(human_tell_scores, 0.90),
        "train_type_ai_score_mean": type_ai_score_mean,
        "train_type_human_score_mean": type_human_score_mean,
        "train_format_no_tells": train_format_no_tells,
        "train_format_invalid_type": train_format_invalid_type,
        "train_format_text_mismatch": train_format_text_mismatch,
        "train_format_char_diff_mean": (sum(train_all_format_char_diffs) / len(train_all_format_char_diffs)) if train_all_format_char_diffs else 0.0,
        "train_text_mismatch_char_diff_mean": (sum(train_text_mismatch_char_diffs) / len(train_text_mismatch_char_diffs)) if train_text_mismatch_char_diffs else 0.0,
        "train_text_mismatch_char_diff_p95": _quantile(train_text_mismatch_char_diffs, 0.95),
        "train_text_mismatch_char_diff_max": max(train_text_mismatch_char_diffs) if train_text_mismatch_char_diffs else 0,
        "_train_ai_tell_scores": ai_tell_scores,
        "_train_human_tell_scores": human_tell_scores,
        "timing_save_weights_s": dt_save,
        "timing_fwd_bwd_s": fb_dt,
        "timing_optim_s": opt_dt,
        "timing_rollouts_mean_s": dt_rollouts_mean,
        "timing_rescore_mean_s": dt_rescore_mean,
        "timing_scoring_mean_s": dt_scoring_mean,
        "timing_frozen_per_rollout_mean_s": dt_frozen_mean_mean,
    }


async def main(resume: str | None = None, resume_step: int = 0):
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

    if resume:
        logger.info("startup | resuming from checkpoint %s (step offset=%d), rank=%d + loading tokenizer and dataset in parallel...", resume, resume_step, CFG.model.lora_rank)
        training_client_coro = service_client.create_training_client_from_state_with_optimizer_async(path=resume)
    else:
        logger.info("startup | creating LoRA training client (base_model=%s, rank=%d) + loading tokenizer and dataset in parallel...", CFG.model.base_model, CFG.model.lora_rank)
        training_client_coro = service_client.create_lora_training_client_async(
            base_model=CFG.model.base_model,
            rank=CFG.model.lora_rank,
            seed=GLOBAL_SEED,
        )
    training_client, tokenizer, all_docs, test_docs = await asyncio.gather(
        training_client_coro,
        loop.run_in_executor(None, _load_tokenizer),
        loop.run_in_executor(None, _load_dataset),
        loop.run_in_executor(None, lambda: load_docs(split="test")),
    )
    logger.info("startup | all ready")

    eval_docs = _select_eval_docs(test_docs)

    frozen_client = get_client()
    step = resume_step
    best_eval_auroc = float("-inf")
    best_eval_path = None
    with open(CFG.training.audit_log_path, "w") as audit_log:
        if eval_docs:
            eval_metrics = await _evaluate_model(training_client, tokenizer, frozen_client, eval_docs, step, eval_audit_path=CFG.training.eval_audit_log_path)
            if CFG.wandb.enabled:
                eval_core = {
                    "eval_reward_mean",
                    "eval_format_rate",
                    "eval_n_excluded_rollouts",
                    "eval_auroc",
                    "eval_tpr_at_fpr_001",
                }
                eval_log_data = {}
                for k, v in eval_metrics.items():
                    if k.startswith("_") or not k.startswith("eval_"):
                        continue
                    prefix = "eval" if k in eval_core else "eval_diag"
                    eval_log_data[f"{prefix}/{k[len('eval_'):]}"] = v
                if eval_metrics.get("_eval_ai_scores"):
                    eval_log_data["eval_diag/hist_ai_scores"] = wandb.Histogram(eval_metrics["_eval_ai_scores"])
                if eval_metrics.get("_eval_human_scores"):
                    eval_log_data["eval_diag/hist_human_scores"] = wandb.Histogram(eval_metrics["_eval_human_scores"])
                wandb.log(eval_log_data, step=step)
            if eval_metrics["eval_auroc"] > best_eval_auroc:
                best_eval_auroc = eval_metrics["eval_auroc"]
                best_eval_path = await _save_state_with_ttl(training_client, name=f"best-step-{step}")
                logger.info("Saved new best eval checkpoint at step %d: %s", step, best_eval_path)
                if CFG.wandb.enabled:
                    wandb.log({"eval/best_auroc": best_eval_auroc}, step=step)

        steps_iter = iter_balanced_steps(all_docs, docs_per_step=CFG.training.docs_per_step)
        pbar = tqdm(total=CFG.training.max_steps, desc="training", unit="step")
        for docs in steps_iter:
            metrics = await train_step(training_client, tokenizer, frozen_client, docs, all_docs=all_docs, step=step, audit_log=audit_log)
            pbar.set_postfix(
                train_reward=f"{metrics['train_reward_mean']:.3f}",
                train_format=f"{metrics['train_format_rate']:.2f}",
            )
            pbar.update(1)

            if CFG.wandb.enabled:
                train_log_data = {
                    "train/reward_mean": metrics["train_reward_mean"],
                    "train/format_rate": metrics["train_format_rate"],
                    "train/n_positive_rollouts": metrics["train_n_positive"],
                    "train/n_negative_rollouts": metrics["train_n_negative"],
                    "train/n_zero_rollouts": metrics["train_n_zero"],
                    "train/n_excluded_rollouts": metrics["train_n_excluded_rollouts"],
                    "train/ai_reward_mean": metrics["train_ai_reward_mean"],
                    "train/human_reward_mean": metrics["train_human_reward_mean"],
                    "train_diag/tell_score_mean": metrics["train_tell_score_mean"],
                    "train_diag/tell_score_std": metrics["train_tell_score_std"],
                    "train_diag/ai_tell_score_mean": metrics["train_ai_tell_score_mean"],
                    "train_diag/human_tell_score_mean": metrics["train_human_tell_score_mean"],
                    "train_diag/tell_score_gap_ai_minus_human": metrics["train_tell_score_gap_ai_minus_human"],
                    "train_diag/ai_positive_rate": metrics["train_ai_positive_rate"],
                    "train_diag/human_negative_rate": metrics["train_human_negative_rate"],
                    "train_diag/ambiguous_rate_abs_lt_02": metrics["train_ambiguous_rate_abs_lt_02"],
                    "train_diag/ai_tell_score_p10": metrics["train_ai_tell_score_p10"],
                    "train_diag/ai_tell_score_p50": metrics["train_ai_tell_score_p50"],
                    "train_diag/ai_tell_score_p90": metrics["train_ai_tell_score_p90"],
                    "train_diag/human_tell_score_p10": metrics["train_human_tell_score_p10"],
                    "train_diag/human_tell_score_p50": metrics["train_human_tell_score_p50"],
                    "train_diag/human_tell_score_p90": metrics["train_human_tell_score_p90"],
                    "train_diag/type_ai_score_mean": metrics["train_type_ai_score_mean"],
                    "train_diag/type_human_score_mean": metrics["train_type_human_score_mean"],
                    "train_diag/format_no_tells": metrics["train_format_no_tells"],
                    "train_diag/format_invalid_type": metrics["train_format_invalid_type"],
                    "train_diag/format_text_mismatch": metrics["train_format_text_mismatch"],
                    "train_diag/format_char_diff_mean": metrics["train_format_char_diff_mean"],
                    "train_diag/text_mismatch_char_diff_mean": metrics["train_text_mismatch_char_diff_mean"],
                    "train_diag/text_mismatch_char_diff_p95": metrics["train_text_mismatch_char_diff_p95"],
                    "train_diag/text_mismatch_char_diff_max": metrics["train_text_mismatch_char_diff_max"],
                    "timing/save_weights_s": metrics["timing_save_weights_s"],
                    "timing/fwd_bwd_s": metrics["timing_fwd_bwd_s"],
                    "timing/optim_s": metrics["timing_optim_s"],
                    "timing/rollouts_mean_s": metrics["timing_rollouts_mean_s"],
                    "timing/rescore_mean_s": metrics["timing_rescore_mean_s"],
                    "timing/scoring_mean_s": metrics["timing_scoring_mean_s"],
                    "timing/frozen_per_rollout_mean_s": metrics["timing_frozen_per_rollout_mean_s"],
                }
                if metrics.get("_train_ai_tell_scores"):
                    train_log_data["train_diag/hist_ai_tell_scores"] = wandb.Histogram(metrics["_train_ai_tell_scores"])
                if metrics.get("_train_human_tell_scores"):
                    train_log_data["train_diag/hist_human_tell_scores"] = wandb.Histogram(metrics["_train_human_tell_scores"])
                wandb.log(train_log_data, step=step)

            logger.info(
                f"step={step} train_reward={metrics['train_reward_mean']:.3f} "
                f"train_format={metrics['train_format_rate']:.2f} "
                f"+={metrics['train_n_positive']} -={metrics['train_n_negative']} 0={metrics['train_n_zero']} "
                f"excluded={metrics['train_n_excluded_rollouts']}"
            )
            step += 1

            if step % EVAL_EVERY_STEPS == 0 and eval_docs:
                eval_metrics = await _evaluate_model(training_client, tokenizer, frozen_client, eval_docs, step, eval_audit_path=CFG.training.eval_audit_log_path)
                if CFG.wandb.enabled:
                    eval_core = {
                        "eval_reward_mean",
                        "eval_format_rate",
                        "eval_n_excluded_rollouts",
                        "eval_auroc",
                        "eval_tpr_at_fpr_001",
                    }
                    eval_log_data = {}
                    for k, v in eval_metrics.items():
                        if k.startswith("_") or not k.startswith("eval_"):
                            continue
                        prefix = "eval" if k in eval_core else "eval_diag"
                        eval_log_data[f"{prefix}/{k[len('eval_'):]}"] = v
                    if eval_metrics.get("_eval_ai_scores"):
                        eval_log_data["eval_diag/hist_ai_scores"] = wandb.Histogram(eval_metrics["_eval_ai_scores"])
                    if eval_metrics.get("_eval_human_scores"):
                        eval_log_data["eval_diag/hist_human_scores"] = wandb.Histogram(eval_metrics["_eval_human_scores"])
                    wandb.log(eval_log_data, step=step)
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
    import argparse
    parser = argparse.ArgumentParser(description="GRPO training loop")
    parser.add_argument(
        "--resume",
        metavar="TINKER_PATH",
        default=None,
        help="Tinker checkpoint path to resume from, e.g. tinker://run-id/weights/checkpoint-001. Restores weights and optimizer state.",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        metavar="N",
        help="Training step to start counting from when resuming (default: 0).",
    )
    args = parser.parse_args()
    asyncio.run(main(resume=args.resume, resume_step=args.resume_step))
