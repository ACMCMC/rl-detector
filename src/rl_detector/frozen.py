"""Frozen model scoring via DeepInfra (OpenAI-compatible API)."""

import asyncio
import logging
import os
import re

from openai import AsyncOpenAI

from rl_detector.config import CFG
from rl_detector.prompts import FROZEN_SCORE_PROMPT

logger = logging.getLogger(__name__)

# global semaphore shared across all docs and rollouts
_SEMAPHORE: asyncio.Semaphore | None = None


def _semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(CFG.frozen.max_concurrent)
    return _SEMAPHORE


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )


async def rank_indicators(
    client: AsyncOpenAI,
    tagged_text: str,
    indicators: list[dict],
) -> list[dict] | None:
    """
    Score all indicators in a single frozen model call.
    Returns list of {score} dicts in the same order as indicators.
    Retries parse failures up to 3 times; returns None if still invalid.
    """
    if not indicators:
        return []

    n = len(indicators)
    max_attempts = 3

    prompt = FROZEN_SCORE_PROMPT.format(tagged_text=tagged_text)

    sem = _semaphore()
    for attempt in range(1, max_attempts + 1):
        in_use_before = CFG.frozen.max_concurrent - sem._value
        logger.info(
            "frozen | waiting semaphore slot (%d tells, in_use=%d/%d, attempt=%d/%d)",
            n,
            in_use_before,
            CFG.frozen.max_concurrent,
            attempt,
            max_attempts,
        )
        async with sem:
            in_use_after = CFG.frozen.max_concurrent - sem._value
            logger.info(
                "frozen | acquired semaphore slot (%d tells, in_use=%d/%d)",
                n,
                in_use_after,
                CFG.frozen.max_concurrent,
            )
            response = await client.chat.completions.create(
                model=CFG.frozen.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=CFG.frozen.max_tokens,
                reasoning_effort=CFG.frozen.reasoning_effort,
            )
        logger.info("frozen | released semaphore slot (%d tells)", n)
        content = response.choices[0].message.content or ""
        logger.debug("frozen model raw response: %r", content)

        raw_scores = re.findall(r'<tell\b[^>]*\bscore="([^"]+)"', content)
        if len(raw_scores) != n:
            logger.warning(
                "frozen model score parse failed: found %d score= attrs, expected %d (attempt %d/%d)",
                len(raw_scores),
                n,
                attempt,
                max_attempts,
            )
            continue

        try:
            scores = [max(-1.0, min(1.0, float(s))) for s in raw_scores]
        except ValueError as e:
            logger.warning(
                "frozen model score parse error: %s (attempt %d/%d)",
                e,
                attempt,
                max_attempts,
            )
            continue

        return [{"score": s} for s in scores]

    logger.warning("frozen model failed to produce valid scores after %d attempts; excluding rollout", max_attempts)
    return None


def aggregate(scored: list[dict]) -> float:
    """Mean of scores."""
    if not scored:
        return 0.0
    return sum(s["score"] for s in scored) / len(scored)
