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
) -> list[dict]:
    """
    Score all indicators in a single frozen model call.
    Returns list of {score} dicts in the same order as indicators.
    Falls back to 0.0 for all on parse failure.
    """
    if not indicators:
        return []

    n = len(indicators)
    fallback = [{"score": 0.0} for _ in indicators]

    prompt = FROZEN_SCORE_PROMPT.format(tagged_text=tagged_text)

    async with _semaphore():
        logger.debug("frozen | acquired slot (%d tells)", n)
        response = await client.chat.completions.create(
            model=CFG.frozen.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=CFG.frozen.max_tokens,
            reasoning_effort=CFG.frozen.reasoning_effort,
        )
    content = response.choices[0].message.content or ""
    logger.debug("frozen model raw response: %r", content)

    raw_scores = re.findall(r'<tell\b[^>]*\bscore="([^"]+)"', content)
    if len(raw_scores) != n:
        logger.warning("frozen model score parse failed: found %d score= attrs, expected %d — raw: %r", len(raw_scores), n, content)
        return fallback

    try:
        scores = [max(-1.0, min(1.0, float(s))) for s in raw_scores]
    except ValueError as e:
        logger.warning("frozen model score parse error: %s — raw: %r", e, content)
        return fallback

    return [{"score": s} for s in scores]


def aggregate(scored: list[dict]) -> float:
    """Mean of scores."""
    if not scored:
        return 0.0
    return sum(s["score"] for s in scored) / len(scored)
