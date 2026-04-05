"""Frozen model scoring via DeepInfra or Google AI Studio (Gemini)."""

import asyncio
import logging
import os
import re

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from rl_detector.config import CFG
from rl_detector.prompts import FROZEN_SCORE_PROMPT

logger = logging.getLogger(__name__)

# global semaphore shared across all docs and rollouts
_SEMAPHORE: asyncio.Semaphore | None = None

_TELL_TAG_RE = re.compile(r"<tell\b([^>]*)>(.*?)</tell>", re.DOTALL)
_ATTR_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\"([^\"]*)\"")


def _extract_scored_tells(text: str) -> list[dict]:
    """Parse full <tell> tags and extract explanation/span/score as raw strings."""
    tells: list[dict] = []
    for m in _TELL_TAG_RE.finditer(text):
        attrs_blob = m.group(1)
        span = m.group(2)
        attrs = {k: v for k, v in _ATTR_RE.findall(attrs_blob)}
        tells.append(
            {
                "span_text": span,
                "explanation": attrs.get("explanation", ""),
                "score_raw": attrs.get("score"),
            }
        )
    return tells


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


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _reasoning_effort_to_thinking_level(effort: str) -> str:
    return {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(effort.lower(), "LOW")


async def _rank_indicators_gemini(
    tagged_text: str,
    indicators: list[dict],
) -> list[dict] | None:
    """Gemini backend for rank_indicators."""
    n = len(indicators)
    max_attempts = 3
    prompt = FROZEN_SCORE_PROMPT.format(tagged_text=tagged_text)
    client = get_gemini_client()
    thinking_level = _reasoning_effort_to_thinking_level(CFG.frozen.reasoning_effort)

    sem = _semaphore()
    for attempt in range(1, max_attempts + 1):
        async with sem:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=CFG.frozen.gemini_model,
                    contents=[
                        genai_types.Content(
                            role="user",
                            parts=[genai_types.Part.from_text(text=prompt)],
                        )
                    ],
                    config=genai_types.GenerateContentConfig(
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=-1),
                        max_output_tokens=CFG.frozen.max_tokens,
                    ),
                ),
            )
        content = response.text or ""
        logger.debug("gemini frozen model raw response: %r", content)

        scored_tells = _extract_scored_tells(content)
        if len(scored_tells) != n:
            logger.warning(
                "gemini frozen score parse failed: found %d <tell> tags, expected %d (attempt %d/%d)",
                len(scored_tells), n, attempt, max_attempts,
            )
            continue

        score_pool: dict[tuple[str, str], list[float]] = {}
        try:
            for tell in scored_tells:
                raw = tell.get("score_raw")
                if raw is None:
                    raise ValueError("missing score attribute in <tell> tag")
                key = (tell["span_text"], tell["explanation"])
                score_pool.setdefault(key, []).append(max(-1.0, min(1.0, float(raw))))

            scores: list[float] = []
            for ind in indicators:
                key = (ind["span_text"], ind.get("explanation", ""))
                bucket = score_pool.get(key)
                if not bucket:
                    raise ValueError(f"missing scored tell for span/explanation: {key}")
                scores.append(bucket.pop(0))
        except ValueError as e:
            logger.warning("gemini frozen score parse error: %s (attempt %d/%d)", e, attempt, max_attempts)
            continue

        return [{"score": s} for s in scores]

    logger.warning("gemini frozen model failed to produce valid scores after %d attempts; excluding rollout", max_attempts)
    return None


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

    if getattr(CFG.frozen, "provider", "deepinfra") == "gemini":
        return await _rank_indicators_gemini(tagged_text, indicators)

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
                seed=CFG.frozen.seed,
                temperature=0.0,
                max_tokens=CFG.frozen.max_tokens,
                reasoning_effort=CFG.frozen.reasoning_effort,
            )
        logger.info("frozen | released semaphore slot (%d tells)", n)
        content = response.choices[0].message.content or ""
        logger.debug("frozen model raw response: %r", content)

        scored_tells = _extract_scored_tells(content)
        if len(scored_tells) != n:
            logger.warning(
                "frozen model score parse failed: found %d <tell> tags, expected %d (attempt %d/%d)",
                len(scored_tells),
                n,
                attempt,
                max_attempts,
            )
            continue

        # match on the original tag identity (span + explanation), tolerate reordered tags
        score_pool: dict[tuple[str, str], list[float]] = {}
        try:
            for tell in scored_tells:
                raw = tell.get("score_raw")
                if raw is None:
                    raise ValueError("missing score attribute in <tell> tag")
                key = (tell["span_text"], tell["explanation"])
                score_pool.setdefault(key, []).append(max(-1.0, min(1.0, float(raw))))

            scores: list[float] = []
            for ind in indicators:
                key = (ind["span_text"], ind.get("explanation", ""))
                bucket = score_pool.get(key)
                if not bucket:
                    raise ValueError(f"missing scored tell for span/explanation: {key}")
                scores.append(bucket.pop(0))
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
