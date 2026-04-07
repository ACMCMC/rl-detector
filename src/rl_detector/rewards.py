"""Reward functions and advantage computation for GRPO."""

import math
import re

# Weight of the granularity bonus relative to the calibration reward.
# Small enough that getting the classification right always dominates.
_GRANULARITY_WEIGHT = 0.05
_MARGIN_WEIGHT = 0.35
_MARGIN_TARGET = 0.45

_TELL_TAG_RE = re.compile(r"<tell\b([^>]*)>(.*?)</tell>", re.DOTALL)
_ATTR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*"([^"]*)"')
_VALID_TYPES = {"AI", "human"}


def _parse_attrs(attrs_blob: str) -> dict[str, str]:
    return {k: v for k, v in _ATTR_RE.findall(attrs_blob)}


def _extract_final_channel(text: str) -> str:
    """Extract content from the 'final' channel if present, else return full text."""
    m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_indicators(output: str) -> list[dict] | None:
    """
    Parse <tell explanation="..." type="...">SPAN</tell> tags from model output.
    Returns list of {"span_text", "explanation", "type"} dicts, or None if no tags found.
    """
    text = _extract_final_channel(output)
    tells = []
    for m in _TELL_TAG_RE.finditer(text):
        attrs = _parse_attrs(m.group(1))
        if "explanation" not in attrs:
            continue
        tells.append({
            "span_text": m.group(2),
            "explanation": attrs["explanation"],
            "type": attrs.get("type"),
        })
    return tells if tells else None


def strip_tags(tagged_text: str) -> str:
    """Remove all <tell ...> and </tell> tags, keeping the inner text."""
    return re.sub(r"</tell>|<tell\b[^>]*>", "", tagged_text)


def format_reward(output: str, document: str) -> float:
    """
    1.0 if the model output (final channel, tags stripped) matches the document exactly
    AND every <tell> tag has a valid type attribute ("AI" or "human").
    0.0 otherwise.
    """
    final = _extract_final_channel(output)
    if not final:
        return 0.0
    tells = [(m.group(1), m.group(2)) for m in _TELL_TAG_RE.finditer(final)]
    if not tells:
        return 0.0
    # every tell must declare a valid type
    for attrs_blob, _ in tells:
        attrs = _parse_attrs(attrs_blob)
        if attrs.get("type") not in _VALID_TYPES:
            return 0.0
    stripped = strip_tags(final)
    return 1.0 if stripped == document else 0.0


def calibration_reward(aggregate_score: float, label: int) -> float:
    """
    Continuous calibration reward in [0, 1].
    Label mapping: 1=AI expects positive score, 0=human expects negative score.
    Uses linear map: (1 + y * a) / 2, where y in {-1, +1}, a in [-1, 1].
    """
    a = max(-1.0, min(1.0, float(aggregate_score)))
    y = 1.0 if label == 1 else -1.0
    return 0.5 * (1.0 + y * a)


def margin_reward(aggregate_score: float, label: int, margin: float = _MARGIN_TARGET) -> float:
    """
    Margin objective in [0, 1].
    If signed score y*a is below the target margin, reward is 0.
    Above margin, reward increases linearly to 1 at y*a=1.
    """
    a = max(-1.0, min(1.0, float(aggregate_score)))
    y = 1.0 if label == 1 else -1.0
    ya = y * a
    m = max(0.0, min(0.95, float(margin)))
    if ya <= m:
        return 0.0
    return (ya - m) / (1.0 - m)


def granularity_reward(n_tells: int) -> float:
    """
    Small bonus in [0, 1] for spreading evidence across many short tells.
    Uses log growth so the benefit of going from 1->2 tells is larger than 4->5.
    Saturates around 5+ tells (log(6)/log(6) = 1.0).
    """
    if n_tells <= 0:
        return 0.0
    return min(1.0, math.log(n_tells + 1) / math.log(6))


def compute_reward(
    output: str,
    document: str,
    label: int,
    frozen_scored: list[dict],
) -> float:
    """
    Combined reward. Format is a gate: if 0, return 0.
    Main signal: calibration (classification correctness).
    Extra signal: margin objective to push confident separation, not just correct sign.
    Minor signal: granularity bonus for many short tells over one long tell.
    """
    if format_reward(output, document) == 0.0:
        return 0.0
    from rl_detector.frozen import aggregate
    agg = aggregate(frozen_scored)
    cal = calibration_reward(agg, label)
    mar = margin_reward(agg, label)
    gran = granularity_reward(len(frozen_scored))
    return cal + _MARGIN_WEIGHT * mar + _GRANULARITY_WEIGHT * gran


def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Normalize rewards within the group: subtract mean and divide by std.
    This is standard GRPO normalization. Without the std division, when all K
    rollouts get similar rewards (common early in training), the raw centered
    advantages are near zero and the gradient effectively vanishes.
    When all rewards are identical (std=0), return all zeros — there is no
    learning signal from a group where every rollout behaved the same.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]
