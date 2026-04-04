"""Reward functions and advantage computation for GRPO."""

import re


def _extract_final_channel(text: str) -> str:
    """Extract content from the 'final' channel if present, else return full text."""
    m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def parse_indicators(output: str) -> list[dict] | None:
    """
    Parse <tell explanation="...">SPAN</tell> tags from model output.
    Returns list of {"span_text", "explanation"} dicts, or None if no tags found.
    """
    text = _extract_final_channel(output)
    pattern = re.compile(r'<tell\s+explanation="([^"]*)">(.*?)</tell>', re.DOTALL)
    matches = pattern.findall(text)
    if not matches:
        return None
    return [{"span_text": span, "explanation": expl} for expl, span in matches]


def strip_tags(tagged_text: str) -> str:
    """Remove all <tell ...> and </tell> tags, keeping the inner text."""
    text = re.sub(r'<tell\s+explanation="[^"]*">', "", tagged_text)
    text = re.sub(r'</tell>', "", text)
    return text


def format_reward(output: str, document: str) -> float:
    """
    1.0 if the model output (final channel, tags stripped) matches the document exactly.
    0.0 otherwise.
    """
    final = _extract_final_channel(output)
    if not final:
        return 0.0
    # must have at least one tag
    if not re.search(r'<tell\s+explanation="[^"]*">', final):
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


def compute_reward(
    output: str,
    document: str,
    label: int,
    frozen_scored: list[dict],
) -> float:
    """Combined reward in [0, 1]. Format is a gate: if 0, skip calibration."""
    if format_reward(output, document) == 0.0:
        return 0.0
    from rl_detector.frozen import aggregate
    agg = aggregate(frozen_scored)
    return calibration_reward(agg, label)


def compute_advantages(rewards: list[float]) -> list[float]:
    """Center rewards within the group (all K rollouts share one baseline)."""
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]
