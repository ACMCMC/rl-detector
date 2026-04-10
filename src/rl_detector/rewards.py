"""Reward functions and advantage computation for GRPO."""

import math
import re
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import ftfy

from rl_detector.config import CFG

# Weight of the granularity bonus relative to the calibration reward.
# Small enough that getting the classification right always dominates.

_GRANULARITY_WEIGHT = float(getattr(CFG.training, "granularity_weight", 0.0))
_MARGIN_WEIGHT = float(getattr(CFG.training, "margin_weight", 0.0))
_MARGIN_TARGET = float(getattr(CFG.training, "margin_target", 0.45))
_MARGIN_SIGMOID_BETA = float(getattr(CFG.training, "margin_sigmoid_beta", 8.0))

_VALID_TYPES = {"AI", "human"}

# Regex fallbacks for when XML parsing is not possible.
# Matches both double-quoted and single-quoted attribute values.
_TELL_TAG_RE = re.compile(r"<tell\b([^>]*)>(.*?)</tell>", re.DOTALL)
_ATTR_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')

# Fix patterns for common XML errors from model output
_UNESCAPED_AMP_RE = re.compile(r"&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[0-9a-fA-F]+);)")
_SINGLE_ATTR_RE = re.compile(r"(\s[a-zA-Z_][a-zA-Z0-9_-]*)='([^']*)'")


def _parse_attr(m: re.Match) -> tuple[str, str]:
    """Return (name, value) from an _ATTR_RE match, handling both quote styles."""
    return m.group(1), m.group(2) if m.group(2) is not None else m.group(3)


def _normalize(text: str) -> str:
    """Normalize unicode for format comparison using ftfy."""
    return ftfy.fix_text(text)


def _try_fix_xml(text: str) -> str:
    """Best-effort repair of common XML errors in model output."""
    # Fix unescaped & (not already part of an entity reference)
    text = _UNESCAPED_AMP_RE.sub("&amp;", text)
    # Convert single-quoted attributes to double-quoted, escaping any " inside
    def _fix_single(m):
        return m.group(1) + '="' + m.group(2).replace('"', "&quot;") + '"'
    text = _SINGLE_ATTR_RE.sub(_fix_single, text)
    return text


def _parse_xml_response(text: str) -> ET.Element | None:
    """
    Parse model output as a <text>...</text> XML document.
    Tries a best-effort fix pass if the first parse fails.
    Returns the root Element on success, None if unparseable.
    """
    if not text.startswith("<text>"):
        text = f"<text>{text}</text>"
    for attempt in range(2):
        try:
            return ET.fromstring(text)
        except ET.ParseError:
            if attempt == 0:
                text = _try_fix_xml(text)
    return None


def parse_indicators(output: str) -> list[dict] | None:
    """
    Parse <tell explanation="..." type="...">SPAN</tell> tags from model output.
    Uses XML parsing (with best-effort fix) and falls back to regex.
    Returns list of {"span_text", "explanation", "type"} dicts, or None if no tags found.
    """
    root = _parse_xml_response(output)
    if root is not None:
        tells = [
            {
                "span_text": "".join(tell.itertext()),
                "explanation": tell.get("explanation", ""),
                "type": tell.get("type"),
            }
            for tell in root.iter("tell")
            if tell.get("explanation") is not None
        ]
        return tells if tells else None

    # Regex fallback
    tells = []
    for m in _TELL_TAG_RE.finditer(output):
        attrs = {k: v for k, v in (_parse_attr(a) for a in _ATTR_RE.finditer(m.group(1)))}
        if "explanation" not in attrs:
            continue
        tells.append({"span_text": m.group(2), "explanation": attrs["explanation"], "type": attrs.get("type")})
    return tells if tells else None


def strip_tags(tagged_text: str) -> str:
    """Remove all <tell ...> and </tell> tags and the <text> wrapper, keeping plain text."""
    root = _parse_xml_response(tagged_text)
    if root is not None:
        return "".join(root.itertext())
    # Regex fallback
    return re.sub(r"</tell>|<tell\b[^>]*>|</text>|<text>", "", tagged_text)

def format_reward(output: str, document: str) -> float:
    """
    1.0 if the model output (final channel, tags stripped) matches the document exactly
    AND every <tell> tag has a valid type attribute ("AI" or "human").
    0.0 otherwise.
    """
    ok, _ = format_status(output, document)
    return 1.0 if ok else 0.0


def stripped_char_diff_count(text_a: str, text_b: str) -> int:
    """Approximate character difference count using sequence matching.

    Returns len(a) + len(b) - 2 * matched_chars; 0 means exact match.
    """
    if text_a == text_b:
        return 0
    matcher = SequenceMatcher(None, text_a, text_b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (len(text_a) + len(text_b) - (2 * matched))


def format_diagnostics(output: str, document: str) -> dict[str, int | str | bool]:
    """Structured format diagnostics for debugging and logging."""
    if not output:
        return {"ok": False, "reason": "empty_final", "char_diff_count": len(document)}
    indicators = parse_indicators(output)
    if not indicators:
        stripped = strip_tags(output)
        return {"ok": False, "reason": "no_tells", "char_diff_count": stripped_char_diff_count(stripped, document)}
    if any(ind.get("type") not in _VALID_TYPES for ind in indicators):
        stripped = strip_tags(output)
        return {"ok": False, "reason": "invalid_type", "char_diff_count": stripped_char_diff_count(stripped, document)}
    stripped = strip_tags(output)
    if stripped != document and _normalize(stripped) != _normalize(document):
        return {"ok": False, "reason": "text_mismatch", "char_diff_count": stripped_char_diff_count(stripped, document)}
    return {"ok": True, "reason": "ok", "char_diff_count": 0}


def format_status(output: str, document: str) -> tuple[bool, str]:
    """
    Structured format check with reason codes.
    Reasons: ok, empty_final, no_tells, invalid_type, text_mismatch.
    """
    diag = format_diagnostics(output, document)
    return bool(diag["ok"]), str(diag["reason"])


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
    Smooth margin objective in [0, 1].
    The reward is a sigmoid centered at the target margin, so pressure keeps
    increasing smoothly after the margin instead of flattening out.
    """
    a = max(-1.0, min(1.0, float(aggregate_score)))
    y = 1.0 if label == 1 else -1.0
    ya = y * a
    m = max(0.0, min(0.95, float(margin)))
    return 1.0 / (1.0 + math.exp(-_MARGIN_SIGMOID_BETA * (ya - m)))


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
    Combined reward in [0, 1]. Format is a gate: if format is invalid, return 0.
    Main signal: calibration_reward (correct classification direction).
    Optional signals (controlled by config weights, currently both 0):
      - margin_reward: push toward confident separation beyond the target margin.
      - granularity_reward: bonus for many short tells vs. one long tell.
    When both weights are 0, reward equals calibration_reward ∈ [0, 1].
    """
    if format_reward(output, document) == 0.0:
        return 0.0
    from rl_detector.frozen import aggregate
    agg = aggregate(frozen_scored)
    cal = calibration_reward(agg, label)
    mar = margin_reward(agg, label)
    gran = granularity_reward(len(frozen_scored))
    total = cal + _MARGIN_WEIGHT * mar + _GRANULARITY_WEIGHT * gran
    denom = 1.0 + _MARGIN_WEIGHT + _GRANULARITY_WEIGHT
    return total / denom


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
