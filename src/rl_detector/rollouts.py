"""Generate rollouts from the current policy using contrastive (teacher) prompts."""

import asyncio
import logging
import re
from difflib import SequenceMatcher

import tinker

from rl_detector.config import CFG
from rl_detector.prompts import training_prompt
from rl_detector.rewards import _parse_xml_response, _try_fix_xml

logger = logging.getLogger(__name__)

_TELL_TAG_RE = re.compile(r"<tell\b[^>]*>|</tell>|</?text>")


def _strip_tags(text: str) -> str:
    """Strip tell and text wrapper tags. Uses XML if parseable, else regex."""
    root = _parse_xml_response(text)
    if root is not None:
        return "".join(root.itertext())
    return _TELL_TAG_RE.sub("", text)


def fix_xml_quotes(response_text: str) -> str:
    """
    Normalize response_text to valid XML: fix single-quoted attributes, escape bare &.
    Returns the fixed text if fixable, or the original text unmodified if not.
    """
    if _parse_xml_response(response_text) is not None:
        return response_text  # already valid
    wrapped = response_text if response_text.startswith("<text>") else f"<text>{response_text}</text>"
    fixed = _try_fix_xml(wrapped)
    if _parse_xml_response(fixed) is not None:
        # Strip the wrapper we added if the original didn't have one
        if not response_text.startswith("<text>"):
            fixed = fixed[len("<text>"):-len("</text>")]
        return fixed
    return response_text


def try_fix_response(response_text: str, document: str, max_fix_ratio: float = 0.05) -> str | None:
    """
    If the stripped response is close to document (< max_fix_ratio of chars changed),
    patch response_text so the stripped version matches document exactly.
    Tag structure is preserved; only non-tag text is corrected.
    Returns the fixed response_text, or None if the diff is too large or can't be applied cleanly.
    """
    stripped = _strip_tags(response_text)
    if stripped == document:
        return None

    matcher = SequenceMatcher(None, stripped, document, autojunk=False)
    opcodes = matcher.get_opcodes()

    # Count chars that differ on either side: deletions from stripped (i2-i1),
    # insertions into document (j2-j1), or replacements (max of both sides).
    n_changed = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in opcodes if tag != "equal"
    )
    if n_changed == 0 or n_changed / max(len(document), 1) > max_fix_ratio:
        return None

    # Build mapping: stripped index → position in response_text, skipping tag spans.
    s2r: list[int] = []
    ri = 0
    while ri < len(response_text) and len(s2r) < len(stripped):
        if response_text[ri] == "<":
            end = response_text.find(">", ri)
            if end != -1:
                ri = end + 1
                continue
        s2r.append(ri)
        ri += 1

    if len(s2r) != len(stripped):
        return None  # tag parse failed

    result: list[str] = []
    prev = 0
    for opcode, i1, i2, j1, j2 in opcodes:
        if opcode == "equal":
            if i2 > i1:
                end = s2r[i2 - 1] + 1
                result.append(response_text[prev:end])
                prev = end
        elif opcode == "replace":
            start = s2r[i1] if i1 < len(s2r) else len(response_text)
            result.append(response_text[prev:start])
            result.append(document[j1:j2])
            end = (s2r[i2 - 1] + 1) if 0 < i2 <= len(s2r) else start
            prev = end
        elif opcode == "delete":
            start = s2r[i1] if i1 < len(s2r) else len(response_text)
            result.append(response_text[prev:start])
            end = (s2r[i2 - 1] + 1) if 0 < i2 <= len(s2r) else start
            prev = end
        elif opcode == "insert":
            pos = s2r[i1] if i1 < len(s2r) else len(response_text)
            result.append(response_text[prev:pos])
            result.append(document[j1:j2])
            prev = pos

    result.append(response_text[prev:])
    fixed = "".join(result)

    if _strip_tags(fixed) != document:
        return None

    return fixed


_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\|>\s*([^<\s]+)\s*<\|message\|>(.*?)(?=(?:<\|channel\|>)|(?:<\|end\|>)|(?:<\|return\|>)|$)",
    re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(r"<(?:think|thinking|reasoning|analysis)>.*?</(?:think|thinking|reasoning|analysis)>", re.DOTALL | re.IGNORECASE)


def extract_response_text(text: str) -> str:
    """Best-effort extraction of user-facing response text from a full completion."""
    channel_blocks = list(_CHANNEL_BLOCK_RE.finditer(text))
    if channel_blocks:
        for m in channel_blocks:
            if m.group(1).strip().lower() == "final":
                return m.group(2).strip()
        return channel_blocks[-1].group(2).strip()

    without_thinking = _THINK_BLOCK_RE.sub("", text).strip()
    without_start_end_backticks = without_thinking.strip("`")
    return without_start_end_backticks if without_start_end_backticks else text.strip()


async def generate_rollouts(
    sampling_client,
    tokenizer,
    document: str,
    contrast_docs: list[dict],
    main_label_hints: list[int],
    show_labels_flags: list[bool],
    K: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate K rollouts using contrastive (teacher) prompts.

    Each rollout uses a different contrast document (opposite label from the main document)
    so the model sees a diverse set of reference comparisons. This is the teacher side of
    self-distillation: the contrast is privileged context that will not be present at
    optimization time (which uses the neutral prompt).

    Args:
        contrast_docs: list of K dicts with {"text", "label"}, one per rollout.
                       Each must have the opposite label from the main document.
    """
    if K is None:
        K = CFG.training.k
    assert len(contrast_docs) == K, f"expected {K} contrast docs, got {len(contrast_docs)}"
    assert len(main_label_hints) == K, f"expected {K} main label hints, got {len(main_label_hints)}"
    assert len(show_labels_flags) == K, f"expected {K} show-label flags, got {len(show_labels_flags)}"

    async def _sample_one(i: int, contrast: dict, main_label_hint: int, show_labels: bool) -> dict:
        prompt_text = training_prompt(
            document,
            contrast_text=contrast["text"],
            contrast_label=contrast["label"],
            main_label_hint=main_label_hint,
            show_labels=show_labels,
        )
        prompt_text_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text_formatted)
        model_input = tinker.ModelInput.from_ints(prompt_tokens)

        # use the same fixed seed for every rollout — diversity comes from different contrast docs
        rollout_seed = seed
        sampling_params = tinker.SamplingParams(
            max_tokens=CFG.sampling.max_tokens,
            temperature=CFG.sampling.temperature,
            top_p=CFG.sampling.top_p,
            reasoning_effort=CFG.sampling.reasoning_effort,
            seed=rollout_seed,
        )

        sampled = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        seq = sampled.sequences[0]
        completion_tokens = list(seq.tokens)
        if seq.logprobs is not None:
            completion_logprobs = list(seq.logprobs)
        else:
            completion_logprobs = [0.0] * len(completion_tokens)
        assert any(lp != 0.0 for lp in completion_logprobs), "completion_logprobs are all 0.0"

        completion_text = tokenizer.decode(completion_tokens)

        response_text = extract_response_text(completion_text)
        # Normalize XML quotes/escaping before the typo fixer so strip_tags works correctly
        response_text = fix_xml_quotes(response_text)
        fixed = try_fix_response(response_text, document)
        wrong_response_text = None
        was_text_fixed = False
        if fixed is not None:
            wrong_response_text = response_text
            response_text = fixed
            if wrong_response_text in completion_text:
                completion_text = completion_text.replace(wrong_response_text, fixed, 1)
            else:
                logger.warning("rollout %d: response_text not found verbatim in completion_text, using fixed response as completion", i)
                completion_text = fixed
            completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
            was_text_fixed = True
            logger.info("rollout %d: fixed %d char(s) of typo in response_text", i, sum(i2 - i1 for tag, i1, i2, j1, j2 in SequenceMatcher(None, wrong_response_text, fixed).get_opcodes() if tag != "equal"))

        # Find where the response starts within completion_text so we can mask
        # reasoning tokens from the loss. The reasoning trace is conditioned on
        # the directed prompt (contrast doc, hints) and is incomparable under the
        # neutral prompt, so it should not contribute to the gradient or IS ratio.
        response_start_idx = completion_text.find(response_text)
        if response_start_idx >= 0:
            prefix = completion_text[:response_start_idx]
            n_reasoning_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        else:
            # response_text not found verbatim — treat all tokens as response (no masking)
            logger.warning("rollout %d: could not locate response_text in completion_text, n_reasoning_tokens=0", i)
            n_reasoning_tokens = 0

        return {
            "completion_text": completion_text,
            "response_text": response_text,
            "completion_tokens": completion_tokens,
            "completion_logprobs": completion_logprobs,
            "n_reasoning_tokens": n_reasoning_tokens,
            "contrast_label": contrast["label"],
            "main_label_hint": main_label_hint,
            "show_labels": show_labels,
            "was_text_fixed": was_text_fixed,
            "wrong_response_text": wrong_response_text,
        }

    tasks = [_sample_one(i, contrast, main_label_hints[i], show_labels_flags[i]) for i, contrast in enumerate(contrast_docs)]
    return await asyncio.gather(*tasks)
