"""All prompt templates. Three variants: contrastive (teacher), neutral (student)."""

from rl_detector.fewshots import pick_fewshot
from rl_detector.config import CFG

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE main text exactly as given, word for word, with no changes
- Wrap each notable "tell" of AI generation or human authorship with <tell explanation="EXPLANATION" type="TYPE">TEXT</tell> tags
- In every <tell> tag, attribute values MUST use double quotes only, never single quotes
- EXPLANATION is a short and convincing comment on why this tell is a good signal
- TYPE must exactly be "AI" or "human". Every <tell> tag MUST have a type attribute
- Add at least one <tell> tag in the main document, tells can be nested
- Don't use a predetermined number of tells. Instead, use your judgment to identify where tells occur in the text, which may be dense or sparse. Some documents may have many tells, others may have few
- Tells can be about linguistic style, content, formatting, inconsistencies, semantics, grammar, or anything else that informs the judgment. Be creative in identifying tells, using your full knowledge and intuition to have varied and insightful explanations
- Get into the head of each writer: consider the intention and the context of the writing to act as a detective uncovering the subtle clues and inconsistencies that reveal the origin of the text
- Keep tells short, focused, and specific to particular phrases, words or characters in the text
- COPY THE MAIN TEXT CHARACTER BY CHARACTER, including punctuation, spacing, capitalization, and line breaks. Also preserve the Unicode characters and formatting of the original text, (i.e. ‑ is not the same as -). And if the original text has typos, missing punctuation, or any other "imperfections", reproduce those exactly as they are, we need the output to be exactly identical to the input text
- OUTPUT ONLY THE ORIGINAL MAIN TEXT WITH <tell ...> TAGS INSERTED; DO NOT OUTPUT ANY EXPLANATION, ANALYSIS, REASONING, PREFACE, OR EXTRA TEXT"""


def contrastive(main_text: str, contrast_text: str, contrast_label: int, main_label_hint: int, show_labels: bool) -> str:
    """Teacher prompt: always has labels internally; optionally hides them from model text."""
    contrast_origin = "AI" if contrast_label == 1 else "human"
    main_origin = "AI" if main_label_hint == 1 else "human"

    if show_labels:
        reference_header = f"Reference document (label: {contrast_origin}):"
        main_origin_line = f"Main document (label hint: {main_origin}, annotate this one):"
    else:
        reference_header = "Reference document:"
        main_origin_line = "Main document (annotate this one):"

    use_fewshot = getattr(getattr(CFG, "training", {}), "use_fewshot_examples", True)
    if use_fewshot:
        fewshot1 = pick_fewshot(
            main_text=main_text,
            contrast_text=contrast_text,
            contrast_label=contrast_label,
            main_label_hint=main_label_hint,
            show_labels=show_labels,
        )
        fewshot2 = pick_fewshot(
            main_text=main_text,
            contrast_text=contrast_text,
            contrast_label=contrast_label,
            main_label_hint=main_label_hint,
            show_labels=show_labels,
            seed=2262 + 1,
        )
        fewshot_block = (
            "Here are two examples of how to annotate a document, with explanations for each tell:\n"
            f"Example 1:\n```\n{fewshot1}\n```\n"
            f"Example 2:\n```\n{fewshot2}\n```\n"
        )
    else:
        fewshot_block = ""

    return f"""\
You are an expert in identifying AI and human text that pays close attention to subtle "tells" that can reveal the origin of a document.

You will be given two documents, a reference document and a main document.

{INSTRUCTIONS}
- You should start by looking at the reference document to understand the style and content of that origin (AI or human)
- The reference document always has the opposite label from the main document
- Then you should analyze the main document, comparing it to the reference to see how it differs, and what features we notice in the main document that reveal its origin in contrast to the reference (but also consider the unique tells in the main document on their own)
- Your final output must contain only the annotated main document. Never output any part of the reference document

{fewshot_block}

{reference_header}
```
{contrast_text}
```

{main_origin_line}
```
{main_text}
```"""


def neutral(text: str) -> str:
    return f"""\
You are an expert in identifying AI and human text that pays close attention to subtle "tells" that can reveal the origin of a document.

{INSTRUCTIONS}

Text:
```
{text}
```"""


FROZEN_SCORE_PROMPT = """\
You are a scoring model that rates how strongly a <tell> indicates AI generation vs. human authorship.

Each <tell> tag already has a type="AI" or type="human" attribute set by the annotator. Add a score="FLOAT" attribute inside each <tell> tag. The score rates how strongly the <tell> is evidence of AI generation vs. human authorship:
    +1.0 = strongest AI signal
     0.0 = ambiguous / mixed signal
    -1.0 = strongest human signal

Write a custom score for every <tell> tag. Do not skip any tags.
Treat all <tell> tags independently, using only the text and explanation within each tag to determine the score. Do not let one tag influence the score of another; there is no need to compare tags to each other.
Be critical, use your full knowledge and intuition to discern what's really a tell and what's not.

Output the ENTIRE input text exactly, with score="FLOAT" added to ALL <tell> tags. The ONLY change you should make is adding score="FLOAT" to ALL <tell> tags. Do not add, remove, or alter any other characters in the text.

Example input:
<tell explanation="formal transition common in AI" type="AI">Furthermore</tell>, it <tell explanation="academic phrasing" type="AI">may be argued</tell> that dogs are loyal <tell explanation="em dash" type="AI">—</tell> I <tell explanation="first-person emotion" type="human">lo<tell explanation="typo" type="human">ev</tell> them</tell><tell explanation="emphatic" type="human">!</tell> <tell explanation="AI assistant talk" type="AI">Would you like me to continue?</tell> Bye!

Reasoning about the example input:
- "Furthermore" is still common in human writing, so it's a mild AI signal, maybe around +0.13
- "may be argued" is a bit more of an AI signal since LLMs tend to use such hedged language, maybe around +0.19
- "—" is hard for a human to type on a keyboard, and AI often uses it, so it's a moderate AI signal, maybe around +0.68
- "love them" expresses personal emotion, but AI also know how to write emotion, so it's a mild human signal, maybe around -0.28
- "ev" is a typo, strong human, -0.78
- "!" is an emphatic punctuation that is more common in human writing, but LLMs can use it too, so it's a slight human signal, -0.15
- "Would you like me to continue?" is AI for sure, +1.00
- Are all the <tell> tags annotated with a score? Yes -> We can produce the output now.

Example output:
<tell explanation="formal transition common in AI" type="AI" score="+0.13">Furthermore</tell>, it <tell explanation="academic phrasing" type="AI" score="+0.19">may be argued</tell> that dogs are loyal <tell explanation="em dash" type="AI" score="+0.68">—</tell> I <tell explanation="first-person emotion" type="human" score="-0.28">lo< tell explanation="typo" type="human" score="-0.78">ev</tell> them</tell><tell explanation="emphatic" type="human" score="-0.15">!</tell> <tell explanation="AI assistant talk" type="AI" score="+1.00">Would you like me to continue?</tell> Bye!

Input:
```
{tagged_text}
```"""
