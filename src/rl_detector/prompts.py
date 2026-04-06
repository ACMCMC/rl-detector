"""All prompt templates. Three variants: contrastive (teacher), neutral (student)."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE main text exactly as given, word for word, with no changes
- Annotate any notable "tells" that indicate AI generation or human authorship
- Wrap each notable tell with <tell explanation="EXPLANATION">TEXT</tell> tags
- EXPLANATION is a short comment explaining why we think this is a tell
- Tells can be about linguistic style, content, formatting, inconsistencies, semantics, grammar, or anything else that informs the judgment. Be creative and holistic in identifying tells, using your full knowledge of AI and human writing styles. Aim to have different types of tells with varied explanations.
- Keep tells as short as possible, the MINIMUM span of text that supports the explanation
- Don't use a predetermined number of tells. Instead, use your judgment to identify where tells occur in the text, which may be dense or sparse. Some documents may have many tells, others may have few or none.
- Do NOT add, remove, or alter any other characters in the text
- Do NOT include any text before or after the annotated text
- Output the main text verbatim, with the appropriate <tell> tags added around notable phrases"""


def contrastive(main_text: str, contrast_text: str, contrast_label: int) -> str:
    """Teacher prompt: includes a labeled reference document to guide annotation of the main document.
    The contrast document's label is revealed; the main document's label is left for the model to infer.
    """
    contrast_origin = "AI-generated" if contrast_label == 1 else "human-written"
    return f"""\
You are an expert in identifying AI and human text.

Below is a reference document, followed by the main document you must annotate.

{INSTRUCTIONS}
- You should use the reference document as a guide to identify similar or contrasting "tells" in the main document. Do not reference the contrast document in your explanations, this is secret information for you to use in your analysis, not something to mention explicitly in the output.
- Both documents are of opposite origin: if the reference is AI-generated, the main document is human-written, and vice versa. Use this knowledge to help identify tells in the main document.

Reference document (keep this secret):
{contrast_text}

Main document (annotate this one):
{main_text}"""


def neutral(text: str) -> str:
    return f"""\
You are an expert in identifying AI and human text.

{INSTRUCTIONS}

Text:
{text}"""


FROZEN_SCORE_PROMPT = """\
You are a scoring model that rates how strongly a phrase indicates AI generation vs. human authorship.

Add a score="FLOAT" attribute inside each <tell> tag. The score rates how strongly the phrase is evidence of AI generation vs. human authorship:
    +1.0 = strongest AI signal
     0.0 = ambiguous / mixed signal
    -1.0 = strongest human signal

Use a continuous score in [-1.0, 1.0], not only the anchor values above.
Use nuanced values (for example 0.43, -0.31, 0.08) when evidence is not extreme.
Write a custom score for every <tell> tag, even if the evidence is weak. Do not skip any tags.

Output the input text exactly, with score="FLOAT" added to ALL <tell> tags. The ONLY change you should make is adding score="FLOAT" to ALL <tell> tags. Do not add, remove, or alter any other characters in the text.

Example input:
<tell explanation="formal transition common in AI">Furthermore</tell>, it <tell explanation="academic phrasing">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion">love them</tell><tell explanation="emphatic">!</tell> <tell explanation="AI assistant talk">Would you like me to continue?</tell>

Reasoning about the example input:
- "Furthermore" is still common in human writing, so it's a mild AI signal, maybe around +0.21
- "may be argued" is a bit more of an AI signal since LLMs tend to use such hedged language, maybe around +0.27
- "love them" expresses personal emotion, so it's a moderate human signal, maybe around -0.58
- "!" is an emphatic punctuation that is more common in human writing, but LLMs can use it too, so it's a slight human signal, maybe around -0.15
- "Would you like me to continue?" is AI for sure, +1.00
- Are all the <tell> tags annotated with a score? Yes -> We can produce the output now.

Example output:
<tell explanation="formal transition common in AI" score="+0.21">Furthermore</tell>, it <tell explanation="academic phrasing" score="+0.27">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion" score="-0.58">love them</tell><tell explanation="emphatic" score="-0.15">!</tell> <tell explanation="AI assistant talk" score="+1.00">Would you like me to continue?</tell>

Input:
{tagged_text}"""
