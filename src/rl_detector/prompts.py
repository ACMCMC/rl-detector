"""All prompt templates. Three variants: contrastive (teacher), neutral (student)."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE main text exactly as given, word for word, with no changes
- Wrap each notable "tell" of AI generation or human authorship with <tell explanation="EXPLANATION" type="TYPE">TEXT</tell> tags
- EXPLANATION is a short comment explaining why you think this is a tell
- TYPE must exactly be "AI" or "human". Every <tell> tag MUST have a type attribute
- Don't use a predetermined number of tells. Instead, use your judgment to identify where tells occur in the text, which may be dense or sparse. Some documents may have many tells, others may have few or none
- Tells can be about linguistic style, content, formatting, inconsistencies, semantics, grammar, or anything else that informs the judgment. Be creative in identifying tells, using your full knowledge and intuition to have varied and insightful explanations
- Consider the intentions, capabilities, attributes and limitations of both human writers and AI models
- Keep tells short, focused, and specific to particular phrases, words or characters in the text
- Do NOT add, remove, or alter any other characters in the text
- Do NOT include any text before or after the annotated text
- Output the main text verbatim, with the appropriate <tell> tags added around notable phrases"""


def contrastive(main_text: str, contrast_text: str, contrast_label: int) -> str:
    """Teacher prompt: includes a labeled reference document to guide annotation of the main document.
    The contrast document's label is revealed; the main document's label is left for the model to infer.
    """
    contrast_origin = "AI-generated" if contrast_label == 1 else "human-written"
    main_label = 0 if contrast_label == 1 else 1
    main_origin = "AI-generated" if main_label == 1 else "human-written"
    return f"""\
You are an expert in identifying AI and human text.

Below is a reference document, followed by the main document you must annotate.

{INSTRUCTIONS}
- You should use the reference document as a guide to identify similar or contrasting "tells" in the main document. Do not reference the contrast document in your explanations, this is secret information for you to use in your analysis, not something to mention explicitly in the output.
- Labels for this pair are known to you and should guide your annotations: reference={contrast_origin}, main={main_origin}.
- Both documents are of opposite origin: if the reference is AI-generated, the main document is human-written, and vice versa. Use this knowledge to help identify tells in the main document.

Reference document (label: {contrast_origin}, keep this secret):
{contrast_text}

Main document (label: {main_origin}, annotate this one):
{main_text}"""


def neutral(text: str) -> str:
    return f"""\
You are an expert in identifying AI and human text.

{INSTRUCTIONS}

Text:
{text}"""


FROZEN_SCORE_PROMPT = """\
You are a scoring model that rates how strongly a phrase indicates AI generation vs. human authorship.

Each <tell> tag already has a type="AI" or type="human" attribute set by the annotator. Add a score="FLOAT" attribute inside each <tell> tag. The score rates how strongly the phrase is evidence of AI generation vs. human authorship:
    +1.0 = strongest AI signal
     0.0 = ambiguous / mixed signal
    -1.0 = strongest human signal

Use a continuous score in [-1.0, 1.0], not only the anchor values above.
Use nuanced values (for example 0.43, -0.31, 0.08) when evidence is not extreme.
Write a custom score for every <tell> tag, even if the evidence is weak. Do not skip any tags.
Treat all <tell> tags independently, using only the text and explanation within each tag to determine the score. Do not let one tag influence the score of another; there is no need to compare tags to each other.

Output the input text exactly, with score="FLOAT" added to ALL <tell> tags. The ONLY change you should make is adding score="FLOAT" to ALL <tell> tags. Do not add, remove, or alter any other characters in the text (including the existing type= attributes).

Example input:
<tell explanation="formal transition common in AI" type="AI">Furthermore</tell>, it <tell explanation="academic phrasing" type="AI">may be argued</tell> that dogs are loyal <tell explanation="em dash" type="AI">—</tell> I <tell explanation="first-person emotion" type="human">lo<tell explanation="typo" type="human">ev</tell> them</tell><tell explanation="emphatic" type="human">!</tell> <tell explanation="AI assistant talk" type="AI">Would you like me to continue?</tell>

Reasoning about the example input:
- "Furthermore" is still common in human writing, so it's a mild AI signal, maybe around +0.21
- "may be argued" is a bit more of an AI signal since LLMs tend to use such hedged language, maybe around +0.27
- "—" is hard for a human to type on a keyboard, and AI often uses it, so it's a moderate AI signal, maybe around +0.68
- "love them" expresses personal emotion, but AI also know how to write emotion, so it's a mild human signal, maybe around -0.28
- "ev" is a typo, strong human, -0.78
- "!" is an emphatic punctuation that is more common in human writing, but LLMs can use it too, so it's a slight human signal, -0.15
- "Would you like me to continue?" is AI for sure, +1.00
- Are all the <tell> tags annotated with a score? Yes -> We can produce the output now.

Example output:
<tell explanation="formal transition common in AI" type="AI" score="+0.21">Furthermore</tell>, it <tell explanation="academic phrasing" type="AI" score="+0.27">may be argued</tell> that dogs are loyal <tell explanation="em dash" type="AI" score="+0.68">—</tell> I <tell explanation="first-person emotion" type="human" score="-0.28">lo< tell explanation="typo" type="human" score="-0.78">ev</tell> them</tell><tell explanation="emphatic" type="human" score="-0.15">!</tell> <tell explanation="AI assistant talk" type="AI" score="+1.00">Would you like me to continue?</tell>

Input:
{tagged_text}"""
