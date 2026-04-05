"""All prompt templates. Three variants: directed AI, directed human, neutral."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE text exactly as given, word for word, with no changes
- Annotate any notable "tells" that indicate AI generation or human authorship
- Wrap each notable tell with <tell explanation="EXPLANATION">TEXT</tell> tags
- EXPLANATION is a short comment explaining why we think this is a tell
- Tells can be about linguistic style, content, formatting, inconsistencies, semantics, grammar, or anything else that informs the judgment. Be creative and holistic in identifying tells, using your full knowledge of AI and human writing styles. Aim to have different types of tells with varied explanations.
- Keep tells as short as possible, the MINIMUM span of text that supports the explanation
- Take a stance. Based on whether you believe the text is AI or human, choose how to guide the annotation. However, you should still include around 10-25% (as appropriate) of the opposite type of tells to keep the annotation balanced and informative.
- Don't use a predetermined number of tells. Instead, use your judgment to identify where tells occur in the text, which may be dense or sparse. Some documents may have many tells, others may have few or none.
- Do NOT add, remove, or alter any other characters in the text
- Do NOT include any text before or after the annotated text
- Output the original text verbatim, with the appropriate <tell> tags added around notable phrases"""


def directed_ai(text: str) -> str:
    return f"""\
You are an expert in identifying AI and human text.

{INSTRUCTIONS}

You believe this text is AI.

Text:
{text}"""


def directed_human(text: str) -> str:
    return f"""\
You are an expert in identifying AI and human text.

{INSTRUCTIONS}

You believe this text is human.

Text:
{text}"""


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
<tell explanation="formal transition common in AI">Furthermore</tell>, it <tell explanation="hedged academic phrasing">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion">felt devastated</tell>. <tell explanation="AI assistant sign-off">Would you like me to continue?</tell>

Reasoning about the example input:
- "Furthermore" is still common in human writing, so it's a mild AI signal, maybe around +0.21
- "may be argued" is a bit more of an AI signal due to the hedged phrasing, maybe around +0.27
- "felt devastated" requires personal emotion, so it's a moderate human signal, maybe around -0.58
- "Would you like me to continue?" confirms the AI assistant sign-off, so it's certain, +1.00
- Are all the <tell> tags annotated with a score? Yes -> We can produce the output now.

Example output:
<tell explanation="formal transition common in AI" score="+0.21">Furthermore</tell>, it <tell explanation="hedged academic phrasing" score="+0.27">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion" score="-0.58">felt devastated</tell>. <tell explanation="AI assistant sign-off" score="+1.00">Would you like me to continue?</tell>

Input:
{tagged_text}"""
