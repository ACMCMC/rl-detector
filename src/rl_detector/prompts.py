"""All prompt templates. Three variants: directed AI, directed human, neutral."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE text exactly as given, word for word, with no changes
- Annotate any notable "tells" that indicate AI generation or human authorship
- Wrap each notable tell with <tell explanation="EXPLANATION">TEXT</tell> tags
- EXPLANATION a short sentence explaining why we think this is a tell
- Tells can be about linguistic style, content, formatting, inconsistencies, semantics, grammar, or anything else that informs the judgment
- Be creative and holistic in identifying tells, using your full knowledge of AI and human writing styles
- Keep tells as short as possible, the MINIMUM span of text that supports the explanation
- Take a stance. Based on the general vibe of the text, choose how to guide the annotation. However, you should still include at least 20% of the opposite type of tells to keep the annotation balanced and informative.
- Don't use a predetermined number of tells. Instead, use your judgment to identify as many or as few tells as needed to support the stance you took, while still including some of the opposite type of tells for balance.
- Do NOT add, remove, or alter any other characters in the text
- Do NOT include any text before or after the annotated text
- Output the original text verbatim, with the appropriate <tell> tags added around notable phrases"""


def directed_ai(text: str) -> str:
    return f"""\
You are an expert in identifying AI-generated text.

You believe this text is AI-generated.

{INSTRUCTIONS}

Text:
{text}"""


def directed_human(text: str) -> str:
    return f"""\
You are an expert in identifying human-written text.

You believe this text is human-written.

{INSTRUCTIONS}

Text:
{text}"""


def neutral(text: str) -> str:
    return f"""\
You are an expert in identifying AI-generated and human-written text.

{INSTRUCTIONS}

Text:
{text}"""


FROZEN_SCORE_PROMPT = """\
Add a score="FLOAT" attribute inside each <tell> tag. The score rates how strongly the \
phrase is evidence of AI generation vs. human authorship:
    +1.0 = strongest AI signal
     0.0 = ambiguous / mixed signal
    -1.0 = strongest human signal

Use a continuous score in [-1.0, 1.0], not only the anchor values above.
Prefer nuanced values (for example 0.73, -0.41, 0.08) when evidence is not extreme.
Write a score for every <tell> tag, even if the evidence is weak. Do not skip any tags.

Output the input text exactly, with score="FLOAT" added to each <tell> tag. \
Do not change anything else.

Example input:
<tell explanation="formal transition common in AI">Furthermore</tell>, it <tell explanation="hedged academic phrasing">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion">felt devastated</tell>. <tell explanation="AI assistant sign-off">Would you like me to continue?</tell>

Example output:
<tell explanation="formal transition common in AI" score="0.64">Furthermore</tell>, it <tell explanation="hedged academic phrasing" score="0.17">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion" score="-0.58">felt devastated</tell>. <tell explanation="AI assistant sign-off" score="0.93">Would you like me to continue?</tell>

Input:
{tagged_text}"""
