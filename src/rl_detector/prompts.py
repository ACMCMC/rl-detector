"""All prompt templates. Three variants: directed AI, directed human, neutral."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE text exactly as given, word for word, with no changes
- Annotate any notable "tells" that indicate AI generation or human authorship
- Wrap each notable tell with <tell explanation="EXPLANATION">TEXT</tell> tags
- EXPLANATION a short sentence explaining why we think this is a tell
- Keep tells as short as possible, the minimum span of text that supports the explanation
- Take a stance. Based on the general vibe of the text, choose how to guide the annotation. However, you should still include at least 20% of the opposite type of tells to keep the annotation balanced and informative.
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
  +1.0 = strong AI signal
  +0.5 = moderate AI signal
   0.0 = ambiguous
  -0.5 = moderate human signal
  -1.0 = strong human signal

Output the input text exactly, with score="FLOAT" added to each <tell> tag. \
Do not change anything else.

Example input:
<tell explanation="formal transition common in AI">Furthermore</tell>, it <tell explanation="hedged academic phrasing">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion">felt devastated</tell>. <tell explanation="AI assistant sign-off">Would you like me to continue?</tell>

Example output:
<tell explanation="formal transition common in AI" score="0.6">Furthermore</tell>, it <tell explanation="hedged academic phrasing" score="-0.2">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion" score="-0.6">felt devastated</tell>. <tell explanation="AI assistant sign-off" score="1.0">Would you like me to continue?</tell>

Input:
{tagged_text}"""
