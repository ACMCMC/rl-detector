"""All prompt templates. Three variants: directed AI, directed human, neutral."""

INSTRUCTIONS = """\
Rules:
- Reproduce the ENTIRE text exactly as given, word for word, with no changes
- Wrap each notable phrase or sentence with <tell explanation="EXPLANATION">PHRASE</tell> tags
- EXPLANATION is one sentence describing why this phrase is a tell
- Keep tells balanced, providing evidence for both AI generation and human authorship where applicable
- Do NOT add, remove, or alter any other characters in the text
- Do NOT include any text before or after the annotated text
- Output the original text verbatim, with the appropriate <tell> tags added around notable phrases"""


def directed_ai(text: str) -> str:
    return f"""\
You are an expert in identifying AI-generated text.

You believe that this text is AI-generated and are trying to find evidence to support that belief.

{INSTRUCTIONS}

Text:
{text}"""


def directed_human(text: str) -> str:
    return f"""\
You are an expert in identifying human-written text.

You believe that this text is human-written and are trying to find evidence to support that belief.

{INSTRUCTIONS}

Text:
{text}"""


def neutral(text: str) -> str:
    return f"""\
You are an expert in distinguishing AI-generated text from human-written text.

Reproduce the text below exactly, wrapping phrases that are notable indicators of either \
AI generation or human authorship with <tell explanation="..."> tags.

{INSTRUCTIONS}

Text:
{text}"""


FROZEN_SCORE_PROMPT = """\
Add a score="FLOAT" attribute inside each <tell> tag. The score rates how strongly the \
phrase is evidence of AI generation vs. human authorship:
  +1.0 = extremely strong AI signal (very rare)
  +0.5 = moderate AI signal
   0.0 = ambiguous
  -0.5 = moderate human signal
  -1.0 = extremely strong human signal (very rare)

Most scores should fall between -0.7 and +0.7.

Output the input text exactly, with score="FLOAT" added to each <tell> tag. \
Do not change anything else.

Example input:
The <tell explanation="formal transition common in AI">furthermore</tell>, it <tell explanation="hedged academic phrasing">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion">felt devastated</tell>.

Example output:
The <tell explanation="formal transition common in AI" score="0.6">furthermore</tell>, it <tell explanation="hedged academic phrasing" score="-0.2">may be argued</tell> that dogs are loyal. I <tell explanation="first-person emotion" score="-0.6">felt devastated</tell>.

Input:
{tagged_text}"""
