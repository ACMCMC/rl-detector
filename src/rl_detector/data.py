"""Dataset loading."""

from datasets import load_dataset

from rl_detector.config import CFG


def load_docs(split: str | None = None) -> list[dict]:
    """Load Ateeqq/AI-and-Human-Generated-Text as flat {"text", "label"} list."""
    if split is None:
        split = CFG.data.split
    ds = load_dataset(CFG.data.dataset, split=split)
    docs = []
    for row in ds:
        text = row["abstract"]
        if text and text.strip():
            docs.append({"text": text.strip(), "label": int(row["label"])})
    return docs


def iter_balanced_steps(docs: list[dict], docs_per_step: int = 4):
    """
    Yield lists of docs_per_step docs, half AI half human, until either pool runs out.
    """
    ai_docs = [d for d in docs if d["label"] == 1]
    human_docs = [d for d in docs if d["label"] == 0]

    n_ai = docs_per_step // 2
    n_human = docs_per_step - n_ai

    i_ai, i_human = 0, 0
    while i_ai + n_ai <= len(ai_docs) and i_human + n_human <= len(human_docs):
        yield ai_docs[i_ai: i_ai + n_ai] + human_docs[i_human: i_human + n_human]
        i_ai += n_ai
        i_human += n_human
