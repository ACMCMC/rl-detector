"""Few-shot banks for annotation prompts, with deterministic rotation."""

from __future__ import annotations

import hashlib

FEWSHOT_SEED = 2262

# diverse examples: style, punctuation, structure, factual weirdness, hesitations, persona
FEWSHOT_EXAMPLES: list[str] = [
    """Example:
Input:
```
In conclusion, this framework provides a robust path forward. Furthermore, it is important to note that future work may improve accuracy, though we would like to note that our work is not necessarily dependent on such improvements.
```
Output:
```
In <tell explanation="this transition sounds very polished and template-like" type="AI">conclusion</tell>, this framework provides a <tell explanation="strong claim but not much concrete detail" type="AI">robust path forward</tell>. <tell explanation="formal connector often seen in assistant writing" type="AI">Furthermore</tell>, it is important to note that <tell explanation="safe hedge that sounds model-generated" type="AI">future work <tell explanation="nuanced relevance" type="human">may</tell> improve accuracy</tell>, though <tell explanation="direct speaker reference" type="human">we</tell> would like to note that <tell explanation="careful caveat aiming to protect from criticism" type="human">our work is not necessarily dependent on such improvements</tell>.
```
""",
    """Example:
Input:
```
I met her at the station, we talked for 5 mins, then I relized I had the wrong ticket.
```
Output:
```
I met her at the <tell explanation="specific place detail feels like lived experience" type="human">station</tell>, we talked for <tell explanation="casual short form people use in quick writing" type="human">5 mins</tell>, then I <tell explanation="small spelling mistake suggests fast human typing" type="human">relized</tell> I had the wrong ticket.
```
""",
    """Example:
Input:
```
1) gather data 2) clean it 3) run model 4) report results, easy.
```
Output:
```
<tell explanation="packed step list has a mechanical rhythm" type="AI">1) gather data 2) clean it 3) run model 4) report results</tell>, <tell explanation="casual ending sounds like a human aside" type="human">easy.</tell>
```
""",
    """Example:
Input:
```
This analysis is nuanced, and both perspectives have merit; however, the final decision depends on context.
```
Output:
```
This analysis is <tell explanation="very balanced framing sounds like assistant tone" type="AI">nuanced, and both perspectives have merit</tell>; however, the final decision <tell explanation="careful caveat feels generic and model-like" type="AI">depends on context</tell>.
```
""",
    """Example:
Input:
```
No way!! I finally fixed the fucking unity meta quest oom bug after like three lucozades and one (or two!!!) terrible ideas lmao
```
Output:
```
<tell explanation="emotional expression feels spontaneous" type="human">No way<tell explanation="double exclamation mark" type="human">!!</tell></tell> I <tell explanation="expresses desperation" type="human">finally</tell> fixed the <tell explanation="swearing" type="human">fucking</tell> <tell explanation="specific technical issue suggests real experience" type="human">unity meta quest oom bug</tell> after <tell explanation="colloquial quantifier" type="human">like three <tell explanation="lucozade is only available in UK, suggests real-world grounding" type="human">lucozades</tell></tell> and one <tell explanation="hesitation and uncertainty feels human" type="human">(or two<tell explanation="multiple exclamation marks, emotionally charged" type="human">!!!</tell>)</tell> <tell explanation="self-deprecating humor is a natural human touch" type="human">terrible ideas</tell> <tell explanation="internet slang" type="human">lmao</tell>
```
""",
    """Example:
Input:
```
The implementation demonstrates scalability, interpretability, and adaptability across domains. Would you like me to explain each point?
```
Output:
```
The implementation demonstrates <tell explanation="stacked abstract terms are common in generated technical text" type="AI">scalability, interpretability, and adaptability</tell> across <tell explanation="very broad claim with little grounding" type="AI">domains</tell>. <tell explanation="AI assistant comment" type="AI">Would you like me to explain each point?</tell>
```
""",
    """Example:
Input:
```
The cat sat on the keyboard and now my terminal says qqqqq and I literally cant stop laughing lololol
```
Output:
```
The <tell explanation="odd concrete scene feels like a real moment" type="human">cat sat on the keyboard</tell> and now my terminal says <tell explanation="messy repeated letters look like accidental typing" type="human">qqqqq</tell> and I <tell explanation="personal speaking style" type="human">literally</tell> <tell explanation="typo suggests fast, careless typing" type="human">cant</tell> stop laughing <tell explanation="internet slang and repetition feels very human" type="human">lololol</tell>
```
""",
    """Example:
Input:
```
This proposal aims to optimize stakeholder alignment while minimizing operational friction in cross-functional workflows.
```
Output:
```
This proposal aims to <tell explanation="corporate wording is polished but vague" type="AI">optimize stakeholder alignment</tell> while minimizing <tell explanation="buzzword-heavy phrase sounds generated" type="AI">operational friction in cross-functional workflows</tell>.
```
""",
    """Example:
Input:
```
Honestly, I changed my mind halfway thruogh, but then the second draft kind of worked.
```
Output:
```
<tell explanation="informal opener feels conversational" type="human">Honestly</tell>, I changed <tell explanation="reference to self-consciousness" type="human">my mind</tell> halfway thr<tell explanation="typo" type="human">uo</tell>gh, but then the second draft <tell explanation="hedged phrasing sounds like real uncertainty" type="human">kind of</tell> worked.
```
""",
    """Example:
Input:
```
To ensure fairness, transparency, and accountability, we adopt a principled evaluation framework for all outcomes.
```
Output:
```
To ensure <tell explanation="three-value list is a common formal pattern" type="AI">fairness, transparency, and accountability</tell>, we adopt a <tell explanation="generic framework claim" type="AI">principled evaluation framework for all outcomes</tell>.
```
""",
]


def pick_fewshot(main_text: str, contrast_text: str, contrast_label: int, main_label_hint: int, show_labels: bool, seed: int = FEWSHOT_SEED) -> str:
    """Pick one few-shot example deterministically from prompt inputs and seed."""
    key = f"{seed}|{contrast_label}|{main_label_hint}|{int(show_labels)}|{main_text[:128]}|{contrast_text[:128]}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(FEWSHOT_EXAMPLES)
    return FEWSHOT_EXAMPLES[idx]
