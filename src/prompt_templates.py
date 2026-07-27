"""Advanced prompt engineering templates.

Builds the actual LLM prompt from (a) user-selected style inputs and
(b) 1-3 relevant article summaries, loaded from Mudit/Zahra's real
handoff file: knowledge_base/filtered/filtered_documents.json.

Only title, summary, and source_filename are used as content — keywords
and tags are intentionally not fed to the LLM (team decision).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Template library — loaded from /templates/prompt_templates.json, the
# same file app.py reads. Each template is a fill-in-the-blank STRUCTURE
# plus a worked EXAMPLE, used to steer the LLM's output shape without
# hand-writing full posts.
# ---------------------------------------------------------------------
_TEMPLATE_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "prompt_templates.json"
)


def load_template_library() -> Dict[str, Dict[str, str]]:
    with open(_TEMPLATE_LIBRARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TEMPLATE_LIBRARY: Dict[str, Dict[str, str]] = load_template_library()


def format_source_context(source_articles: List[Dict[str, Any]]) -> str:
    """Turns 1-3 filtered article summaries into a labeled context block.

    Only title, summary, and source_filename are treated as real content
    fed to the LLM — keywords/tags are deliberately NOT included (team
    decision: title + summary are enough; tags now hold topic categories,
    not a reliable newsletter/podcast type label, so we don't try to
    derive a type from them anymore).
    """
    blocks = []
    for i, article in enumerate(source_articles, start=1):
        blocks.append(
            f"Source {i} (from \"{article['title']}\"):\n"
            f"{article['summary']}\n"
            f"[Cite as: {article['source_filename']}]"
        )
    return "\n\n".join(blocks)


def build_prompt(payload: Dict[str, Any]) -> str:
    """Builds the full LLM prompt from user inputs + filtered source articles.

    payload shape (matches knowledge_base/filtered/filtered_documents.json):
        {
            "inputs": {topic, word_count, language, tone, style, goal,
                       target_audience, call_to_action, template},
            "source_articles": [ {title, summary, source_filename, ...}, ... ]
        }
    """
    inputs = payload["inputs"]
    template_name = inputs["template"]

    if template_name not in TEMPLATE_LIBRARY:
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Valid options: {list(TEMPLATE_LIBRARY.keys())}"
        )

    template = TEMPLATE_LIBRARY[template_name]
    context = format_source_context(payload["source_articles"])

    return f"""Write a LinkedIn post using this structural pattern:
{template['structure']}

Example of this pattern in use:
{template['example']}

Ground the post in this real source material. Reference it specifically —
do not write generically, and do not invent facts not present below:
{context}

Style parameters:
- Topic: {inputs['topic']}
- Tone: {inputs['tone']}
- Style: {inputs['style']}
- Length: {inputs['word_count']} words
- Goal: {inputs['goal']}
- Audience: {inputs['target_audience']}
- End with a call to action: {inputs['call_to_action']}

Cite the source article(s) explicitly in the post (e.g. "In Lenny's
newsletter on X..."). Do not fabricate statistics, quotes, or claims not
present in the source material above."""


# ---------------------------------------------------------------------
# REAL data — loaded from Mudit/Zahra's actual handoff file. This is
# the one and only handoff location, per team agreement: they filter
# and hand us title + summary (+ some extra metadata we don't use,
# e.g. keywords — deliberately not fed to the LLM, per team decision).
# ---------------------------------------------------------------------
_FILTERED_DOCS_PATH = (
    Path(__file__).resolve().parents[1]
    / "knowledge_base" / "filtered" / "filtered_documents.json"
)


def load_filtered_articles(count: int = 3) -> List[Dict[str, Any]]:
    """Loads the pre-filtered articles Mudit/Zahra hand off for this
    request. As of today this file is a static 2-entry sample; once
    their real per-topic filtering is live, this same function keeps
    working unchanged — only the file's contents change, not this code.
    """
    with open(_FILTERED_DOCS_PATH, "r", encoding="utf-8") as f:
        all_entries = json.load(f)

    # Light defensive check only — real filtering already happened
    # upstream, so we're not re-filtering here, just guarding against
    # a genuinely broken/empty record.
    usable = [entry for entry in all_entries if entry.get("summary")]
    return usable[:count]


FILTERED_SOURCE_ARTICLES: List[Dict[str, Any]] = load_filtered_articles(3)

DUMMY_PAYLOAD: Dict[str, Any] = {
    "inputs": {
        "topic": "shipping AI features without proper evaluation",
        "word_count": "150-500",
        "language": "English",
        "tone": "Professional",
        "style": "Storytelling",
        "goal": "Build Personal Brand",
        "target_audience": "Product Managers",
        "call_to_action": "Share Your Thoughts",
        "template": "Promises vs Reality",
    },
    "source_articles": FILTERED_SOURCE_ARTICLES,
}


if __name__ == "__main__":
    prompt = build_prompt(DUMMY_PAYLOAD)
    print(prompt)