"""Advanced prompt engineering templates.

Builds the actual LLM prompt from (a) user-selected style inputs and
(b) 1-3 relevant article summaries, loaded from Mudit/Zahra's real
handoff file: knowledge_base/filtered/filtered_documents.json.

Only title, summary, and source_filename are used as content — keywords
and tags are intentionally not fed to the LLM (team decision).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PromptTemplateError(Exception):
    """Raised when the template library or payload is invalid."""


# ---------------------------------------------------------------------
# Template library — loaded from /templates/prompt_templates.json, the
# same file app.py reads. Each template is a fill-in-the-blank STRUCTURE
# plus a worked EXAMPLE, used to steer the LLM's output shape without
# hand-writing full posts.
# ---------------------------------------------------------------------
_TEMPLATE_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "prompt_templates.json"
)


def load_template_library(path: Path = _TEMPLATE_LIBRARY_PATH) -> Dict[str, Dict[str, str]]:
    """Loads the template library JSON.

    Raises:
        PromptTemplateError: if the file is missing or isn't valid JSON.
    """
    if not path.is_file():
        raise PromptTemplateError(f"Template library file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise PromptTemplateError(f"Malformed JSON in {path}: {exc}") from exc


# Loaded lazily/defensively at import time: a missing or broken template
# file shouldn't crash every module that imports this one (e.g. tests,
# llm_integration.py). build_prompt() re-checks and raises a clear error
# if the library is actually needed but unavailable.
try:
    TEMPLATE_LIBRARY: Dict[str, Dict[str, str]] = load_template_library()
except PromptTemplateError as exc:
    logger.warning("Could not load template library at import time: %s", exc)
    TEMPLATE_LIBRARY = {}


def format_source_context(source_articles: List[Dict[str, Any]]) -> str:
    """Turns 1-3 filtered article summaries into a labeled context block.

    Only title, summary, and source_filename are treated as real content
    fed to the LLM — keywords/tags are deliberately NOT included (team
    decision: title + summary are enough; tags now hold topic categories,
    not a reliable newsletter/podcast type label, so we don't try to
    derive a type from them anymore).

    Articles missing title/summary/source_filename are skipped with a
    warning rather than raising, so one malformed record doesn't block
    the whole post generation.
    """
    blocks = []
    for i, article in enumerate(source_articles, start=1):
        title = article.get("title")
        summary = article.get("summary")
        source_filename = article.get("source_filename")

        if not title or not summary or not source_filename:
            logger.warning(
                "Skipping source article %d: missing title/summary/source_filename (%r)",
                i, article,
            )
            continue

        blocks.append(
            f"Source {i} (from \"{title}\"):\n"
            f"{summary}\n"
            f"[Cite as: {source_filename}]"
        )
    return "\n\n".join(blocks)


REQUIRED_INPUT_FIELDS = (
    "topic", "word_count", "language", "tone", "style",
    "goal", "target_audience", "call_to_action", "template",
)


def build_prompt(payload: Dict[str, Any]) -> str:
    """Builds the full LLM prompt from user inputs + filtered source articles.

    payload shape (matches knowledge_base/filtered/filtered_documents.json):
        {
            "inputs": {topic, word_count, language, tone, style, goal,
                       target_audience, call_to_action, template},
            "source_articles": [ {title, summary, source_filename, ...}, ... ]
        }

    Raises:
        PromptTemplateError: if the payload is missing required keys, the
        template name is unknown, or no usable source articles remain.
    """
    if "inputs" not in payload:
        raise PromptTemplateError("payload is missing required key 'inputs'")
    if "source_articles" not in payload:
        raise PromptTemplateError("payload is missing required key 'source_articles'")

    inputs = payload["inputs"]

    missing_fields = [field for field in REQUIRED_INPUT_FIELDS if field not in inputs]
    if missing_fields:
        raise PromptTemplateError(f"inputs is missing required field(s): {missing_fields}")

    if not TEMPLATE_LIBRARY:
        raise PromptTemplateError(
            "Template library is empty or failed to load — check "
            f"{_TEMPLATE_LIBRARY_PATH} exists and is valid JSON."
        )

    template_name = inputs["template"]
    if template_name not in TEMPLATE_LIBRARY:
        raise PromptTemplateError(
            f"Unknown template '{template_name}'. "
            f"Valid options: {list(TEMPLATE_LIBRARY.keys())}"
        )

    template = TEMPLATE_LIBRARY[template_name]
    context = format_source_context(payload["source_articles"])

    if not context:
        raise PromptTemplateError(
            "No usable source articles to ground the post in — "
            "source_articles was empty or every entry was malformed."
        )

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


def load_filtered_articles(count: int = 3, path: Path = _FILTERED_DOCS_PATH) -> List[Dict[str, Any]]:
    """Loads the pre-filtered articles Mudit/Zahra hand off for this
    request. As of today this file is a static 2-entry sample; once
    their real per-topic filtering is live, this same function keeps
    working unchanged — only the file's contents change, not this code.

    Raises:
        PromptTemplateError: if the file is missing or isn't valid JSON.
    """
    if not path.is_file():
        raise PromptTemplateError(f"Filtered documents file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
    except json.JSONDecodeError as exc:
        raise PromptTemplateError(f"Malformed JSON in {path}: {exc}") from exc

    if not isinstance(all_entries, list):
        raise PromptTemplateError(
            f"Expected a JSON list in {path}, got {type(all_entries).__name__}"
        )

    # Light defensive check only — real filtering already happened
    # upstream, so we're not re-filtering here, just guarding against
    # a genuinely broken/empty record.
    usable = [entry for entry in all_entries if entry.get("summary")]
    return usable[:count]


# Loaded defensively so importing this module (e.g. from tests or
# llm_integration.py) doesn't crash if filtered_documents.json doesn't
# exist yet in a fresh checkout.
try:
    FILTERED_SOURCE_ARTICLES: List[Dict[str, Any]] = load_filtered_articles(3)
except PromptTemplateError as exc:
    logger.warning("Could not load filtered articles at import time: %s", exc)
    FILTERED_SOURCE_ARTICLES = []

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
    try:
        prompt = build_prompt(DUMMY_PAYLOAD)
        print(prompt)
    except PromptTemplateError as exc:
        print(f"ERROR: {exc}")
