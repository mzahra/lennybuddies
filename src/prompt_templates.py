"""Advanced prompt engineering templates.

Builds the actual LLM prompt from (a) user-selected style inputs and
(b) 1-3 relevant article summaries, already filtered and handed to us
by the knowledge_base layer (owned by Mudit/Zahra).

NOTE: as of today, that filtering step isn't live yet, so this file's
__main__ block uses DUMMY_SOURCE_ARTICLES below to develop/test against.
Swap for the real function call once knowledge_base.py exposes one —
see the TODO near the bottom.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Template library — migrated from Mudit's app.py prototype.
# Each template is a fill-in-the-blank STRUCTURE plus a worked EXAMPLE,
# used to steer the LLM's output shape without hand-writing full posts.
# ---------------------------------------------------------------------
TEMPLATE_LIBRARY: Dict[str, Dict[str, str]] = {
    "Redefining Success": {
        "structure": (
            "Only [a small fraction] of [specific initiatives/individuals] "
            "[achieve a desirable outcome].\n"
            "But you don't have to follow the [typical definition of success].\n"
            "Define what success means to you - something you'll look back on "
            "in [time period] with pride.\n"
            "Maybe it's\nOr perhaps it's\nOr even\n"
            "No matter the path, the choice is yours to make."
        ),
        "example": (
            "Only 10% of startups secure venture capital funding in their "
            "first year.\nBut you don't have to follow the typical definition "
            "of success.\nDefine what success means to you - something you'll "
            "look back on in 5 years with pride.\nMaybe it's\nOr perhaps it's\n"
            "Or even\nNo matter the path, the choice is yours to make."
        ),
    },
    "Promises vs Reality": {
        "structure": (
            "What [Technology/Product] promised: [Grand promise].\n"
            "What [Technology/Product] delivered: [Funny, mundane, or ironic reality]."
        ),
        "example": (
            "What AI promised: End of manual work.\n"
            "What AI delivered: A second layer of manual work to double-check the AI."
        ),
    },
    "Turning Point in Life": {
        "structure": (
            "You hit [specific age or milestone], and suddenly [group of people "
            "or community] starts [unexpected or stereotypical activity]."
        ),
        "example": (
            "You turn 30 and the whole squad starts playing pickleball or "
            "running half marathons."
        ),
    },
    "Finding Motivation": {
        "structure": (
            "Finding motivation can be tough, especially when [specific "
            "challenge] feels overwhelming.\nHere are a few things that keep me "
            "energized and focused: [Practical habit or routine].\nWhat's your "
            "favorite way to push through tough days?"
        ),
        "example": (
            "Finding motivation can be tough, especially when you're a creator "
            "battling a creative block.\nHere are a few things that helped me "
            "get back on track:\n1. Taking a step back to recharge and reflect "
            "- it's okay to pause."
        ),
    },
}


def format_source_context(source_articles: List[Dict[str, Any]]) -> str:
    """Turns 1-3 filtered article summaries into a labeled context block.

    Expects each dict to have the shape produced by document_summariser.py:
    title, summary, source_filename, type, tags (+ any extra fields, ignored).
    """
    blocks = []
    for i, article in enumerate(source_articles, start=1):
        blocks.append(
            f"Source {i} ({article.get('type', 'article')}, "
            f"from \"{article['title']}\"):\n"
            f"{article['summary']}\n"
            f"[Cite as: {article['source_filename']}]"
        )
    return "\n\n".join(blocks)


def build_prompt(payload: Dict[str, Any]) -> str:
    """Builds the full LLM prompt from user inputs + filtered source articles.

    payload shape (agreed contract, pending final confirmation from
    Mudit/Zahra on the knowledge_base side):
        {
            "inputs": {topic, word_count, language, tone, style, goal,
                       target_audience, call_to_action, template},
            "source_articles": [ {title, summary, source_filename, type, tags}, ... ]
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
# DUMMY test data — matches the real primary_summary_index.json schema
# we've already seen. Use this until knowledge_base.py's real filtering
# function exists.
# TODO: once Mudit/Zahra confirm the real function (e.g.
#       get_relevant_summaries(topic) -> list[dict]), replace
#       DUMMY_SOURCE_ARTICLES below with a real call to it.
# ---------------------------------------------------------------------
DUMMY_SOURCE_ARTICLES: List[Dict[str, Any]] = [
  {
    "title": "How to find hidden growth opportunities in your product | Albert Cheng (Duolingo, Grammarly, Chess.com)",
    "subtitle": "",
    "date": "2025-10-05",
    "tags": ["podcast"],
    "source_filename": "knowledge_base/primary/test/albert-cheng.md",
    "keywords": [
      "user retention",
      "growth opportunities",
      "experimentation",
      "explore-exploit framework",
      "AI in product development",
      "company culture",
      "product management"
    ],
    "summary": "In this podcast episode, Albert Cheng, a growth expert with experience at Duolingo, Grammarly, and Chess.com, discusses the importance of user retention and the exploration-exploitation framework for identifying growth opportunities. He emphasizes the need for experimentation in product development, sharing insights on how to effectively connect users to the value of a product. Cheng also highlights the role of AI in enhancing growth strategies and the significance of building a strong company culture that fosters creativity and rapid iteration.",
    "model": "gpt-4o-mini"
  },
  {
    "title": "An AI glossary",
    "subtitle": "The most common AI terms explained, simply",
    "date": "2025-06-24",
    "tags": ["newsletter"],
    "source_filename": "knowledge_base/primary/test/an-ai-glossary.md",
    "keywords": [
      "AI glossary",
      "large language models",
      "training methods",
      "prompt engineering",
      "reinforcement learning",
      "transformer architecture",
      "generative AI",
      "synthetic data"
    ],
    "summary": "This newsletter provides simplified definitions for over 20 common AI terms, making complex concepts accessible for those unfamiliar with the jargon. It covers essential topics such as AI models, large language models (LLMs), training methods, and various learning techniques, along with practical applications like prompt engineering and retrieval-augmented generation. The glossary serves as a handy reference for anyone looking to better understand the rapidly evolving field of artificial intelligence.",
    "model": "gpt-4o-mini"
  }
]

DUMMY_PAYLOAD: Dict[str, Any] = {
    "inputs": {
        "topic": "AI",
        "word_count": "150-500",
        "language": "English",
        "tone": "Professional",
        "style": "Storytelling",
        "goal": "Build Personal Brand",
        "target_audience": "Product Managers",
        "call_to_action": "Share Your Thoughts",
        "template": "Promises vs Reality",
    },
    "source_articles": DUMMY_SOURCE_ARTICLES,
}


if __name__ == "__main__":
    prompt = build_prompt(DUMMY_PAYLOAD)
    print(prompt)