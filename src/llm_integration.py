# LLM API integration
"""LLM API integration.

Two-step generation, per Mudit's Slack message on removing AI slop:
  1. generate_draft()  — writes the initial post from the built prompt
  2. polish_draft()    — runs it through an editing pass that strips AI
                          slop patterns while preserving the writer's voice

generate_post() runs both steps and returns the final, ready-to-paste
LinkedIn post text — no markdown headers, no restated metadata.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from prompt_templates import build_prompt, DUMMY_PAYLOAD

load_dotenv()

MODEL = "gpt-4o-mini"  # per agents.md — do not switch without team sign-off
_client = OpenAI(api_key=os.getenv("LLM_API_KEY"))


DRAFT_SYSTEM_PROMPT = """You are a skilled LinkedIn ghostwriter. Write the
post exactly as instructed by the user's prompt: follow the given
structural template, ground every claim in the provided source material,
and match the requested tone, style, length, and audience. Output ONLY the
post text itself — no markdown formatting, no headers, no restated
instructions, no commentary. This text will be copy-pasted directly to
LinkedIn, which does not render markdown."""


# Mudit's editing-principles spec (shared in Slack), used verbatim as the
# system prompt for the polish pass. This is the "remove AI slop" step
# from our uniqueness/differentiation strategy.
EDITING_SYSTEM_PROMPT = """Editing principles
Preserve the writer's real voice. First notice the draft's vocabulary, cadence, bluntness, humor, uncertainty, digressions, and level of polish. Keep the traits that feel personal to the writer. Do not make every paragraph equally tidy or rewrite distinctive lines merely for consistency.
Make the minimum effective edit. Fix AI patterns, errors, repetition, and unclear passages. Leave strong human sentences alone.
Lead with the point when the setup adds nothing. Cut generic throat-clearing.
Keep the user's meaning. Don't invent claims, examples, stats, or opinions.
Open it up, don't dumb it down. Keep the substance, nuance, and precision. Strip out only what makes it hard to read.
Use active voice. Never let inanimate things do human verbs.
Make every sentence earn its place. Cut empty qualifiers and throat-clearing.
Be concrete and specific. Names, numbers, dates, mechanisms, and examples beat abstractions.
Make verbs do the work. Replace weak verb phrases with direct verbs.

Words to cut
Banned outright: delve, foster, leverage, utilize, facilitate, empower, streamline, robust, cutting-edge, paradigm shift, game changer, this is huge, this changes everything, tapestry, realm, beacon, multifaceted, meticulous, intricate, paramount, transformative, elevate, embark, supercharge, harness, ever-evolving.
Often-empty phrases: it's worth noting, it's important to note, at the end of the day, when it comes to, at its core, in today's world, in the age of, the reality is, the truth is, in order to, going forward, let's dive in.

Patterns to cut
Binary contrasts ("This is not X. It's Y."). State Y directly.
Throat-clearing openers ("Here's the thing," "Let me be clear,"). Cut them.
Faux-insight setups ("What most people get wrong,"). Cut the setup, make the claim stand alone.
Colon reveals (noun phrase, colon, dramatic reveal). Rewrite as a plain sentence.
Importance puffery ("marks a pivotal moment," "underscores its significance"). State the fact plainly.
Weasel attribution ("Experts agree," "studies show"). Name the source or cut the claim.
Fake-profound kickers (a "deep" closing metaphor/aphorism). Delete it, end on the clearest concrete sentence.
Summary-recap endings ("In conclusion," "Overall,"). End on the last concrete point or next action instead.
Formatting slop: emoji in headings, bold sprinkled mid-sentence, bullet lists where prose reads better.
Em dashes: do not use as a default rhythm crutch. In short copy, use none.

Workflow
Read the full draft. Identify the core point and voice signals to preserve.
Make the minimum effective changes to strip the patterns above.
Output ONLY the fully edited post text. No "What changed" section, no
commentary, no markdown formatting — this is the final, ready-to-paste
LinkedIn post."""


def generate_draft(prompt: str) -> str:
    """Step 1: generate the initial post draft from the built prompt."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def polish_draft(draft: str) -> str:
    """Step 2: run the draft through the AI-slop editing pass."""
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EDITING_SYSTEM_PROMPT},
            {"role": "user", "content": draft},
        ],
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def generate_post(payload: Dict[str, Any]) -> str:
    """Public entry point: payload in, finished ready-to-paste post out.

    payload shape: see prompt_templates.build_prompt() docstring.
    """
    prompt = build_prompt(payload)
    draft = generate_draft(prompt)
    final_post = polish_draft(draft)
    return final_post


if __name__ == "__main__":
    post = generate_post(DUMMY_PAYLOAD)
    print("=" * 60)
    print("FINAL LINKEDIN POST")
    print("=" * 60)
    print(post)