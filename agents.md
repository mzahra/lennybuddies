# agents.md — Lenny Pulse (AI Content Creator)

> Point any coding agent (Claude Code, Cursor, Copilot agent mode, etc.) at this file first, before it touches the repo.

## Purpose

Lenny Pulse generates authentic, brand-differentiated LinkedIn posts from Lenny's Podcast episode transcripts. It grounds every post in two knowledge bases — episode content (primary) and PM/industry trend context (secondary) — so output is traceable to real source material, not generic AI commentary. Team: Mudit, Jay, Parisa, Zahra.

## Stack & run

- Python 3.8+, virtualenv (`venv`)
- Install: `pip install -r requirements.txt`
- Main command: `python src/main.py --episode <transcript_file>`
- Env vars in `.env`: `LLM_API_KEY` (never commit `.env` or print its contents in logs/output)
- Model: gpt-4o-mini (free/cheap tier) — do not switch to a more expensive model without team sign-off, per our shared cost cap

## Repo map

```
src/
  document_processor.py   # markdown loader for both KB folders
  knowledge_base.py        # primary + secondary KB access
  prompt_templates.py      # ≥2 reusable prompt templates
  content_pipeline.py      # document → generate (→ monitor/brief/iterate if time permits)
  llm_integration.py       # LLM API client
  main.py                  # entry point
knowledge_base/
  primary/                 # episode transcripts/notes + style guide (markdown)
  secondary/                # PM/industry trend docs (markdown)
rag_decision.md            # RAG vs non-RAG choice + defense
project_structure.md       # scope, requirements, WBS, risks (source of truth for Must IDs)
```

## Conventions

- All KB content is markdown, one topic/episode per file.
- Every generated post must cite its source episode explicitly (see risk: attribution).
- Type-hint function signatures in `src/`; short docstrings on public functions.
- Prompt templates live only in `prompt_templates.py` — don't inline prompts elsewhere.
- Commits reference the Trello card ID they close (e.g. `[3.2] add prompt templates`).

## Definition of Done (agent changes)

- Matches the Kanban DoD on the active card
- Touches only the files needed for the active card's Must ID (see `project_structure.md`)
- Change is runnable via `python src/main.py` without new errors
- No secrets committed; `.env` untouched or updated only with placeholder keys
- If the change affects a Must (M1–M8), note which one in the commit/PR message

## Never do

- Commit `.env` or any API key
- Implement anything on the Won't list (full vector RAG, automated LinkedIn publishing, full monitor/brief/iterate stages) without team agreement first — see `project_structure.md`
- Invent KB facts, episode quotes, or API behavior not present in `knowledge_base/` — if source content is missing, flag it instead of fabricating
- Reproduce long verbatim transcript excerpts in generated posts — short excerpt/paraphrase + citation only
- Switch the RAG approach (currently: non-RAG, context injection) without updating `rag_decision.md`

## How we use agents

- One Kanban card at a time. Paste the card title + Must ID (e.g. "3.2 Context → prompts — M3") into the agent's prompt.
- Point the agent at this file first, then at `project_structure.md` for the relevant requirement row.
- Review agent output against the DoD above before moving the card to Review.
