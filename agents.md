# agents.md — Lenny Pulse (AI Content Creator)

> Point any coding agent (Claude Code, Cursor, Copilot agent mode, etc.) at this file first, before it touches the repo.

## Purpose

Lenny Pulse generates authentic, brand-differentiated LinkedIn posts from Lenny's Podcast episode transcripts and related newsletter notes. It grounds each post in locally stored markdown plus a filtered document handoff so output stays traceable to real source material, not generic AI commentary. Team: Mudit, Jay, Parisa, Zahra.

## Stack & run

- Python 3.8+
- Install: `pip install -r requirements.txt`
- Main command: `python src/main.py`
- Env vars in `.env`: `OPENAI_API_KEY` (never commit `.env` or print its contents in logs/output)
- Model: `gpt-4o-mini` (free/cheap tier) - do not switch to a more expensive model without team sign-off, per our shared cost cap

## Repo map

```
src/
  document_processor.py    # markdown -> structured metadata + summary
  knowledge_base.py        # TF-IDF document matcher and filtered-docs writer
  prompt_templates.py      # prompt builder and template loader
  llm_integration.py      # LLM API client + draft/polish orchestration
  main.py                  # Gradio entry point
knowledge_base/
  primary/                # source markdown by contributor/source set
  filtered/filtered_documents.json # latest filtered handoff to the prompt builder
templates/
  prompt_templates.json    # template library for the post generator
tests/                     # is the unit test suite. Each file checks one part of the codebas
  test_document_processor.py 
  test_knowledge_base.py
  test_prompt_templates.py
  test_llm_integration.py
  conftest.py
rag_decision.md            # Decision:use direct context injection with local markdown sources instead of a retrieval system.
project_structure.md       # scope, requirements, WBS, risks (source of truth for Must IDs)
INTERFACES.md              # current pipeline/interface (short design note)
pytest.ini                 # pytest config
requirements.txt           # runtime/test dependencies
```

## Conventions

- All source content is markdown, one topic/episode per file.
- The pipeline currently filters into `knowledge_base/filtered/filtered_documents.json` before prompt assembly.
- Every generated post must cite its source episode or article explicitly (see risk: attribution).
- Type-hint function signatures in `src/`; short docstrings on public functions.
- Prompt templates live in `templates/prompt_templates.json` and are loaded by `src/prompt_templates.py`; don't duplicate template structure elsewhere.
- Tests live under `tests/` and should be updated when changing `src/` behavior.
- Commits reference the Trello card ID they close (e.g. `[3.2] add prompt templates`).

## Definition of Done (agent changes)

- Matches the Kanban DoD on the active card
- Touches only the files needed for the active card's Must ID (see `project_structure.md`)
- Change is runnable via `python src/main.py` without new errors
- No secrets committed; `.env` untouched or updated only with placeholder keys
- If the change affects a Must (M1-M8), note which one in the commit/PR message

## Never do

- Commit `.env` or any API key
- Implement anything on the Won't list (full vector RAG, automated LinkedIn publishing, full monitor/brief/iterate stages) without team agreement first - see `project_structure.md`
- Invent KB facts, episode quotes, or API behavior not present in `knowledge_base/` - if source content is missing, flag it instead of fabricating
- Reproduce long verbatim transcript excerpts in generated posts - short excerpt/paraphrase + citation only
- Switch the RAG approach without updating `rag_decision.md`

## How we use agents

- One Kanban card at a time. Paste the card title + Must ID (e.g. "3.2 Context -> prompts - M3") into the agent's prompt.
- Point the agent at this file first, then at `project_structure.md` for the relevant requirement row.
- Review agent output against the DoD above before moving the card to Review.
