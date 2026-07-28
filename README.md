# Lenny Buddies

Lenny Buddies generates authentic, brand-differentiated LinkedIn posts grounded in Lenny's Podcast episode transcripts and newsletter notes. Pick a topic, a tone, and a structural template, and the pipeline finds the most relevant source material, drafts a post, and runs it through an editing pass that strips generic "AI slop" while preserving your voice.

Built as a 2-day team project (R&D). Team: Mudit, Jay, Parisa, Zahra.

## How it works

```
knowledge_base/primary/*.md          (episode transcripts + newsletters, front matter + body)
        │  document_processor.py     summarize/tag each doc via LLM
        ▼
knowledge_base/processed/processed_documents.json
        │  knowledge_base.py         TF-IDF + cosine similarity match against the user's topic
        ▼
knowledge_base/filtered/filtered_documents.json   (top 1-5 most relevant docs)
        │  prompt_templates.py       assemble prompt: user inputs + template structure + sources
        ▼
        │  llm_integration.py        generate_draft() -> polish_draft()
        ▼
Finished, ready-to-paste LinkedIn post (src/main.py, Gradio UI)
```

The knowledge base retrieval step uses plain TF-IDF (scikit-learn), not a vector store — this is a deliberate non-RAG decision documented in [rag_decision.md](rag_decision.md): the corpus is a few dozen static markdown files, so a lightweight local matcher avoids the complexity of an embeddings/retrieval system. See [project_structure.md](project_structure.md) for the full requirements/scope, [agents.md](agents.md) for repo conventions when using a coding agent, and [INTERFACES.md](INTERFACES.md) for the exact function contracts between `src/` modules.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your OpenAI key (never commit this file):

```
OPENAI_API_KEY=sk-...
```

The pipeline calls `gpt-4o-mini` — do not switch models without team sign-off (shared cost cap).

## Running the app

```bash
python src/main.py
```

This launches a Gradio UI where you enter a topic, pick word count/language/template/tone/style/goal/audience/call-to-action, and click **Generate Post**. The generated text is grounded in the actual episode/newsletter content matched to your topic and is ready to copy-paste directly to LinkedIn.

## Running the pipeline modules standalone

Each `src/` module can also be run directly for debugging one stage of the pipeline:

```bash
python src/document_processor.py   # summarize/tag knowledge_base/ markdown into processed_documents.json
python src/knowledge_base.py       # test TF-IDF matching for a sample topic
python src/llm_integration.py      # run the full draft+polish pipeline on sample inputs
```

## Tests

```bash
pytest
```

Test suite lives in `tests/`, one file per `src/` module (`test_document_processor.py`, `test_knowledge_base.py`, `test_prompt_templates.py`, `test_llm_integration.py`).

## Project layout

```
src/
  document_processor.py    markdown -> structured metadata + LLM summary/tags
  knowledge_base.py         TF-IDF document matcher and filtered-docs writer
  prompt_templates.py       prompt builder and template loader
  llm_integration.py        LLM API client + draft/polish orchestration
  main.py                   Gradio entry point
knowledge_base/
  primary/                 source markdown (podcasts/ and newsletters/), one file per episode/article
  processed/                LLM-generated summaries/keywords/tags for every source doc
  filtered/                 latest topic-matched handoff to the prompt builder
templates/
  prompt_templates.json     structural templates (with worked examples) for the post generator
tests/                      unit tests, one file per src/ module
rag_decision.md             why TF-IDF/context-injection was chosen over vector RAG
project_structure.md        scope, requirements, WBS, risks (source of truth for Must IDs)
agents.md                   conventions and guardrails for coding agents working in this repo
INTERFACES.md               module contracts: cross-file functions (name/inputs/outputs/example)
```

## Conventions

- All source content is markdown, one topic/episode per file, under `knowledge_base/primary/`.
- Every generated post must cite its source episode/article — no invented claims or fabricated quotes.
- Long verbatim transcript excerpts are never reproduced; short excerpt/paraphrase + citation only.
- See [agents.md](agents.md) for the full list of conventions and things agents should never do in this repo.
