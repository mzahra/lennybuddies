# INTERFACES.md — Module Contracts

This is the contract between `src/` modules: every function that is called from a file other than the one it's defined in, plus the exceptions that cross with it. Anything not listed here (e.g. `_doc_text`, `format_source_context`, `load_documents`, `generate_draft`, `polish_draft`) is a private helper used only within its own module and is free to change without breaking another file.

Pipeline call order:

```
main.py
  -> llm_integration.generate_post_from_inputs()
       -> knowledge_base.get_relevant_summaries()
       -> knowledge_base.save_filtered_documents()
       -> prompt_templates.build_prompt()
       -> (internal) generate_draft() -> polish_draft()
```

`document_processor.py` sits upstream of all of this but is **not** wired in via a Python import — see [File-based contract](#file-based-contract-document_processorpy--knowledge_basepy) below.

---

## `knowledge_base.py` → `llm_integration.py`

### `get_relevant_summaries`

- **Inputs:** `topic: str`, `documents: Optional[List[Dict[str, Any]]] = None`, `top_k: int = 5`, `min_score: float = 0.05`
- **Outputs:** `List[Dict[str, Any]]` — 1 to `top_k` document records (`title`, `subtitle`, `summary`, `tags`, `keywords`, `source_filename`, ...), ordered most-to-least relevant. Always returns at least one document if the corpus is non-empty.
- **Raises:** `KnowledgeBaseError` if `topic` is empty/not a string, `top_k` is not a positive int, or the backing `processed_documents.json` can't be loaded.
- **Example:**
  ```python
  from knowledge_base import get_relevant_summaries

  docs = get_relevant_summaries("AI evals for product managers", top_k=3)
  # -> [
  #      {"title": "Beyond Vibe Checks: A PM's Complete Guide to Evals",
  #       "summary": "...", "tags": ["AI/ML"], "keywords": [...],
  #       "source_filename": "knowledge_base/primary/newsletters/beyond-vibe-checks-a-pms-complete-guide-to-evals.md"},
  #      ...
  #    ]
  ```

### `save_filtered_documents`

- **Inputs:** `documents: List[Dict[str, Any]]`, `path: str = "knowledge_base/filtered/filtered_documents.json"`
- **Outputs:** `None`. Side effect: writes `documents` as indented JSON to `path`, creating parent directories if needed.
- **Raises:** `KnowledgeBaseError` if the directory can't be created or the file can't be written.
- **Example:**
  ```python
  from knowledge_base import get_relevant_summaries, save_filtered_documents

  docs = get_relevant_summaries("AI evals for product managers")
  save_filtered_documents(docs)
  # -> knowledge_base/filtered/filtered_documents.json now contains `docs`
  ```

### `KnowledgeBaseError` (exception)

- Raised by `get_relevant_summaries` / `save_filtered_documents`.
- Caught in `llm_integration.generate_post_from_inputs`, which re-raises it as `LLMIntegrationError` so callers outside the pipeline only ever need to catch one exception type.

---

## `prompt_templates.py` → `llm_integration.py`

### `build_prompt`

- **Inputs:** `payload: Dict[str, Any]` shaped as:
  ```python
  {
    "inputs": {"topic": str, "word_count": str, "language": str, "tone": str,
               "style": str, "goal": str, "target_audience": str,
               "call_to_action": str, "template": str},
    "source_articles": [{"title": str, "summary": str, "source_filename": str, ...}, ...],
  }
  ```
- **Outputs:** `str` — the fully assembled LLM prompt (template structure + example + cited source context + style parameters).
- **Raises:** `PromptTemplateError` if `inputs`/`source_articles` keys are missing, a required input field is missing, `inputs["template"]` isn't a known template name, or no source article has a usable `title`/`summary`/`source_filename`.
- **Example:**
  ```python
  from prompt_templates import build_prompt

  payload = {
      "inputs": {
          "topic": "shipping AI features without proper evaluation",
          "word_count": "150-500", "language": "English", "tone": "Professional",
          "style": "Storytelling", "goal": "Build Personal Brand",
          "target_audience": "Product Managers", "call_to_action": "Share Your Thoughts",
          "template": "Promises vs Reality",
      },
      "source_articles": [
          {"title": "Beyond Vibe Checks", "summary": "...",
           "source_filename": "knowledge_base/primary/newsletters/beyond-vibe-checks-a-pms-complete-guide-to-evals.md"},
      ],
  }
  prompt = build_prompt(payload)
  # -> "Write a LinkedIn post using this structural pattern:\n...(full prompt text)..."
  ```

### `PromptTemplateError` (exception)

- Raised by `build_prompt`.
- Caught in `llm_integration.generate_post` and re-raised as `LLMIntegrationError`.

> Note: `llm_integration.py` also imports `DUMMY_PAYLOAD` from `prompt_templates.py`, but nothing in `llm_integration.py` currently references it — it's an unused import left over from manual testing, not part of the live contract.

---

## `llm_integration.py` → `main.py`

### `generate_post_from_inputs`

- **Inputs:** `inputs: Dict[str, Any]` — `{topic, word_count, language, tone, style, goal, target_audience, call_to_action, template}`; `top_k: int = 5`
- **Outputs:** `str` — the finished, ready-to-paste LinkedIn post (already drafted and polished; no markdown, no headers).
- **Raises:** `LLMIntegrationError` if `inputs` isn't a dict with a non-empty `"topic"`, or if knowledge-base filtering, prompt building, drafting, or polishing fails.
- **Example** (as called in `src/main.py`, imported under the alias `run_generation_pipeline`):
  ```python
  from llm_integration import generate_post_from_inputs as run_generation_pipeline, LLMIntegrationError

  inputs = {
      "topic": "Product build", "word_count": "150-500", "language": "English",
      "tone": "Professional", "style": "Storytelling", "goal": "Build Personal Brand",
      "target_audience": "Executives", "call_to_action": "Share Your Thoughts",
      "template": "Redefining Happiness",
  }
  try:
      post = run_generation_pipeline(inputs)
      # -> "Redefining Happiness isn't about ping-pong tables...(final post text)..."
  except LLMIntegrationError as exc:
      post = f"Something went wrong generating this post: {exc}"
  ```

### `LLMIntegrationError` (exception)

- Raised by every function in `llm_integration.py`.
- Caught in `main.py`'s `generate_post_from_inputs` (Gradio callback), which renders the message directly in the UI instead of crashing the app.

---

## File-based contract: `document_processor.py` → `knowledge_base.py`

`document_processor.py` exports no function that another `src/` module imports — its handoff to the rest of the pipeline is a JSON file on disk, not a Python call.

- **Producer:** `document_processor.save_processed_documents(documents, path="knowledge_base/processed/processed_documents_test.json")` writes a JSON list of records shaped `{title, subtitle, date, tags, source_filename, keywords, summary, model}`.
- **Consumer:** `knowledge_base.load_documents(path=DEFAULT_DOCS_PATH)`, where `DEFAULT_DOCS_PATH = "knowledge_base/processed/processed_documents.json"`, reads that same shape.
- **⚠ Path mismatch:** the two default paths differ (`processed_documents_test.json` vs `processed_documents.json`). Running `python src/document_processor.py` as-is will **not** populate the file `knowledge_base.py` reads by default — pass matching `path` arguments explicitly, or align the defaults, until this is reconciled.
- **Example record** (one entry in the JSON list, either side):
  ```json
  {
    "title": "Beyond Vibe Checks: A PM's Complete Guide to Evals",
    "subtitle": "...",
    "date": "2025-06-01",
    "tags": ["AI/ML"],
    "source_filename": "knowledge_base/primary/newsletters/beyond-vibe-checks-a-pms-complete-guide-to-evals.md",
    "keywords": ["evals", "LLM testing", "product quality"],
    "summary": "...",
    "model": "gpt-4o-mini"
  }
  ```
