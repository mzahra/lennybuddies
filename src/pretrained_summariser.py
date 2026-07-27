"""Pre-trained document summarisation for the primary knowledge base.

This module turns each markdown file into a compact bullet summary with the
fields requested by the workflow: title, filename, summary, and source.

Supported backends:
- Pegasus via Hugging Face Transformers
- Longformer-style long document summarisation via LED
- Ollama local models for key-point extraction

If the requested backend is unavailable, the module falls back to a
deterministic extractive summary so the pipeline still runs offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from document_processor import PRIMARY_KB_DIR
except ImportError:  # pragma: no cover - supports `python -m src.pretrained_summariser`
    from src.document_processor import PRIMARY_KB_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON_PATH = REPO_ROOT / "knowledge_base" / "pretrained_summary_index.json"
SUMMARY_CSV_PATH = REPO_ROOT / "knowledge_base" / "pretrained_summary_index.csv"

MODEL_REGISTRY = {
    "pegasus": "google/pegasus-cnn_dailymail",
    "longformer": "allenai/led-base-16384",
    "ollama": "mistral",
}

DEFAULT_MODEL_KIND = "pegasus"
MAX_INPUT_CHARS = 40000
MAX_WINDOW_CHARS = 6000
MAX_SUMMARY_BULLETS = 5

FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'“A-Z(])")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if re.fullmatch(r"-?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return stripped.strip('"').strip("'")


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text

    boundary = FRONTMATTER_BOUNDARY.search(text, 4)
    if boundary is None:
        return {}, text

    raw_frontmatter = text[4:boundary.start()]
    body = text[boundary.end():]
    metadata: Dict[str, Any] = {}
    current_key: Optional[str] = None

    for raw_line in raw_frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith(" ") and current_key and isinstance(metadata.get(current_key), list):
            item = line.strip()
            if item.startswith("- "):
                metadata[current_key].append(_parse_scalar(item[2:]))
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key

        if value == "|":
            metadata[key] = ""
            continue

        if value == "":
            metadata[key] = []
            continue

        if value.startswith("[") and value.endswith("]"):
            metadata[key] = _parse_scalar(value)
            continue

        metadata[key] = _parse_scalar(value)

    return metadata, body


def _first_non_empty_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        cleaned = _clean_text(block)
        if cleaned:
            return cleaned
    return ""


def _document_source(metadata: Dict[str, Any], relative_path: Path) -> str:
    doc_type = str(metadata.get("type", "")).lower()
    if doc_type == "podcast":
        return str(metadata.get("channel") or "Lenny's Podcast")
    if doc_type == "newsletter":
        return "Lenny's Newsletter"
    return relative_path.parts[0] if relative_path.parts else "knowledge_base"


def _extract_title(metadata: Dict[str, Any], body: str, fallback: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    match = HEADING_PATTERN.search(body)
    if match:
        return match.group(2).strip()

    paragraph = _first_non_empty_paragraph(body)
    if paragraph:
        return paragraph[:80]

    return fallback


def _extract_filename(file_path: Path, kb_dir: Path) -> str:
    try:
        return file_path.relative_to(kb_dir).as_posix()
    except ValueError:
        return file_path.as_posix()


def _split_into_windows(text: str, max_chars: int = MAX_WINDOW_CHARS) -> List[str]:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []

    windows: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        para_len = len(paragraph)
        if current and current_len + para_len + 2 > max_chars:
            windows.append("\n\n".join(current))
            current = [paragraph]
            current_len = para_len
        else:
            current.append(paragraph)
            current_len += para_len + 2

    if current:
        windows.append("\n\n".join(current))

    return windows


def _summarization_input(metadata: Dict[str, Any], body: str) -> str:
    parts: List[str] = []

    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")

    for key in ("subtitle", "summary", "description", "excerpt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.capitalize()}: {value.strip()}")
            break

    headings = [match.group(2).strip() for match in HEADING_PATTERN.finditer(body)]
    if headings:
        parts.append("Key headings: " + "; ".join(headings[:8]))

    bullets = [match.group(1).strip() for match in BULLET_PATTERN.finditer(body)]
    if bullets:
        parts.append("Key bullets: " + "; ".join(bullets[:10]))

    cleaned_body = _clean_text(body)
    if cleaned_body:
        parts.append(f"Content: {cleaned_body}")

    return "\n".join(parts)


def _sentence_bullets(text: str, max_bullets: int = MAX_SUMMARY_BULLETS) -> List[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    if cleaned.startswith("- ") or "\n-" in cleaned or cleaned.startswith("* "):
        bullets = [re.sub(r"^\s*[-*+]\s+", "", line).strip() for line in cleaned.splitlines()]
        bullets = [bullet for bullet in bullets if bullet]
        return bullets[:max_bullets]

    sentences = [part.strip() for part in SENTENCE_PATTERN.split(cleaned) if part.strip()]
    if not sentences:
        sentences = [cleaned]

    bullets: List[str] = []
    for sentence in sentences:
        bullets.append(sentence)
        if len(bullets) >= max_bullets:
            break
    return bullets


def _format_bullets(items: Sequence[str]) -> str:
    unique: List[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)

    return "\n".join(f"- {bullet}" for bullet in unique[:MAX_SUMMARY_BULLETS])


def _heuristic_summary(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    headings = [match.group(2).strip() for match in HEADING_PATTERN.finditer(text)]
    bullets = [match.group(1).strip() for match in BULLET_PATTERN.finditer(text)]
    lead = _first_non_empty_paragraph(text)

    sections: List[str] = []
    if headings:
        sections.append(f"Key theme: {headings[0]}")
    if lead:
        sections.extend(_sentence_bullets(lead, max_bullets=2))
    if bullets:
        sections.extend(bullets[:3])

    if not sections:
        sections = _sentence_bullets(cleaned, max_bullets=3)

    return _format_bullets(sections)


@lru_cache(maxsize=4)
def _load_transformer_backend(model_kind: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is not installed. Install it to enable Pegasus/Longformer summarization."
        ) from exc

    model_name = MODEL_REGISTRY[model_kind]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _load_ollama_summary(prompt: str, model_name: str) -> str:
    try:
        import urllib.error
        import urllib.request
    except ImportError:  # pragma: no cover - stdlib should always exist
        return ""

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""

    text = data.get("response", "")
    return text.strip() if isinstance(text, str) else ""


def _normalize_model_output(text: str) -> str:
    bullets = _sentence_bullets(text, max_bullets=MAX_SUMMARY_BULLETS)
    return _format_bullets(bullets)


def _build_prompt(summary_input: str) -> str:
    return textwrap.dedent(
        f"""
        You are summarizing a Lenny's Podcast or Lenny's Newsletter document.
        Extract 3 to 5 concise bullet takeaways.
        Focus on concrete facts, strong claims, named numbers, and actionable ideas.
        Avoid introductions, disclaimers, and generic filler.
        Return bullets only.

        Document:
        {summary_input}
        """
    ).strip()


def _generate_summary_with_transformer(text: str, model_kind: str) -> str:
    tokenizer, model = _load_transformer_backend(model_kind)
    windows = _split_into_windows(text)
    if not windows:
        return ""

    summaries: List[str] = []
    for window in windows[:4]:
        prompt = _build_prompt(window)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(tokenizer, "model_max_length", 1024) or 1024, 4096),
        )

        generation_kwargs = {
            "max_new_tokens": 160 if model_kind == "longformer" else 120,
            "min_new_tokens": 24,
            "num_beams": 4,
            "length_penalty": 1.0,
            "early_stopping": True,
        }

        if model_kind == "longformer":
            try:
                import torch  # type: ignore

                global_attention_mask = torch.zeros_like(inputs["input_ids"])
                global_attention_mask[:, 0] = 1
                inputs["global_attention_mask"] = global_attention_mask
            except Exception:
                pass

        generated = model.generate(**inputs, **generation_kwargs)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if decoded:
            summaries.append(decoded)

    if not summaries:
        return ""

    combined = "\n".join(summaries)
    return _normalize_model_output(combined)


def _generate_summary_with_ollama(text: str, model_name: str) -> str:
    windows = _split_into_windows(text)
    if not windows:
        return ""

    summaries: List[str] = []
    for window in windows[:4]:
        prompt = _build_prompt(window)
        response = _load_ollama_summary(prompt, model_name)
        if response:
            summaries.append(response)

    if not summaries:
        return ""

    return _normalize_model_output("\n".join(summaries))


def summarize_markdown_file(
    file_path: Path,
    model_kind: str = DEFAULT_MODEL_KIND,
    kb_dir: Path = PRIMARY_KB_DIR,
) -> Dict[str, str]:
    """Summarize one markdown file into a compact bullet list."""

    raw_text = _read_text(file_path)
    metadata, body = _parse_frontmatter(raw_text)
    title = _extract_title(metadata, body, file_path.stem)
    filename = _extract_filename(file_path, kb_dir)
    source = _document_source(metadata, file_path)

    summary_input = _summarization_input(metadata, body)
    summary = ""

    try:
        if model_kind == "ollama":
            summary = _generate_summary_with_ollama(summary_input, MODEL_REGISTRY[model_kind])
        else:
            summary = _generate_summary_with_transformer(summary_input, model_kind)
    except Exception:
        summary = ""

    if not summary:
        summary = _heuristic_summary(body)

    return {
        "title": title,
        "filename": filename,
        "summary": summary,
        "source": source,
    }


def summarize_knowledge_base(
    model_kind: str = DEFAULT_MODEL_KIND,
    kb_dir: Path = PRIMARY_KB_DIR,
) -> List[Dict[str, str]]:
    """Summarize every markdown document in the primary knowledge base."""

    if model_kind not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_kind: {model_kind!r}. Use one of: {', '.join(sorted(MODEL_REGISTRY))}"
        )

    if not kb_dir.exists():
        return []

    files = sorted(path for path in kb_dir.rglob("*") if path.suffix.lower() in {".md", ".markdown"})
    return [summarize_markdown_file(file_path, model_kind=model_kind, kb_dir=kb_dir) for file_path in files]


def build_summary_rows(
    model_kind: str = DEFAULT_MODEL_KIND,
    kb_dir: Path = PRIMARY_KB_DIR,
) -> List[Dict[str, str]]:
    """Return the summary dataset as plain dictionaries."""

    return summarize_knowledge_base(model_kind=model_kind, kb_dir=kb_dir)


def filter_summary_rows_by_topic(records: Sequence[Dict[str, str]], topic: str) -> List[Dict[str, str]]:
    """Filter summaries by topic, filename, title, source, or summary text."""

    needle = topic.strip().lower()
    if not needle:
        return list(records)

    filtered: List[Dict[str, str]] = []
    for record in records:
        values = (
            record.get("title", ""),
            record.get("filename", ""),
            record.get("summary", ""),
            record.get("source", ""),
        )
        if any(needle in value.lower() for value in values if value):
            filtered.append(record)
    return filtered


def load_summary_index(path: Path = SUMMARY_JSON_PATH) -> List[Dict[str, str]]:
    """Load a previously generated summary JSON file."""

    if not path.exists():
        return []
    loaded = json.loads(_read_text(path))
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict) and isinstance(loaded.get("summaries"), list):
        return [dict(item) for item in loaded["summaries"] if isinstance(item, dict)]
    return []


def write_summary_index(
    output_path: Path = SUMMARY_JSON_PATH,
    model_kind: str = DEFAULT_MODEL_KIND,
    kb_dir: Path = PRIMARY_KB_DIR,
) -> Path:
    """Write the summaries to disk as JSON."""

    summaries = build_summary_rows(model_kind=model_kind, kb_dir=kb_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_summary_csv(
    output_path: Path = SUMMARY_CSV_PATH,
    model_kind: str = DEFAULT_MODEL_KIND,
    kb_dir: Path = PRIMARY_KB_DIR,
) -> Path:
    """Write the summaries to disk as CSV."""

    summaries = build_summary_rows(model_kind=model_kind, kb_dir=kb_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "filename", "summary", "source"])
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize knowledge base markdown files.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=DEFAULT_MODEL_KIND)
    parser.add_argument("--limit", type=int, default=5, help="Number of summaries to print.")
    parser.add_argument("--write-json", type=Path, help="Optional JSON file to write all summaries to.")
    parser.add_argument("--write-csv", type=Path, help="Optional CSV file to write all summaries to.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    summaries = build_summary_rows(model_kind=args.model)

    if args.write_json:
        write_summary_index(args.write_json, model_kind=args.model)
    if args.write_csv:
        write_summary_csv(args.write_csv, model_kind=args.model)

    for idx, item in enumerate(summaries[: max(args.limit, 0)], start=1):
        print(f"{idx}. {item['title']} ({item['filename']})")
        print(f"   source: {item['source']}")
        print(f"   summary: {item['summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
