"""Summarize knowledge base markdown files for downstream post generation.

This module builds one short note per markdown file under knowledge_base/primary.
It prefers a seq2seq model from Hugging Face (BART or T5) when available, and
falls back to a light extractive summary when the ML stack is not installed.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from document_processor import PRIMARY_KB_DIR
except ImportError:  # pragma: no cover - supports `python -m src.document_summariser`
    from src.document_processor import PRIMARY_KB_DIR


MODEL_REGISTRY = {
    "bart": "sshleifer/distilbart-cnn-12-6",
    "t5": "google/flan-t5-small",
}

DEFAULT_MODEL_KIND = "bart"
MAX_INPUT_CHARS = 40000
MAX_OUTPUT_TOKENS = 2000
MAX_NOTE_CHARS = 9000

_FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$", re.MULTILINE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

ALLOWED_KEYWORDS = [
    "data",
    "project management",
    "QA",
    "evals",
    "artificial intelligence",
    "strategy",
    "product",
    "b2b",
    "b2c",
    "web3",
    "finance",
]

KEYWORD_ALIASES = {
    "data": [
        r"\bdata\b",
        r"\banalytics\b",
        r"\bbi\b",
        r"\binstrumentation\b",
        r"\bmetrics\b",
        r"\bmeasurement\b",
    ],
    "project management": [
        r"\bproject management\b",
        r"\bproject manager\b",
        r"\bpm\b",
        r"\broadmap\b",
        r"\bplanning\b",
        r"\bexecution\b",
    ],
    "QA": [
        r"\bqa\b",
        r"\bquality assurance\b",
        r"\btesting\b",
        r"\btest automation\b",
        r"\bmanual testing\b",
    ],
    "evals": [
        r"\bevals?\b",
        r"\bevaluation\b",
        r"\bbenchmark\b",
        r"\bassessment\b",
    ],
    "artificial intelligence": [
        r"\bartificial intelligence\b",
        r"\bai\b",
        r"\bmachine learning\b",
        r"\bml\b",
        r"\bllm\b",
        r"\bgenai\b",
        r"\bgenerative ai\b",
    ],
    "strategy": [
        r"\bstrategy\b",
        r"\bstrategic\b",
        r"\bgo-to-market\b",
        r"\bgtm\b",
        r"\bpositioning\b",
        r"\bplanning\b",
    ],
    "product": [
        r"\bproduct\b",
        r"\bproduct management\b",
        r"\bproduct manager\b",
        r"\bpm\b",
        r"\broadmap\b",
        r"\bproduct discovery\b",
    ],
    "b2b": [
        r"\bb2b\b",
        r"\bbusiness-to-business\b",
        r"\benterprise\b",
        r"\bsmb\b",
    ],
    "b2c": [
        r"\bb2c\b",
        r"\bbusiness-to-consumer\b",
        r"\bconsumer\b",
        r"\bend user\b",
    ],
    "web3": [
        r"\bweb3\b",
        r"\bblockchain\b",
        r"\bcrypto\b",
        r"\bdefi\b",
        r"\bnft\b",
    ],
    "finance": [
        r"\bfinance\b",
        r"\bfintech\b",
        r"\bbanking\b",
        r"\brevenue\b",
        r"\bpayments\b",
        r"\bmonetization\b",
    ],
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _truncate(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text

    boundary = _FRONTMATTER_BOUNDARY.search(text, 4)
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
                metadata[current_key].append(item[2:].strip())
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_key = key

        if value == "":
            metadata[key] = []
            continue

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            metadata[key] = [part.strip() for part in inner.split(",") if part.strip()] if inner else []
            continue

        metadata[key] = value.strip('"').strip("'")

    return metadata, body


def _first_non_empty_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        cleaned = _clean_text(block)
        if cleaned:
            return cleaned
    return ""


def _title_from_markdown(metadata: Dict[str, Any], body: str, fallback: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    match = _HEADING_PATTERN.search(body)
    if match:
        return match.group(2).strip()

    paragraph = _first_non_empty_paragraph(body)
    if paragraph:
        return paragraph[:80]

    return fallback


def _extract_tags(metadata: Dict[str, Any], file_path: Path) -> List[str]:
    tags: List[str] = []
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, str):
        tags.extend([part.strip() for part in re.split(r"[,|]", raw_tags) if part.strip()])
    elif isinstance(raw_tags, list):
        tags.extend([str(item).strip() for item in raw_tags if str(item).strip()])

    folder = file_path.parent.name
    if folder and folder not in tags:
        tags.append(folder)
    return tags


def _extract_subtitle(metadata: Dict[str, Any], body: str) -> str:
    for key in ("subtitle", "deck", "description", "summary", "excerpt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value, 220)

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    if len(paragraphs) > 1:
        return _truncate(paragraphs[1], 220)
    if paragraphs:
        return _truncate(paragraphs[0], 220)
    return ""


def _extract_date(metadata: Dict[str, Any]) -> str:
    for key in ("date", "published", "publish_date", "published_at", "publishedAt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_type(metadata: Dict[str, Any], file_path: Path) -> str:
    for key in ("type", "kind", "content_type", "document_type"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return file_path.parent.name or "markdown"


def _normalize_keyword_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,|;]", value) if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _extract_keywords(metadata: Dict[str, Any], body: str, title: str) -> List[str]:
    searchable_parts: List[str] = []
    for key in ("keywords", "keyword", "tags"):
        searchable_parts.extend(_normalize_keyword_value(metadata.get(key)))

    searchable_parts.append(title)
    searchable_parts.extend([match.group(2).strip() for match in _HEADING_PATTERN.finditer(body)][:8])
    searchable_parts.extend([match.group(1).strip() for match in _BULLET_PATTERN.finditer(body)][:8])
    searchable_parts.append(_clean_text(body))

    haystack = " ".join(part for part in searchable_parts if part).lower()
    keywords: List[str] = []
    for canonical in ALLOWED_KEYWORDS:
        patterns = KEYWORD_ALIASES[canonical]
        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in patterns):
            keywords.append(canonical)

    return keywords


def _summarization_input(metadata: Dict[str, Any], body: str) -> str:
    parts: List[str] = []
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"Title: {title.strip()}")

    for key in ("summary", "excerpt", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key.capitalize()}: {value.strip()}")
            break

    headings = [match.group(2).strip() for match in _HEADING_PATTERN.finditer(body)]
    if headings:
        parts.append("Key headings: " + "; ".join(headings[:6]))

    bullets = [match.group(1).strip() for match in _BULLET_PATTERN.finditer(body)]
    if bullets:
        parts.append("Key bullets: " + "; ".join(bullets[:6]))

    cleaned_body = _clean_text(body)
    parts.append(f"Content: {cleaned_body}")
    return "\n".join(parts)


def _split_into_windows(text: str, max_chars: int = MAX_INPUT_CHARS) -> List[str]:
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []

    windows: List[str] = []
    current: List[str] = []
    current_len = 0

    for paragraph in paragraphs:
        para = paragraph.strip()
        para_len = len(para)
        if current and current_len + para_len + 2 > max_chars:
            windows.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        windows.append("\n\n".join(current))

    return windows


def _heuristic_summary(text: str, max_chars: int = MAX_NOTE_CHARS) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    lead = _first_non_empty_paragraph(text)
    heading_hits = [match.group(2).strip() for match in _HEADING_PATTERN.finditer(text)]
    bullet_hits = [match.group(1).strip() for match in _BULLET_PATTERN.finditer(text)]

    parts: List[str] = []
    if heading_hits:
        parts.append(heading_hits[0])
    if lead:
        parts.append(lead)
    if bullet_hits:
        parts.append("Notable points: " + "; ".join(bullet_hits[:3]))
    if len(paragraphs) > 1:
        parts.append(f"Structured around {len(heading_hits) or len(paragraphs)} sections.")

    summary = " ".join(parts) if parts else cleaned[:max_chars]
    return _truncate(summary, max_chars)


@lru_cache(maxsize=4)
def _load_transformer_backend(model_kind: str):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it to enable BART/T5 summarization."
        ) from exc

    model_name = MODEL_REGISTRY[model_kind]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _generate_summary_with_model(text: str, model_kind: str) -> str:
    tokenizer, model = _load_transformer_backend(model_kind)

    windows = _split_into_windows(text)
    if not windows:
        return ""

    summaries: List[str] = []
    for window in windows[:4]:
        if model_kind == "t5":
            prompt = f"summarize: {window}"
        else:
            prompt = window

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT_TOKENS,
            min_new_tokens=24,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if decoded:
            summaries.append(decoded)

    if not summaries:
        return ""

    if len(summaries) == 1:
        return _truncate(summaries[0], MAX_NOTE_CHARS)

    combined = " ".join(summaries)
    if len(combined) <= MAX_NOTE_CHARS:
        return combined

    # Reduce the combined notes to a single compact note.
    if model_kind == "t5":
        prompt = f"summarize: {combined}"
    else:
        prompt = combined
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    generated = model.generate(
        **inputs,
        max_new_tokens=96,
        min_new_tokens=24,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    return _truncate(decoded or combined, MAX_NOTE_CHARS)


def summarize_markdown_file(file_path: Path, model_kind: str = DEFAULT_MODEL_KIND) -> Dict[str, Any]:
    """Summarize one markdown file into a short note."""

    raw_text = _read_text(file_path)
    metadata, body = _parse_frontmatter(raw_text)
    title = _title_from_markdown(metadata, body, file_path.stem)
    subtitle = _extract_subtitle(metadata, body)
    date = _extract_date(metadata)
    doc_type = _extract_type(metadata, file_path)
    tags = _extract_tags(metadata, file_path)
    keywords = _extract_keywords(metadata, body, title)
    try:
        source_path = file_path.relative_to(PRIMARY_KB_DIR).as_posix()
    except ValueError:
        source_path = file_path.as_posix()

    summarization_input = _summarization_input(metadata, body)
    summary = ""

    try:
        summary = _generate_summary_with_model(summarization_input, model_kind)
    except Exception:
        summary = ""

    if not summary:
        summary = _heuristic_summary(body)

    return {
        "title": title,
        "subtitle": subtitle,
        "date": date,
        "type": doc_type,
        "keywords": keywords,
        "tags": tags,
        "source_filename": source_path,
        "summary": summary,
        "model": MODEL_REGISTRY.get(model_kind, model_kind),
    }


def summarize_knowledge_base(model_kind: str = DEFAULT_MODEL_KIND) -> List[Dict[str, Any]]:
    """Summarize all markdown files in knowledge_base/primary."""

    if model_kind not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_kind: {model_kind!r}. Use one of: {', '.join(MODEL_REGISTRY)}")

    if not PRIMARY_KB_DIR.exists():
        return []

    files = sorted(path for path in PRIMARY_KB_DIR.rglob("*") if path.suffix.lower() in {".md", ".markdown"})
    return [summarize_markdown_file(file_path, model_kind=model_kind) for file_path in files]


def write_summary_index(output_path: Path, model_kind: str = DEFAULT_MODEL_KIND) -> Path:
    """Write the summaries to disk as JSON."""

    summaries = summarize_knowledge_base(model_kind=model_kind)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize knowledge base markdown files.")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), default=DEFAULT_MODEL_KIND)
    parser.add_argument("--limit", type=int, default=5, help="Number of summaries to print.")
    parser.add_argument("--write", type=Path, help="Optional JSON file to write all summaries to.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    summaries = summarize_knowledge_base(model_kind=args.model)

    if args.write:
        write_summary_index(args.write, model_kind=args.model)

    for idx, item in enumerate(summaries[: max(args.limit, 0)], start=1):
        print(f"{idx}. {item['title']} ({item['source_filename']})")
        print(f"   subtitle: {item['subtitle'] or 'none'}")
        print(f"   date: {item['date'] or 'none'}")
        print(f"   type: {item['type']}")
        print(f"   summary: {item['summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
