"""Markdown document ingestion and chunking for the knowledge base."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_ROOT = REPO_ROOT / "knowledge_base"
PRIMARY_KB_DIR = KNOWLEDGE_BASE_ROOT / "primary"
INDEX_JSON_PATH = PRIMARY_KB_DIR / "index.json"

MAX_EXCERPT_LENGTH = 280

_FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_text(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip()


def _truncate(text: str, limit: int = MAX_EXCERPT_LENGTH) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        tags = [part.strip() for part in re.split(r"[,|]", value) if part.strip()]
        return tags
    if isinstance(value, (list, tuple, set)):
        tags = []
        for item in value:
            text = str(item).strip()
            if text:
                tags.append(text)
        return tags
    text = str(value).strip()
    return [text] if text else []


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


def _extract_title(metadata: Mapping[str, Any], body: str, fallback: str) -> str:
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


def _extract_excerpt(metadata: Mapping[str, Any], body: str) -> str:
    for key in ("excerpt", "summary", "description"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate(value)

    paragraph = _first_non_empty_paragraph(body)
    return _truncate(paragraph)


def _extract_document_tags(metadata: Mapping[str, Any], bucket: str) -> List[str]:
    tags = _normalize_tags(metadata.get("tags"))
    if bucket and bucket not in tags:
        tags.append(bucket)
    return tags


def _load_index_metadata(index_path: Path) -> Dict[str, Dict[str, Any]]:
    if not index_path.exists():
        return {}

    try:
        raw_index = json.loads(_read_text(index_path))
    except json.JSONDecodeError:
        return {}

    records: Iterable[Any]
    if isinstance(raw_index, list):
        records = raw_index
    elif isinstance(raw_index, dict):
        for key in ("documents", "items", "files", "entries"):
            if isinstance(raw_index.get(key), list):
                records = raw_index[key]
                break
        else:
            records = []
    else:
        records = []

    index_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue

        candidates = []
        for key in ("path", "file", "filename", "source", "relative_path"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        slug = record.get("slug")
        if isinstance(slug, str) and slug.strip():
            candidates.append(slug.strip())

        if not candidates:
            continue

        for candidate in candidates:
            index_map[_normalize_index_key(candidate)] = record

    return index_map


def _normalize_index_key(value: str) -> str:
    return str(Path(value)).replace("\\", "/").lstrip("./").lower()


def _merge_index_metadata(
    file_path: Path,
    relative_path: Path,
    index_map: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = {
        _normalize_index_key(str(relative_path)),
        _normalize_index_key(relative_path.name),
        _normalize_index_key(relative_path.stem),
        _normalize_index_key(str(file_path)),
    }
    for candidate in candidates:
        record = index_map.get(candidate)
        if record:
            return dict(record)
    return {}


def _section_chunks(
    text: str,
    base_metadata: Mapping[str, Any],
    source_label: str,
    relative_path: Path,
) -> List[Dict[str, Any]]:
    matches = list(_HEADING_PATTERN.finditer(text))
    if not matches:
        content = _clean_text(text)
        if not content:
            return []
        return [
            {
                "chunk_type": "document",
                "title": base_metadata["title"],
                "excerpt": base_metadata["excerpt"],
                "tags": list(base_metadata["tags"]),
                "source": source_label,
                "path": str(relative_path).replace("\\", "/"),
                "section": None,
                "order": 0,
                "content": content,
                "metadata": dict(base_metadata["metadata"]),
            }
        ]

    chunks: List[Dict[str, Any]] = []
    lead_text = text[: matches[0].start()].strip()
    if lead_text:
        chunks.append(
            {
                "chunk_type": "document",
                "title": base_metadata["title"],
                "excerpt": _truncate(lead_text),
                "tags": list(base_metadata["tags"]),
                "source": source_label,
                "path": str(relative_path).replace("\\", "/"),
                "section": None,
                "order": 0,
                "content": _clean_text(lead_text),
                "metadata": dict(base_metadata["metadata"]),
            }
        )

    for idx, match in enumerate(matches):
        section_title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_body = text[start:end].strip()
        if not section_body:
            continue

        combined_text = f"{section_title}\n\n{section_body}"
        chunks.append(
            {
                "chunk_type": "section",
                "title": section_title,
                "excerpt": _truncate(section_body),
                "tags": list(base_metadata["tags"]),
                "source": source_label,
                "path": str(relative_path).replace("\\", "/"),
                "section": section_title,
                "order": len(chunks),
                "content": _clean_text(combined_text),
                "metadata": dict(base_metadata["metadata"]),
            }
        )

    if not chunks:
        content = _clean_text(text)
        if content:
            chunks.append(
                {
                    "chunk_type": "document",
                    "title": base_metadata["title"],
                    "excerpt": base_metadata["excerpt"],
                    "tags": list(base_metadata["tags"]),
                    "source": source_label,
                    "path": str(relative_path).replace("\\", "/"),
                    "section": None,
                    "order": 0,
                    "content": content,
                    "metadata": dict(base_metadata["metadata"]),
                }
            )

    return chunks


def _document_base_metadata(
    file_path: Path,
    relative_path: Path,
    body: str,
    parsed_metadata: Mapping[str, Any],
    index_metadata: Mapping[str, Any],
    bucket: str,
) -> Dict[str, Any]:
    merged_metadata: Dict[str, Any] = dict(index_metadata)
    merged_metadata.update(parsed_metadata)

    title = _extract_title(merged_metadata, body, file_path.stem)
    excerpt = _extract_excerpt(merged_metadata, body)
    tags = _extract_document_tags(merged_metadata, bucket)

    return {
        "title": title,
        "excerpt": excerpt,
        "tags": tags,
        "metadata": merged_metadata,
        "path": str(relative_path).replace("\\", "/"),
        "slug": _slugify(title),
    }


def _process_markdown_file(
    file_path: Path,
    bucket: str,
    index_map: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    relative_path = file_path.relative_to(PRIMARY_KB_DIR)
    raw_text = _read_text(file_path)
    parsed_metadata, body = _parse_frontmatter(raw_text)
    index_metadata = _merge_index_metadata(file_path, relative_path, index_map)
    base_metadata = _document_base_metadata(
        file_path=file_path,
        relative_path=relative_path,
        body=body,
        parsed_metadata=parsed_metadata,
        index_metadata=index_metadata,
        bucket=bucket,
    )
    source_label = index_metadata.get("source") if isinstance(index_metadata.get("source"), str) else None
    if not source_label:
        source_label = str(relative_path).replace("\\", "/")
    return _section_chunks(body, base_metadata, source_label, relative_path)


@lru_cache(maxsize=1)
def load_knowledge_chunks() -> List[Dict[str, Any]]:
    """Load all markdown and index metadata into chunk dictionaries once."""

    chunks: List[Dict[str, Any]] = []
    index_map = _load_index_metadata(INDEX_JSON_PATH)

    if PRIMARY_KB_DIR.exists():
        for file_path in sorted(path for path in PRIMARY_KB_DIR.rglob("*") if path.suffix.lower() in {".md", ".markdown"}):
            chunks.extend(_process_markdown_file(file_path, "primary", index_map))

    return chunks


def get_knowledge_chunks() -> List[Dict[str, Any]]:
    """Return the cached knowledge chunks for reuse at runtime."""

    return load_knowledge_chunks()


KNOWLEDGE_CHUNKS = load_knowledge_chunks()


if __name__ == "__main__":
    for idx, chunk in enumerate(KNOWLEDGE_CHUNKS[:5], start=1):
        print(f"{idx}. {chunk['title']} ({chunk['path']})")
        print(f"   excerpt: {chunk['excerpt']}")
        print(f"   tags: {', '.join(chunk['tags']) if chunk['tags'] else 'none'}")
