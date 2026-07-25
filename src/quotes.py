"""Extract high-signal quotes, facts, and key points from the primary KB.

The script scans the markdown corpus under `knowledge_base/primary`, scores
candidate snippets, and writes a compact retrieval table to CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from document_processor import PRIMARY_KB_DIR
except ImportError:  # pragma: no cover - supports `python -m src.quotes`
    from src.document_processor import PRIMARY_KB_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
QUOTES_JSON_PATH = REPO_ROOT / "knowledge_base" / "quotes.json"
QUOTES_CSV_PATH = REPO_ROOT / "knowledge_base" / "quotes.csv"

FRONTMATTER_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TRANSCRIPT_TURN_PATTERN = re.compile(r"\*\*(?P<speaker>[^*]+)\*\*\s*\([^)]*\):\s*")
BULLET_PATTERN = re.compile(r"^\s*[-*+]\s+(.+?)\s*$", re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r"\s+")

BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsubscribe\b",
        r"\bfollow\b",
        r"\bthanks for listening\b",
        r"\bdon't forget\b",
        r"\bwelcome to\b",
        r"\bif you enjoy\b",
        r"\bsee you next time\b",
        r"\bhead on over\b",
        r"\bpodcast\b",
    )
]

POWER_PHRASES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bkey\b",
        r"\bimportant\b",
        r"\bshould\b",
        r"\bmust\b",
        r"\bneed to\b",
        r"\blearned\b",
        r"\brealized\b",
        r"\bthe truth is\b",
        r"\bwhat matters\b",
        r"\bin practice\b",
        r"\bwe found\b",
        r"\bI think\b",
        r"\bI believe\b",
    )
]

LENNY_ALIASES = {"lenny rachitsky", "lenny"}


@dataclass(frozen=True)
class QuoteRecord:
    """A single retrieval row for downstream context injection."""

    topic: str
    quote: str
    source: str
    title: str
    filename: str
    quote_by: str
    _score: float
    _order: int

    def public_dict(self) -> Dict[str, str]:
        return {
            "topic": self.topic,
            "quote": self.quote,
            "source": self.source,
            "title": self.title,
            "filename": self.filename,
            "quote_by": self.quote_by,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_whitespace(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_]{1,3}(.+?)[*_]{1,3}", r"\1", text)
    text = _clean_whitespace(text)
    return text.strip()


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


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
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


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'“A-Z(])", text)
    return [part.strip() for part in parts if part.strip()]


def _first_non_empty_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text):
        cleaned = _clean_whitespace(block)
        if cleaned:
            return cleaned
    return ""


def _normalize_topic(title: str) -> str:
    topic = title.strip()
    if "|" in topic:
        topic = topic.split("|", 1)[0].strip()
    topic = re.sub(r"\s+[—-]\s+with\s+.+$", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\s+\((?:2x|x)?[^)]+\)$", "", topic).strip()
    return topic


def _source_label(metadata: Dict[str, Any], relative_path: Path) -> str:
    doc_type = str(metadata.get("type", "")).lower()
    if doc_type == "podcast":
        return str(metadata.get("channel") or "Lenny's Podcast")
    if doc_type == "newsletter":
        return "Lenny's Newsletter"
    return relative_path.parts[0] if relative_path.parts else "knowledge_base"


def _speaker_name(raw_name: str) -> str:
    name = raw_name.strip()
    return re.sub(r"\s+", " ", name)


def _extract_transcript_turns(body: str) -> List[tuple[str, str]]:
    matches = list(TRANSCRIPT_TURN_PATTERN.finditer(body))
    if not matches:
        return []

    turns: List[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        speaker = _speaker_name(match.group("speaker"))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            turns.append((speaker, text))
    return turns


def _extract_bullet_items(text: str) -> List[str]:
    return [match.group(1).strip() for match in BULLET_PATTERN.finditer(text) if match.group(1).strip()]


def _clean_candidate(text: str) -> str:
    text = _strip_markdown(text)
    text = re.sub(r"\[inaudible[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _excerpt_sentences(text: str, max_words: int = 45, max_sentences: int = 2) -> str:
    cleaned = _clean_candidate(text)
    if not cleaned:
        return ""

    sentences = _split_sentences(cleaned)
    if not sentences:
        words = cleaned.split()
        if len(words) <= max_words:
            return cleaned
        return " ".join(words[:max_words]).rstrip(" ,;:") + "…"

    excerpt: List[str] = []
    word_count = 0
    for sentence in sentences[:max_sentences]:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        if excerpt and word_count + len(sentence_words) > max_words:
            break
        excerpt.append(sentence)
        word_count += len(sentence_words)
        if word_count >= max_words:
            break

    if not excerpt:
        first_sentence = sentences[0]
        words = first_sentence.split()
        return first_sentence if len(words) <= max_words else " ".join(words[:max_words]).rstrip(" ,;:") + "…"

    joined = " ".join(excerpt)
    if _word_count(joined) > max_words:
        words = joined.split()
        joined = " ".join(words[:max_words]).rstrip(" ,;:") + "…"
    return joined


def _is_boilerplate(text: str) -> bool:
    return any(pattern.search(text) for pattern in BOILERPLATE_PATTERNS)


def _word_count(text: str) -> int:
    return len(text.split())


def _candidate_score(text: str, speaker: str, doc_type: str) -> float:
    score = 0.0
    words = _word_count(text)

    if words < 6:
        return -10.0
    if words > 60:
        score -= 2.0

    if 10 <= words <= 35:
        score += 2.0
    elif 6 <= words < 10 or 36 <= words <= 45:
        score += 1.0

    if re.search(r'["“”]', text):
        score += 3.0
    if re.search(r"\b\d[\d,.%x/-]*\b", text):
        score += 2.5
    if any(pattern.search(text) for pattern in POWER_PHRASES):
        score += 2.0
    if speaker and speaker.lower() not in LENNY_ALIASES:
        score += 1.0
    if doc_type == "podcast" and speaker:
        score += 0.5

    if _is_boilerplate(text):
        score -= 8.0

    if re.search(r"\b(?:better|worse|faster|harder|easier|bigger|smaller)\b", text, re.IGNORECASE):
        score += 0.75

    return score


def _dedupe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _candidate_records_for_document(
    metadata: Dict[str, Any],
    body: str,
    relative_path: Path,
    order_seed: int,
    max_quotes_per_file: int,
) -> List[QuoteRecord]:
    doc_type = str(metadata.get("type", "")).lower()
    title = str(metadata.get("title") or relative_path.stem)
    topic = _normalize_topic(title)
    source = _source_label(metadata, relative_path)
    filename = str(relative_path).replace("\\", "/")
    default_quote_by = str(metadata.get("guest") or "Lenny Rachitsky")

    candidates: List[tuple[float, int, str, str]] = []
    seen_candidates: set[str] = set()
    local_order = 0

    if doc_type == "podcast":
        turns = _extract_transcript_turns(body)
        for speaker, turn_text in turns:
            cleaned_turn = _excerpt_sentences(turn_text, max_words=48, max_sentences=2)
            if not cleaned_turn:
                continue
            key = _dedupe_key(cleaned_turn)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            score = _candidate_score(cleaned_turn, speaker, doc_type)
            candidates.append((score, local_order, cleaned_turn, speaker))
            local_order += 1
    else:
        blocks = re.split(r"\n\s*\n", body)
        for block in blocks:
            cleaned_block = block.strip()
            if not cleaned_block:
                continue

            heading_match = HEADING_PATTERN.match(cleaned_block)
            if heading_match:
                heading = _excerpt_sentences(heading_match.group(2), max_words=16, max_sentences=1)
                if heading:
                    key = _dedupe_key(heading)
                    if key not in seen_candidates:
                        seen_candidates.add(key)
                        candidates.append((_candidate_score(heading, "", doc_type), local_order, heading, default_quote_by))
                        local_order += 1

            bullet_items = _extract_bullet_items(cleaned_block)
            if bullet_items:
                for item in bullet_items:
                    cleaned = _excerpt_sentences(item, max_words=30, max_sentences=2)
                    if not cleaned:
                        continue
                    key = _dedupe_key(cleaned)
                    if key in seen_candidates:
                        continue
                    seen_candidates.add(key)
                    candidates.append((_candidate_score(cleaned, "", doc_type), local_order, cleaned, default_quote_by))
                    local_order += 1
                continue

            cleaned_block = _excerpt_sentences(cleaned_block, max_words=40, max_sentences=2)
            if not cleaned_block:
                continue
            key = _dedupe_key(cleaned_block)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidates.append((_candidate_score(cleaned_block, "", doc_type), local_order, cleaned_block, default_quote_by))
            local_order += 1

    candidates.sort(key=lambda item: (-item[0], item[1], len(item[2])))

    records: List[QuoteRecord] = []
    for rank, (score, _, quote, speaker) in enumerate(candidates[: max_quotes_per_file * 4]):
        if score < -1.0:
            continue
        record = QuoteRecord(
            topic=topic,
            quote=quote,
            source=source,
            title=title,
            filename=filename,
            quote_by=speaker or default_quote_by,
            _score=score,
            _order=order_seed + rank,
        )
        records.append(record)
        if len(records) >= max_quotes_per_file:
            break

    return records


def build_quote_records(
    kb_dir: Path = PRIMARY_KB_DIR,
    max_quotes_per_file: int = 4,
) -> List[QuoteRecord]:
    """Build scored quote records for every markdown document in the KB."""

    records: List[QuoteRecord] = []
    markdown_files = sorted(
        path for path in kb_dir.rglob("*") if path.suffix.lower() in {".md", ".markdown"}
    )

    for file_index, file_path in enumerate(markdown_files):
        relative_path = file_path.relative_to(kb_dir)
        raw_text = _read_text(file_path)
        metadata, body = _parse_frontmatter(raw_text)
        records.extend(
            _candidate_records_for_document(
                metadata=metadata,
                body=body,
                relative_path=relative_path,
                order_seed=file_index * 1000,
                max_quotes_per_file=max_quotes_per_file,
            )
        )

    records.sort(key=lambda record: (-record._score, record._order, record.filename))
    return records


def build_quote_rows(
    kb_dir: Path = PRIMARY_KB_DIR,
    max_quotes_per_file: int = 4,
) -> List[Dict[str, str]]:
    """Return the public quote dataset as plain dictionaries."""

    return [record.public_dict() for record in build_quote_records(kb_dir, max_quotes_per_file)]


def _match_topic(record: Dict[str, str], topic: str) -> bool:
    needle = topic.strip().lower()
    if not needle:
        return True

    values = (record.get("topic", ""), record.get("title", ""), record.get("quote", ""), record.get("quote_by", ""))
    return any(needle in value.lower() for value in values if value)


def filter_quotes_by_topic(records: Sequence[Dict[str, str]], topic: str) -> List[Dict[str, str]]:
    """Filter the quote rows by topic, title, quote, or speaker name."""

    return [record for record in records if _match_topic(record, topic)]


def load_quotes_json(path: Path = QUOTES_JSON_PATH) -> List[Dict[str, str]]:
    """Load a previously generated quotes JSON file."""

    if not path.exists():
        return []
    loaded = json.loads(_read_text(path))
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict) and isinstance(loaded.get("quotes"), list):
        return [dict(item) for item in loaded["quotes"] if isinstance(item, dict)]
    return []


def write_quotes_dataset(
    records: Sequence[Dict[str, str]],
    json_path: Path = QUOTES_JSON_PATH,
    csv_path: Path = QUOTES_CSV_PATH,
    output_format: str = "both",
) -> List[Path]:
    """Write the quote dataset to JSON and/or CSV."""

    output_format = output_format.lower()
    written: List[Path] = []

    json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format in {"both", "json"}:
        json_path.write_text(
            json.dumps(list(records), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(json_path)

    if output_format in {"both", "csv"}:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["topic", "quote", "source", "title", "filename", "quote_by"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, "") for key in writer.fieldnames})
        written.append(csv_path)

    if not written:
        raise ValueError("output_format must be one of: json, csv, both")

    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a quote dataset from the primary KB.")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=PRIMARY_KB_DIR,
        help="Root directory containing markdown documents.",
    )
    parser.add_argument(
        "--max-quotes-per-file",
        type=int,
        default=4,
        help="Maximum number of rows to keep per markdown file.",
    )
    parser.add_argument(
        "--format",
        choices=("both", "json", "csv"),
        default="both",
        help="Output format to write.",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=QUOTES_JSON_PATH,
        help="Destination for the JSON file.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=QUOTES_CSV_PATH,
        help="Destination for the CSV file.",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Optional topic filter for the rows printed to stdout.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for generating the quote retrieval table."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    rows = build_quote_rows(kb_dir=args.kb_dir, max_quotes_per_file=args.max_quotes_per_file)
    written = write_quotes_dataset(
        rows,
        json_path=args.json_path,
        csv_path=args.csv_path,
        output_format=args.format,
    )

    if args.topic:
        rows = filter_quotes_by_topic(rows, args.topic)

    print(f"Built {len(rows)} quote rows from {args.kb_dir}")
    for path in written:
        print(f"Wrote {path}")
    if args.topic:
        print(f"Matched topic filter: {args.topic}")

    preview = rows[:5]
    if preview:
        print("\nPreview:")
        for row in preview:
            print(f"- {row['topic']} | {row['quote_by']}: {row['quote']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
