"""Merge the pretrained summary index and quote index into one JSON payload."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_INDEX_PATH = REPO_ROOT / "knowledge_base" / "pretrained_summary_index.json"
QUOTE_INDEX_PATH = REPO_ROOT / "knowledge_base" / "quotes.json"
MERGED_CONTEXT_PATH = REPO_ROOT / "knowledge_base" / "merged_context.json"


def _read_json(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        return [dict(item) for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict):
        for key in ("items", "rows", "records", "summaries", "quotes"):
            value = loaded.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_filename(record: Dict[str, object]) -> str:
    for key in ("filename", "source_filename", "path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_first_non_empty(record: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _merge_summary_records(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = OrderedDict()
    for record in records:
        filename = _normalize_filename(record)
        if not filename:
            continue

        current = merged.setdefault(
            filename,
            {
                "title": "",
                "filename": filename,
                "source": "",
                "summary": "",
                "quotes": [],
            },
        )

        if not current["title"]:
            current["title"] = _pick_first_non_empty(record, ("title",))
        if not current["source"]:
            current["source"] = _pick_first_non_empty(record, ("source",))
        if not current["summary"]:
            current["summary"] = _pick_first_non_empty(record, ("summary",))

    return merged


def _merge_quote_records(
    merged: Dict[str, Dict[str, object]],
    records: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    for record in records:
        filename = _normalize_filename(record)
        if not filename:
            continue

        current = merged.setdefault(
            filename,
            {
                "title": "",
                "filename": filename,
                "source": "",
                "summary": "",
                "quotes": [],
            },
        )

        if not current["title"]:
            current["title"] = _pick_first_non_empty(record, ("title",))
        if not current["source"]:
            current["source"] = _pick_first_non_empty(record, ("source",))

        quotes = current.setdefault("quotes", [])
        if not isinstance(quotes, list):
            quotes = []
            current["quotes"] = quotes

        quote_text = _pick_first_non_empty(record, ("quote",))
        quote_by = _pick_first_non_empty(record, ("quote_by",))
        topic = _pick_first_non_empty(record, ("topic",))

        quote_entry = {
            "topic": topic,
            "quote": quote_text,
            "quote_by": quote_by,
            "source": _pick_first_non_empty(record, ("source",)),
            "title": _pick_first_non_empty(record, ("title",)),
            "filename": filename,
        }

        dedupe_key = _normalize_text(quote_text) + "||" + _normalize_text(quote_by)
        seen = current.setdefault("_seen_quote_keys", [])
        if dedupe_key and dedupe_key not in seen:
            seen.append(dedupe_key)
            quotes.append(quote_entry)

    return merged


def build_merged_context() -> List[Dict[str, object]]:
    """Build one deduplicated JSON-ready payload from both indexes."""

    summaries = _read_json(SUMMARY_INDEX_PATH)
    quotes = _read_json(QUOTE_INDEX_PATH)

    merged = _merge_summary_records(summaries)
    merged = _merge_quote_records(merged, quotes)

    output: List[Dict[str, object]] = []
    for record in merged.values():
        record.pop("_seen_quote_keys", None)
        quotes_list = record.get("quotes", [])
        if isinstance(quotes_list, list):
            quotes_list.sort(
                key=lambda item: (
                    _normalize_text(item.get("topic")),
                    _normalize_text(item.get("quote_by")),
                    _normalize_text(item.get("quote")),
                )
            )
        output.append(record)

    output.sort(key=lambda item: (_normalize_text(item.get("title")), _normalize_text(item.get("filename"))))
    return output


def write_merged_context(output_path: Path = MERGED_CONTEXT_PATH) -> Path:
    """Write the merged context payload to disk."""

    payload = build_merged_context()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge quote and summary indexes into one JSON file.")
    parser.add_argument("--output", type=Path, default=MERGED_CONTEXT_PATH, help="Destination JSON path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    output_path = write_merged_context(args.output)
    print(f"Wrote merged context to {output_path}")
    print(f"Documents: {len(build_merged_context())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
