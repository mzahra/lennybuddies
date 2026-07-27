"""Combine quote and summary retrieval into one topic context payload."""

from __future__ import annotations

from typing import Dict, List

try:
    from quotes import build_quote_rows, filter_quotes_by_topic, load_quotes_json, write_quotes_dataset
except ImportError:  # pragma: no cover - supports `python -m src.topic_context`
    from src.quotes import build_quote_rows, filter_quotes_by_topic, load_quotes_json, write_quotes_dataset

try:
    from pretrained_summariser import build_summary_rows, filter_summary_rows_by_topic, load_summary_index, write_summary_index
except ImportError:  # pragma: no cover - supports `python -m src.topic_context`
    from src.pretrained_summariser import build_summary_rows, filter_summary_rows_by_topic, load_summary_index, write_summary_index


def load_topic_quotes(topic: str, rebuild: bool = False) -> List[Dict[str, str]]:
    """Return the quote rows that match a selected topic."""

    rows = load_quotes_json()
    if rebuild or not rows:
        rows = build_quote_rows()
        write_quotes_dataset(rows)
    return filter_quotes_by_topic(rows, topic)


def load_topic_summaries(topic: str, rebuild: bool = False) -> List[Dict[str, str]]:
    """Return the summary rows that match a selected topic."""

    rows = load_summary_index()
    if rebuild or not rows:
        rows = build_summary_rows()
        write_summary_index()
    return filter_summary_rows_by_topic(rows, topic)


def build_topic_payload(
    topic: str,
    quote_limit: int = 8,
    summary_limit: int = 8,
    rebuild: bool = False,
) -> Dict[str, List[Dict[str, str]]]:
    """Return one topic payload with both quotes and summaries."""

    quotes = load_topic_quotes(topic=topic, rebuild=rebuild)[:quote_limit]
    summaries = load_topic_summaries(topic=topic, rebuild=rebuild)[:summary_limit]
    return {
        "topic": topic,
        "quotes": quotes,
        "summaries": summaries,
    }
