import json

import pytest

import knowledge_base as kb


SAMPLE_DOCS = [
    {
        "title": "An AI glossary",
        "subtitle": "The most common AI terms explained, simply",
        "date": "2025-06-24",
        "tags": ["artificial intelligence", "education"],
        "source_filename": "knowledge_base/test/an-ai-glossary.md",
        "keywords": ["large language models", "prompt engineering", "generative AI"],
        "summary": "A glossary of common AI terms for beginners.",
        "model": "gpt-4o-mini",
    },
    {
        "title": "Contrarian leadership truths",
        "subtitle": "",
        "date": "2025-12-28",
        "tags": ["leadership", "team dynamics"],
        "source_filename": "knowledge_base/test/matt-macinnis.md",
        "keywords": ["feedback culture", "under-staffing", "extraordinary effort"],
        "summary": "A leader discusses contrarian views on team intensity and feedback.",
        "model": "gpt-4o-mini",
    },
]


class TestLoadDocuments:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises(kb.KnowledgeBaseError, match="not found"):
            kb.load_documents(str(missing))

    def test_malformed_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        with pytest.raises(kb.KnowledgeBaseError, match="Malformed JSON"):
            kb.load_documents(str(bad_file))

    def test_non_list_json_raises(self, tmp_path):
        bad_file = tmp_path / "not_a_list.json"
        bad_file.write_text(json.dumps({"oops": "this should be a list"}))
        with pytest.raises(kb.KnowledgeBaseError, match="Expected a JSON list"):
            kb.load_documents(str(bad_file))

    def test_valid_file_loads(self, tmp_path):
        good_file = tmp_path / "docs.json"
        good_file.write_text(json.dumps(SAMPLE_DOCS))
        loaded = kb.load_documents(str(good_file))
        assert loaded == SAMPLE_DOCS


class TestGetRelevantSummaries:
    def test_empty_topic_raises(self):
        with pytest.raises(kb.KnowledgeBaseError, match="non-empty string"):
            kb.get_relevant_summaries("   ", documents=SAMPLE_DOCS)

    def test_non_string_topic_raises(self):
        with pytest.raises(kb.KnowledgeBaseError, match="non-empty string"):
            kb.get_relevant_summaries(None, documents=SAMPLE_DOCS)

    def test_invalid_top_k_raises(self):
        with pytest.raises(kb.KnowledgeBaseError, match="top_k must be"):
            kb.get_relevant_summaries("AI", documents=SAMPLE_DOCS, top_k=0)

    def test_empty_documents_returns_empty_list(self):
        assert kb.get_relevant_summaries("AI", documents=[]) == []

    def test_returns_relevant_doc_first(self):
        results = kb.get_relevant_summaries(
            "large language models and prompt engineering",
            documents=SAMPLE_DOCS,
            top_k=5,
        )
        assert len(results) >= 1
        assert results[0]["title"] == "An AI glossary"

    def test_respects_top_k_cap(self):
        results = kb.get_relevant_summaries("AI leadership", documents=SAMPLE_DOCS, top_k=1)
        assert len(results) == 1

    def test_always_returns_at_least_one_match(self):
        # A topic sharing no vocabulary with either doc should still
        # return the single best (if imperfect) match, never an empty list.
        results = kb.get_relevant_summaries("xyzabc123 completely unrelated", documents=SAMPLE_DOCS)
        assert len(results) >= 1


class TestSaveFilteredDocuments:
    def test_writes_file_and_creates_dirs(self, tmp_path):
        out_path = tmp_path / "nested" / "filtered.json"
        kb.save_filtered_documents(SAMPLE_DOCS, path=str(out_path))

        assert out_path.exists()
        saved = json.loads(out_path.read_text())
        assert saved == SAMPLE_DOCS
