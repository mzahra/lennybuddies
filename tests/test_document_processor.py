import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai import OpenAIError

import document_processor as dp


SAMPLE_MARKDOWN = """---
title: An AI glossary
subtitle: The most common AI terms explained, simply
date: 2025-06-24
tags: []
---
This is the body content of the article about AI terminology.
"""


def _fake_openai_response(content_dict):
    """Builds a fake OpenAI chat.completions.create() response shape."""
    message = SimpleNamespace(content=json.dumps(content_dict))
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class TestLoadDocument:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.md"
        with pytest.raises(dp.DocumentProcessingError, match="not found"):
            dp.load_document(str(missing))

    def test_valid_file_loads_metadata_and_body(self, tmp_path):
        md_file = tmp_path / "article.md"
        md_file.write_text(SAMPLE_MARKDOWN)

        metadata, body = dp.load_document(str(md_file))

        assert metadata["title"] == "An AI glossary"
        assert metadata["subtitle"] == "The most common AI terms explained, simply"
        assert metadata["tags"] == []
        assert metadata["source_filename"] == str(md_file)
        assert "body content" in body


class TestProcessDocument:
    def setup_method(self):
        self.metadata = {
            "title": "An AI glossary",
            "subtitle": "",
            "date": "2025-06-24",
            "tags": [],
            "source_filename": "test.md",
        }

    def test_empty_body_raises(self):
        with pytest.raises(dp.DocumentProcessingError, match="Empty body content"):
            dp.process_document(self.metadata, "   ")

    def test_api_error_raises_document_processing_error(self):
        with patch.object(
            dp.client.chat.completions, "create", side_effect=OpenAIError("rate limited")
        ):
            with pytest.raises(dp.DocumentProcessingError, match="OpenAI API call failed"):
                dp.process_document(self.metadata, "Some real body text.")

    def test_invalid_json_response_raises(self):
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))]
        )
        with patch.object(dp.client.chat.completions, "create", return_value=fake_response):
            with pytest.raises(dp.DocumentProcessingError, match="invalid JSON"):
                dp.process_document(self.metadata, "Some real body text.")

    def test_missing_required_field_raises(self):
        fake_response = _fake_openai_response({"keywords": ["a", "b"]})  # no "summary"
        with patch.object(dp.client.chat.completions, "create", return_value=fake_response):
            with pytest.raises(dp.DocumentProcessingError, match="missing required field"):
                dp.process_document(self.metadata, "Some real body text.")

    def test_successful_processing_returns_merged_result(self):
        fake_response = _fake_openai_response({
            "summary": "A short summary.",
            "keywords": ["AI", "glossary"],
            "tags": ["technology"],
        })
        with patch.object(dp.client.chat.completions, "create", return_value=fake_response):
            result = dp.process_document(self.metadata, "Some real body text.")

        assert result["title"] == "An AI glossary"
        assert result["summary"] == "A short summary."
        assert result["keywords"] == ["AI", "glossary"]
        # front matter tags were empty, so LLM-generated tags should be used
        assert result["tags"] == ["technology"]
        assert result["model"] == "gpt-4o-mini"

    def test_frontmatter_tags_take_priority_over_llm_tags(self):
        metadata_with_tags = dict(self.metadata, tags=["leadership"])
        fake_response = _fake_openai_response({
            "summary": "A short summary.",
            "keywords": ["AI"],
            "tags": ["technology"],
        })
        with patch.object(dp.client.chat.completions, "create", return_value=fake_response):
            result = dp.process_document(metadata_with_tags, "Some real body text.")

        assert result["tags"] == ["leadership"]


class TestProcessKnowledgeBase:
    def test_skips_missing_folder_without_raising(self, tmp_path):
        missing_folder = str(tmp_path / "does_not_exist")
        results = dp.process_knowledge_base(folders=[missing_folder])
        assert results == []

    def test_continues_after_one_bad_file(self, tmp_path):
        good_file = tmp_path / "good.md"
        good_file.write_text(SAMPLE_MARKDOWN)
        bad_file = tmp_path / "bad.md"
        # Malformed YAML front matter (unbalanced brackets) — frontmatter.load
        # raises a YAML parse error on this, which load_document should wrap
        # in DocumentProcessingError.
        bad_file.write_text("---\nthis: [is, broken, yaml\n---\nbody text\n")

        fake_response = _fake_openai_response({
            "summary": "A short summary.",
            "keywords": ["AI"],
            "tags": ["technology"],
        })

        with patch.object(dp.client.chat.completions, "create", return_value=fake_response):
            results = dp.process_knowledge_base(folders=[str(tmp_path)])

        # both files exist, but only the parseable one should produce a result
        assert len(results) == 1
        assert results[0]["title"] == "An AI glossary"


class TestSaveProcessedDocuments:
    def test_creates_parent_dirs_and_writes_file(self, tmp_path):
        out_path = tmp_path / "nested" / "processed.json"
        docs = [{"title": "x"}]
        dp.save_processed_documents(docs, path=str(out_path))

        assert out_path.exists()
        assert json.loads(out_path.read_text()) == docs
