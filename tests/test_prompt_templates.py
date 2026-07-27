import json
from pathlib import Path

import pytest

import prompt_templates as pt


SAMPLE_TEMPLATE_LIBRARY = {
    "Promises vs Reality": {
        "title": "Promises vs Reality",
        "structure": "What [X] promised: [Y].\nWhat [X] delivered: [Z].",
        "example": "What AI promised: magic.\nWhat AI delivered: more work.",
    }
}

SAMPLE_ARTICLE = {
    "title": "An AI glossary",
    "summary": "A glossary of common AI terms for beginners.",
    "source_filename": "knowledge_base/test/an-ai-glossary.md",
}

BASE_INPUTS = {
    "topic": "AI",
    "word_count": "150-500",
    "language": "English",
    "tone": "Professional",
    "style": "Storytelling",
    "goal": "Build Personal Brand",
    "target_audience": "Product Managers",
    "call_to_action": "Share Your Thoughts",
    "template": "Promises vs Reality",
}


class TestLoadTemplateLibrary:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "missing.json"
        with pytest.raises(pt.PromptTemplateError, match="not found"):
            pt.load_template_library(path=missing)

    def test_malformed_json_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        with pytest.raises(pt.PromptTemplateError, match="Malformed JSON"):
            pt.load_template_library(path=bad_file)

    def test_valid_file_loads(self, tmp_path):
        good_file = tmp_path / "templates.json"
        good_file.write_text(json.dumps(SAMPLE_TEMPLATE_LIBRARY))
        loaded = pt.load_template_library(path=good_file)
        assert loaded == SAMPLE_TEMPLATE_LIBRARY


class TestFormatSourceContext:
    def test_formats_valid_articles(self):
        context = pt.format_source_context([SAMPLE_ARTICLE])
        assert "An AI glossary" in context
        assert "glossary of common AI terms" in context
        assert "an-ai-glossary.md" in context

    def test_skips_malformed_articles(self):
        malformed = {"title": "No summary here"}  # missing summary/source_filename
        context = pt.format_source_context([malformed, SAMPLE_ARTICLE])
        # only the valid article should appear
        assert "An AI glossary" in context
        assert context.count("Source") == 1


class TestBuildPrompt:
    def setup_method(self):
        # monkeypatch the module-level template library each test uses
        self._original_library = pt.TEMPLATE_LIBRARY
        pt.TEMPLATE_LIBRARY = SAMPLE_TEMPLATE_LIBRARY

    def teardown_method(self):
        pt.TEMPLATE_LIBRARY = self._original_library

    def test_missing_inputs_key_raises(self):
        with pytest.raises(pt.PromptTemplateError, match="inputs"):
            pt.build_prompt({"source_articles": [SAMPLE_ARTICLE]})

    def test_missing_source_articles_key_raises(self):
        with pytest.raises(pt.PromptTemplateError, match="source_articles"):
            pt.build_prompt({"inputs": BASE_INPUTS})

    def test_missing_required_input_field_raises(self):
        incomplete_inputs = dict(BASE_INPUTS)
        del incomplete_inputs["tone"]
        payload = {"inputs": incomplete_inputs, "source_articles": [SAMPLE_ARTICLE]}
        with pytest.raises(pt.PromptTemplateError, match="tone"):
            pt.build_prompt(payload)

    def test_unknown_template_raises(self):
        bad_inputs = dict(BASE_INPUTS)
        bad_inputs["template"] = "Nonexistent Template"
        payload = {"inputs": bad_inputs, "source_articles": [SAMPLE_ARTICLE]}
        with pytest.raises(pt.PromptTemplateError, match="Unknown template"):
            pt.build_prompt(payload)

    def test_empty_source_articles_raises(self):
        payload = {"inputs": BASE_INPUTS, "source_articles": []}
        with pytest.raises(pt.PromptTemplateError, match="No usable source articles"):
            pt.build_prompt(payload)

    def test_empty_template_library_raises(self):
        pt.TEMPLATE_LIBRARY = {}
        payload = {"inputs": BASE_INPUTS, "source_articles": [SAMPLE_ARTICLE]}
        with pytest.raises(pt.PromptTemplateError, match="Template library"):
            pt.build_prompt(payload)

    def test_valid_payload_builds_prompt(self):
        payload = {"inputs": BASE_INPUTS, "source_articles": [SAMPLE_ARTICLE]}
        prompt = pt.build_prompt(payload)
        assert "An AI glossary" in prompt
        assert "Professional" in prompt
        assert "Share Your Thoughts" in prompt
