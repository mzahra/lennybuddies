from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai import OpenAIError

import llm_integration as li
from knowledge_base import KnowledgeBaseError
from prompt_templates import PromptTemplateError


def _fake_response(text):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


SAMPLE_PAYLOAD = {
    "inputs": {
        "topic": "AI",
        "word_count": "150-500",
        "language": "English",
        "tone": "Professional",
        "style": "Storytelling",
        "goal": "Build Personal Brand",
        "target_audience": "Product Managers",
        "call_to_action": "Share Your Thoughts",
        "template": "Promises vs Reality",
    },
    "source_articles": [
        {
            "title": "An AI glossary",
            "summary": "A glossary of common AI terms.",
            "source_filename": "test.md",
        }
    ],
}


class TestGenerateDraft:
    def test_empty_prompt_raises(self):
        with pytest.raises(li.LLMIntegrationError, match="empty prompt"):
            li.generate_draft("   ")

    def test_api_error_raises(self):
        with patch.object(li._client.chat.completions, "create", side_effect=OpenAIError("boom")):
            with pytest.raises(li.LLMIntegrationError, match="Draft generation failed"):
                li.generate_draft("write something")

    def test_empty_response_raises(self):
        with patch.object(
            li._client.chat.completions, "create", return_value=_fake_response("   ")
        ):
            with pytest.raises(li.LLMIntegrationError, match="empty content"):
                li.generate_draft("write something")

    def test_successful_draft_returns_stripped_text(self):
        with patch.object(
            li._client.chat.completions, "create", return_value=_fake_response("  Hello world  ")
        ):
            result = li.generate_draft("write something")
        assert result == "Hello world"


class TestPolishDraft:
    def test_empty_draft_raises(self):
        with pytest.raises(li.LLMIntegrationError, match="empty draft"):
            li.polish_draft("")

    def test_api_error_raises(self):
        with patch.object(li._client.chat.completions, "create", side_effect=OpenAIError("boom")):
            with pytest.raises(li.LLMIntegrationError, match="Polish pass failed"):
                li.polish_draft("some draft text")

    def test_successful_polish_returns_stripped_text(self):
        with patch.object(
            li._client.chat.completions, "create", return_value=_fake_response("  Polished  ")
        ):
            result = li.polish_draft("some draft text")
        assert result == "Polished"


class TestGeneratePost:
    def test_prompt_build_failure_raises_llm_integration_error(self):
        with patch("llm_integration.build_prompt", side_effect=PromptTemplateError("bad payload")):
            with pytest.raises(li.LLMIntegrationError, match="Failed to build prompt"):
                li.generate_post(SAMPLE_PAYLOAD)

    def test_full_generate_post_runs_draft_then_polish(self):
        with patch("llm_integration.build_prompt", return_value="a real prompt"), \
             patch.object(li, "generate_draft", return_value="draft text") as mock_draft, \
             patch.object(li, "polish_draft", return_value="final text") as mock_polish:
            result = li.generate_post(SAMPLE_PAYLOAD)

        mock_draft.assert_called_once_with("a real prompt")
        mock_polish.assert_called_once_with("draft text")
        assert result == "final text"


class TestGeneratePostFromInputs:
    def test_missing_topic_raises(self):
        with pytest.raises(li.LLMIntegrationError, match="non-empty 'topic'"):
            li.generate_post_from_inputs({"word_count": "150-500"})

    def test_non_dict_inputs_raises(self):
        with pytest.raises(li.LLMIntegrationError, match="non-empty 'topic'"):
            li.generate_post_from_inputs(None)

    def test_knowledge_base_error_wrapped(self):
        with patch("llm_integration.get_relevant_summaries", side_effect=KnowledgeBaseError("no docs")):
            with pytest.raises(li.LLMIntegrationError, match="Knowledge base filtering failed"):
                li.generate_post_from_inputs({"topic": "AI"})

    def test_full_pipeline_wires_filter_save_and_generate(self):
        sample_articles = [{"title": "x", "summary": "y", "source_filename": "z.md"}]
        with patch("llm_integration.get_relevant_summaries", return_value=sample_articles) as mock_filter, \
             patch("llm_integration.save_filtered_documents") as mock_save, \
             patch.object(li, "generate_post", return_value="final post") as mock_generate:
            result = li.generate_post_from_inputs({"topic": "AI"}, top_k=3)

        mock_filter.assert_called_once_with("AI", top_k=3)
        mock_save.assert_called_once_with(sample_articles)
        mock_generate.assert_called_once()
        assert result == "final post"
