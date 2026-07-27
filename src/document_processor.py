import os
import glob
import json
import logging

import frontmatter
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
logger = logging.getLogger(__name__)

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your .env file before running "
        "document_processor.py."
    )
client = OpenAI(api_key=_api_key)

SYSTEM_PROMPT = """You are a content analysis assistant for a newsletter/podcast processing pipeline. You will be given existing metadata (title, subtitle, date, tags, source_filename) plus the raw body text of the content. Your job is to generate the fields: "summary", "keywords", and "tags".

Rules for "summary":
- 2-4 sentences capturing the core argument or main takeaways
- Written in clear, complete prose (no cut-off sentences, no stray quote marks)
- Do not include the title/subtitle verbatim — add value beyond them

Rules for "keywords":
- 5-8 items, each a real topic, concept, named entity, or theme actually discussed in the content
- Never include: URLs, file paths, footnote/reference numbers, generic phrases like "part 1", 'step 1', firstly, secondly, or fragments of the title/subtitle
- Prefer specific nouns/phrases a reader would use to search for this content

Rules for "tags":
- 2-4 broad category labels that classify what this content is about (e.g. "leadership", "AI/ML", "product growth", "marketing", "career advice")
- These should be higher-level than keywords — think of them as folder/category labels, not specific topics
- Do not include content-format labels like "podcast" or "newsletter" — tags describe subject matter, not format
- Use consistent, reusable terms so similar content across documents shares the same tags

Return ONLY valid JSON in this exact structure, with no other text:
{
  "summary": "...",
  "keywords": ["...", "...", "..."],
  "tags": ["...", "..."]
}
"""


class DocumentProcessingError(Exception):
    """Raised when a single document fails to load or process."""


def load_document(filepath):
    """Loads a markdown file's front matter + body.

    Raises:
        DocumentProcessingError: if the file doesn't exist or can't be
        parsed as front matter.
    """
    if not os.path.isfile(filepath):
        raise DocumentProcessingError(f"File not found: {filepath}")

    try:
        post = frontmatter.load(filepath)
    except Exception as exc:
        raise DocumentProcessingError(
            f"Failed to parse front matter in {filepath}: {exc}"
        ) from exc

    metadata = {
        "title": post.get("title", ""),
        "subtitle": post.get("subtitle", ""),
        "date": post.get("date", ""),
        "tags": post.get("tags", []),
        "source_filename": filepath,
    }
    return metadata, post.content


def process_document(metadata, body_text, model_name="gpt-4o-mini"):
    """Calls the LLM to generate summary/keywords/tags for one document.

    Raises:
        DocumentProcessingError: if the API call fails, the response isn't
        valid JSON, or required fields are missing from the response.
    """
    if not body_text or not body_text.strip():
        raise DocumentProcessingError(
            f"Empty body content for {metadata.get('source_filename', 'unknown file')}"
        )

    user_prompt = f"""Existing metadata:
Title: {metadata['title']}
Subtitle: {metadata['subtitle']}
Tags: {metadata['tags']}

Body content:
{body_text}"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
    except OpenAIError as exc:
        raise DocumentProcessingError(
            f"OpenAI API call failed for {metadata.get('source_filename', 'unknown file')}: {exc}"
        ) from exc

    raw_content = response.choices[0].message.content

    try:
        llm_output = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise DocumentProcessingError(
            f"LLM returned invalid JSON for {metadata.get('source_filename', 'unknown file')}: {exc}"
        ) from exc

    missing = [key for key in ("summary", "keywords") if key not in llm_output]
    if missing:
        raise DocumentProcessingError(
            f"LLM response missing required field(s) {missing} for "
            f"{metadata.get('source_filename', 'unknown file')}"
        )

    # only use LLM-generated tags if front matter didn't already provide any
    tags = metadata["tags"] if metadata["tags"] else llm_output.get("tags", [])

    return {
        **metadata,
        "tags": tags,
        "keywords": llm_output["keywords"],
        "summary": llm_output["summary"],
        "model": model_name
    }


def process_knowledge_base(folders=None):
    """Processes every .md file in the given folders.

    Individual file failures are logged and skipped rather than aborting
    the whole batch, so one bad file doesn't block processing the rest.
    """
    if folders is None:
        folders = ["knowledge_base/test/"]

    results = []
    for folder in folders:
        if not os.path.isdir(folder):
            logger.warning("Skipping missing folder: %s", folder)
            continue

        for filepath in glob.glob(os.path.join(folder, "*.md")):
            try:
                metadata, body_text = load_document(filepath)
                result = process_document(metadata, body_text)
                results.append(result)
                print(f"Processed: {filepath}")
            except DocumentProcessingError as exc:
                logger.error("Skipping %s: %s", filepath, exc)
                print(f"ERROR processing {filepath}: {exc}")

    return results


def save_processed_documents(documents, path="knowledge_base/processed/processed_documents_test.json"):
    """Writes processed documents to disk, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(documents, f, indent=2)
    except OSError as exc:
        raise DocumentProcessingError(f"Failed to write {path}: {exc}") from exc


if __name__ == "__main__":
    all_results = process_knowledge_base()
    save_processed_documents(all_results)
    print(f"Done. {len(all_results)} document(s) processed successfully.")
