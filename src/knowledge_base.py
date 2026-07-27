"""Knowledge base filtering layer.

Loads the processed documents (output of document_processor.py) and,
given a user topic/prompt, returns the 1-5 most relevant documents so
prompt_templates.py can ground the LinkedIn post in real source material.

Approach: offline TF-IDF + cosine similarity (scikit-learn). No API calls,
no network, no API key needed — everything runs locally against the text
already in processed_documents.json (title, summary, tags, keywords).
This is the function referenced in prompt_templates.py's docstring:
get_relevant_summaries(topic) -> list[dict].
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DOCS_PATH = "knowledge_base/processed/processed_documents.json"
DEFAULT_FILTERED_PATH = "knowledge_base/filtered/filtered_documents.json"


class KnowledgeBaseError(Exception):
    """Raised when the knowledge base can't be loaded, filtered, or saved."""


def load_documents(path: str = DEFAULT_DOCS_PATH) -> List[Dict[str, Any]]:
    """Loads the processed_documents.json produced by document_processor.py.

    Raises:
        KnowledgeBaseError: if the file is missing, unreadable, malformed
        JSON, or not a JSON list.
    """
    if not os.path.isfile(path):
        raise KnowledgeBaseError(
            f"Processed documents file not found: {path}. "
            "Run document_processor.py first."
        )

    try:
        with open(path, "r") as f:
            documents = json.load(f)
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Malformed JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Failed to read {path}: {exc}") from exc

    if not isinstance(documents, list):
        raise KnowledgeBaseError(
            f"Expected a JSON list of documents in {path}, got {type(documents).__name__}"
        )

    return documents


def _doc_text(doc: Dict[str, Any]) -> str:
    """Builds the text representation of a document used for matching.

    Combines title, summary, tags, and keywords so the similarity score
    reflects both the topic and the specific content, not just the title.
    Keywords/tags are repeated once for extra TF-IDF weight since they're
    already hand-picked as the most relevant terms for the document.
    """
    tags = " ".join(doc.get("tags", []) or [])
    keywords = " ".join(doc.get("keywords", []) or [])
    return (
        f"{doc.get('title', '')} {doc.get('subtitle', '')} "
        f"{doc.get('summary', '')} {tags} {keywords} {tags} {keywords}"
    )


def get_relevant_summaries(
    topic: str,
    documents: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    min_score: float = 0.05,
) -> List[Dict[str, Any]]:
    """Returns the 1-top_k documents most relevant to the given topic.

    Args:
        topic: the user's input prompt/topic for the LinkedIn post.
        documents: optional pre-loaded documents (mainly for testing);
                   defaults to loading DEFAULT_DOCS_PATH.
        top_k: max number of documents to return (hard cap per the
               "1-5 related documents" requirement).
        min_score: minimum TF-IDF cosine similarity to be considered
                   relevant. TF-IDF scores run lower than embedding
                   similarity scores, so this threshold is intentionally
                   low. If nothing clears it, we still return the single
                   best match so the pipeline always has something to
                   ground the post in.

    Returns:
        A list of 1 to top_k document dicts, ordered most-to-least relevant.

    Raises:
        KnowledgeBaseError: if topic is empty/not a string, top_k is not a
        positive integer, or the documents can't be loaded.
    """
    if not isinstance(topic, str) or not topic.strip():
        raise KnowledgeBaseError("topic must be a non-empty string")

    if not isinstance(top_k, int) or top_k < 1:
        raise KnowledgeBaseError("top_k must be a positive integer")

    if documents is None:
        documents = load_documents()

    if not documents:
        return []

    doc_texts = [_doc_text(doc) for doc in documents]

    try:
        # Fit TF-IDF over the corpus + the topic query together so they
        # share the same vocabulary/vector space.
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(doc_texts + [topic])
    except ValueError:
        # Happens if every document + the topic contain only stop words /
        # no usable vocabulary at all. Fall back to returning the first
        # top_k documents unranked rather than crashing the pipeline.
        return documents[:top_k]

    doc_vectors = tfidf_matrix[:-1]
    topic_vector = tfidf_matrix[-1]

    scores = cosine_similarity(topic_vector, doc_vectors)[0]

    scored = list(zip(documents, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    relevant = [doc for doc, score in scored if score >= min_score][:top_k]

    # Always return at least the single best match, even if none clear
    # min_score — an imperfect grounding source beats generating with none.
    if not relevant:
        relevant = [scored[0][0]]

    return relevant


def save_filtered_documents(
    documents: List[Dict[str, Any]],
    path: str = DEFAULT_FILTERED_PATH,
) -> None:
    """Writes filtered documents to a JSON file in the same format/shape
    as processed_documents.json (same keys, same list-of-dicts structure).

    Raises:
        KnowledgeBaseError: if the parent directory can't be created or
        the file can't be written.
    """
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        with open(path, "w") as f:
            json.dump(documents, f, indent=2)
    except OSError as exc:
        raise KnowledgeBaseError(f"Failed to write {path}: {exc}") from exc


if __name__ == "__main__":
    test_topic = "AI"
    try:
        results = get_relevant_summaries(test_topic)
        save_filtered_documents(results)
        print(f"Topic: {test_topic}")
        print(f"Matched {len(results)} document(s), saved to {DEFAULT_FILTERED_PATH}:\n")
        for doc in results:
            print(f"- {doc['title']}")
    except KnowledgeBaseError as exc:
        print(f"ERROR: {exc}")
