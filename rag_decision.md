# RAG vs Non-RAG Decision

**Choice: Non-RAG (context injection with local relevance filtering)**

**Corpus size & structure:** Our knowledge base is roughly 60 processed documents (newsletter + podcast summaries), static once processing finished. This is small enough to reason about without a vector database.

**Complexity vs. project scope:** A full vector RAG stack (embeddings + a vector store like FAISS/Chroma) was out of scope for our timeframe. Instead, `knowledge_base.py` uses TF-IDF + cosine similarity (scikit-learn) to score every document against the user's topic locally — no external embedding API, no vector database, no added latency or cost per request.

**Context window, cost, and latency:** Injecting all ~60 summaries into every prompt would be wasteful and unnecessary — most aren't relevant to any given topic. TF-IDF lets us cheaply narrow to the 1-5 most relevant documents before ever calling the LLM, keeping each generation to exactly 2 API calls (draft + polish) regardless of corpus size.

**Why we call this non-RAG, not RAG:** TF-IDF/cosine similarity is a classical, sparse, local relevance-scoring technique — not the dense-embedding, vector-database retrieval most people mean by "RAG" in practice. We treat it as a lightweight pre-filtering step within a non-RAG context-injection design, not a RAG pipeline.

**When we'd revisit this:** If the corpus grew substantially larger, or if TF-IDF's keyword-based matching started missing semantically related content that doesn't share exact terms with the topic, we'd move to embeddings-based retrieval.
