# project_structure.md

> DRAFT — starting point for team discussion, not final. Team: Mudit, Jay, Parisa, Zahra.

## 1. Project identity

- **Project name:** Lenny Pulse *(working title — team to confirm)*
- **Primary project type:** R&D
- **Defined goal (1 sentence):** Generate authentic, brand-differentiated LinkedIn posts from Lenny's Podcast episode transcripts, grounded in a two-tier knowledge base (episode content + a defined voice/style guide).
- **Start / end:** Day 1 morning → Day 2 presentation
- **Why this is a project (tick ≥4):**
  ☑ defined goal ☑ limited resources ☑ interdisciplinary ☑ complex ☑ novel ☑ defined start/end
  ☐ responsibility for results *(team to discuss — arguably applies too)*

## 2. Objectives (Quality / Time / Cost)

| Constraint | Objective |
|---|---|
| **Quality** | Generated posts must be traceable to specific episode content (no invented claims), written in a consistent, defined voice distinct from generic AI output, and clearly attributed to the source episode. |
| **Time** | Hard deadline: Day 2 presentation slot. Internal milestone: working document → LLM → output pipeline running end-to-end by end of Day 1. |
| **Cost** | Free-tier LLM usage only (e.g. gpt-4o-mini); cap test/dev calls to a shared daily budget so 4 people don't burn quota independently. |

## 3. Stakeholder analysis

| Role | Interest | Influence | Quadrant | Engagement |
|---|---|---|---|---|
| Content creator / consultant (end user) | H | L | II | Defines what "good" output looks like; gives feedback on tone/voice during testing |
| Instructor (grader) | H | H | I | Sets rubric, reviews demo + deliverables — manage closely, keep informed of scope choices |
| Lenny / podcast IP source | L | H | III | Non-builder. Not directly consulted, but shapes how we handle attribution and avoid verbatim reproduction |
| LinkedIn audience (readers) | L | L | IV | Passive — content must read as genuine, non-spammy, platform-appropriate |

*(Pass bar met: 4 roles, ≥1 non-builder — Lenny — every row has a quadrant.)*

## 4. Requirements → implementation

**Use case (1 sentence):** A content creator picks a Lenny's Podcast episode and gets back 2-3 LinkedIn-ready posts, written in their own defined voice, grounded in what was actually said.

**Must have:**

| ID | Must requirement | Maps to (file/module) | How we verify |
|---|---|---|---|
| M1 | Ingest primary KB markdown (episode transcript/notes + style guide) | `src/document_processor.py`, `knowledge_base/primary/` | Loader returns parsed docs with no errors on a test run |
| M2 | Ingest secondary KB markdown (industry/PM trend context) | `src/document_processor.py`, `knowledge_base/secondary/` | Same loader handles both KB folders |
| M3 | Call LLM with KB context | `src/llm_integration.py` | Returns non-empty, on-topic response referencing episode content |
| M4 | Reusable prompt templates (≥2) | `src/prompt_templates.py` | Two templates produce visibly different tone/structure on the same input |
| M5 | End-to-end pipeline command | `src/content_pipeline.py`, `src/main.py` | One command: transcript in → post(s) out |
| M6 | Uniqueness evidence vs. generic ChatGPT | `src/content_pipeline.py` + comparison doc | Side-by-side example exists, differences documented |
| M7 | RAG decision documented | `rag_decision.md` | Choice stated + ≥3 criteria addressed |
| M8 | Project structured + agents guided | `project_structure.md`, `agents.md` | All sections complete; `agents.md` actually referenced when prompting agents |

**Won't have this sprint:**

| Won't | Why deferred |
|---|---|
| Full vector RAG / embeddings store | Corpus is a handful of static episodes — likely doesn't justify retrieval complexity in 2 days (pending final RAG decision) |
| Automated LinkedIn publishing (API posting) | Manual copy-paste of generated output is sufficient for MVP demo |
| Full monitor/brief/iterate pipeline stages | MVP scope is document → generate; optional stages only if time permits |

## 5. WBS (→ Trello cards)

```
1. Structure & board
   1.1 Write project_structure.md
   1.2 Create Trello lists + WIP + DoD
   1.3 Write agents.md
   1.4 Create cards from this WBS
2. Knowledge bases
   2.1 Primary: 3-5 episode transcripts/notes + style guide (markdown)
   2.2 Secondary: PM/product industry trend docs (markdown)
3. Ingest & context
   3.1 Markdown loader
   3.2 Context → prompts (or retrieval, if RAG chosen)
4. Generate & differentiate
   4.1 LLM client + .env
   4.2 Prompt templates (≥2)
   4.3 End-to-end pipeline command
   4.4 Uniqueness comparison artifact
5. Close
   5.1 Finalize rag_decision.md (+ structure §7)
   5.2 README + demo prep
   5.3 Day 1 / Day 2 board screenshots
```

## 6. Risks (exactly 3)

| Risk | P | I | Strategy | Concrete action |
|---|---|---|---|---|
| API rate limits / cost, shared across 4 people | M | M | Reduction | Use gpt-4o-mini, cap test calls per person, cache sample outputs for repeated demo runs |
| Output reads as generic "AI-slop," not real Lenny content | M | H | Reduction | Run uniqueness test by Day 2 midday; strengthen style-guide + episode-excerpt injection if it fails |
| Attribution/IP concerns using another creator's podcast content | L | M | Mitigation | Every generated post cites the source episode explicitly; use short excerpts/paraphrase only, never long verbatim quotes |

## 7. Bridge to rag_decision.md

*(Draft — finalize after team discussion)*

The Instructor's and end user's shared Quality need — content that's traceably grounded in what Lenny's guests actually said, not generic PM commentary — is what drives our context strategy: we need the LLM to see real episode material directly, not a vague summary. M1 and M3 (ingest KB markdown, call LLM with KB context) can be satisfied by loading 3-5 full or excerpted transcripts directly into the prompt, since that corpus is small and static for the 2-day window — none of our Musts require searching across a large, changing archive. Our working decision is **non-RAG (context injection)**: load selected episode transcripts + style guide directly into the prompt. We'd revisit this if we scaled beyond a handful of episodes or needed to search across Lenny's full back-catalog on demand.

---
*Draft prepared ahead of group brainstorm — confirm project name, stakeholder engagement details, and Won't list with Mudit, Parisa, and Zahra before finalizing.*
