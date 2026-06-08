# RAG Q&A Service

A lightweight Retrieval-Augmented Generation (RAG) API that answers questions from a local folder of plain-text documents — no external LLM APIs required.

## How it works

1. `POST /index` reads every `.txt` file in `docs/`, splits each document into individual sentences, and fits a TF-IDF index in memory where every sentence is its own indexed unit.
2. `POST /ask` transforms the question with the same TF-IDF vocabulary, finds the top matching sentences via cosine similarity, and returns only those specific sentences as the answer — not the whole document or chunk.
3. `POST /clear` drops the in-memory index so it can be rebuilt with a fresh call to `POST /index`.

If the best similarity score falls below a threshold the service responds with `"insufficient_context"` rather than hallucinating.

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload
```

The API is now available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/index` | Index all `.txt` files in the `docs/` folder |
| `POST` | `/ask` | Ask a question against the indexed documents |
| `POST` | `/clear` | Clear the in-memory index |

## Quick start

### 1 — Build the index

```bash
curl -X POST http://localhost:8000/index
```

Example response:
```json
{
  "documents_indexed": 10,
  "sentences_indexed": 336,
  "files": [
    "apex_overview.txt",
    "attachments_loot.txt",
    "beginners_tips.txt",
    "legends_classes.txt",
    "maps_guide.txt",
    "movement_mechanics.txt",
    "ranked_mode.txt",
    "ring_and_strategy.txt",
    "team_composition.txt",
    "weapons_guide.txt"
  ]
}
```

### 2 — Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the highest damage weapon in Apex Legend"}'
```

Example response:
```json
{
  "answer": "The Kraber .50-Cal Sniper is a care package-only weapon that deals massive damage per shot and uses unique ammo found only with the weapon. The Wingman uses Heavy Rounds and deals very high damage per shot, making it a favourite for skilled players.",
  "sources": [
    {
      "file": "weapons_guide.txt",
      "sentence": "The Kraber .50-Cal Sniper is a care package-only weapon that deals massive damage per shot and uses unique ammo found only with the weapon.",
      "score": 0.3393
    }
  ],
  "confidence": "answered_from_docs"
}
```

The `sentence` field in each source is the exact sentence from the document that contributed to the answer — nothing more.

### 3 — Insufficient context example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "what is overcooked"}'
```

```json
{
  "answer": "The provided documents do not contain enough information to answer this question confidently.",
  "sources": [],
  "confidence": "insufficient_context"
}
```

### 4 — Clear the index

```bash
curl -X POST http://localhost:8000/clear
```

```json
{ "message": "Index cleared." }
```

After clearing, calling `POST /ask` returns a 400 until `POST /index` is called again.

## Sample questions to try

| Question | Expected source | Confidence |
|---|---|---|
| What is Apex Legends? | apex_overview.txt | `answered_from_docs` |
| What legends are good for beginners? | team_composition.txt | `answered_from_docs` |
| What is the highest damage weapon in Apex Legend? | weapons_guide.txt | `answered_from_docs` |
| What is rank in Apex Legend? | ranked_mode.txt | `answered_from_docs` |
| How does the Evo Shield work? | attachments_loot.txt | `answered_from_docs` |
| Which map has the Zip Rail system? | maps_guide.txt | `answered_from_docs` |
| What is tap strafing? | movement_mechanics.txt | `answered_from_docs` |
| How do I respawn my teammate? | team_composition.txt | `answered_from_docs` |
| What is overcooked? | — | `insufficient_context` |

## Error handling

| Scenario | HTTP status | Detail |
|---|---|---|
| `POST /ask` before `POST /index` | 400 | "Index is not built yet." |
| `POST /ask` after `POST /clear` | 400 | "Index is not built yet." |
| Empty question | 422 | "'question' must not be empty." |
| `docs/` folder missing | 404 | "docs folder not found" |
| `docs/` folder has no `.txt` files | 422 | "No .txt files found" |

## Providing your own documents

Drop any `.txt` files into the `docs/` folder, then call `POST /index` to build or rebuild the index. The previous index is replaced in memory.

---

## Tradeoffs and what I would improve next

### Retrieval granularity

The index is built at the sentence level — each sentence is its own vector. This means retrieval returns only the specific sentences that matched the query rather than large word-window chunks. The tradeoff is that very short sentences produce sparse TF-IDF vectors, which can make scoring noisy for brief or ambiguous sentences. Grouping 2–3 consecutive sentences per unit would balance precision against vector density.

### TF-IDF vs. embeddings

TF-IDF with suffix-stripping is fast, dependency-free, and fully explainable, but it still relies on vocabulary overlap. A semantic embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) would handle paraphrases and synonyms far better ("cost" vs. "price", "cancel" vs. "terminate"). Given the constraint of no paid APIs, this would be the single highest-impact upgrade.

### Confidence threshold

The `MIN_SCORE = 0.13` cutoff was calibrated against the sample documents. A different document set with different vocabulary density may require tuning this value. A more robust approach would be to set the threshold relative to the score distribution (e.g., flag as insufficient if the top score is less than 2× the mean score).

### Answer generation

The current approach is fully extractive — it returns real sentences verbatim from the source documents, so it cannot fabricate information. The tradeoff is that the answer can feel disjointed when relevant sentences come from different parts of a document. A local instruction-following LLM (e.g., via `llama.cpp` or Ollama) could generate a fluent synthesised answer from the retrieved sentences while remaining grounded.

### Persistence

The index lives only in memory and is lost on server restart. The TF-IDF matrix and vocabulary can be serialised to disk with `joblib` (already installed as a scikit-learn dependency) so the server can warm-start without re-indexing.

### Scalability

For thousands of documents, the in-memory numpy cosine search should be replaced with an approximate nearest-neighbour index (FAISS, hnswlib) or a dedicated vector database (Chroma, Qdrant).

---

## Future plans

### Web search fallback for unanswered questions

When the service returns `"insufficient_context"`, the question could not be answered from the indexed documents. The planned improvement is to automatically fall back to a web search tool when this happens, so the user always gets an answer regardless of whether the information exists in the local docs.

**Planned flow:**

```
POST /ask
    │
    ├── Search indexed docs (TF-IDF)
    │       │
    │       ├── score ≥ MIN_SCORE  ──► return answer from docs
    │       │                          confidence: "answered_from_docs"
    │       │
    │       └── score < MIN_SCORE  ──► fallback to web search
    │                                  confidence: "answered_from_web"
    │
    └── return answer with source URL
```

**Planned response shape when web search is used:**

```json
{
  "answer": "Overcooked is a cooperative cooking simulation video game developed by Ghost Town Games, released in 2016.",
  "sources": [
    {
      "url": "https://en.wikipedia.org/wiki/Overcooked",
      "title": "Overcooked - Wikipedia",
      "snippet": "Overcooked is a cooperative cooking simulation video game..."
    }
  ],
  "confidence": "answered_from_web"
}
```

**Implementation approach:**

- Use a free web search API such as [DuckDuckGo Instant Answer API](https://api.duckduckgo.com/) (no API key required) or [SerpAPI](https://serpapi.com/) for richer results.
- The fallback only triggers when `confidence` would be `"insufficient_context"`, keeping doc-based answers as the primary and trusted source.
- Web results will include a `"url"` field in sources so the user can verify the origin.
- A `search_enabled` flag in the request body will allow callers to opt out of web search if they only want answers from the indexed documents.

**Planned request body:**

```json
{
  "question": "what is overcooked",
  "search_enabled": true
}
```
