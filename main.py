from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from retrieval import DocumentIndex

app = FastAPI(title="RAG Q&A Service", version="1.0.0")

_index = DocumentIndex()

DOCS_DIR = "docs"


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/index", status_code=status.HTTP_200_OK)
def build_index():
    """
    Read all .txt files from the docs/ folder, chunk them, and build an
    in-memory TF-IDF retrieval index.
    """
    try:
        stats = _index.build(DOCS_DIR)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return stats


@app.post("/clear", status_code=status.HTTP_200_OK)
def clear_index():
    """Drop the in-memory index so it can be rebuilt with POST /index."""
    _index.clear()
    return {"message": "Index cleared."}


@app.post("/ask", status_code=status.HTTP_200_OK)
def ask(body: AskRequest):
    """
    Answer a question using only the indexed document chunks.
    Returns the answer, top source references, and a confidence label.
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="'question' must not be empty.")

    if not _index.is_ready():
        raise HTTPException(
            status_code=400,
            detail="Index is not built yet. Call POST /index first.",
        )

    return _index.answer(body.question.strip())
