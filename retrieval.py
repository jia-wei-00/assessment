"""
Core retrieval logic: sentence-level TF-IDF indexing and extractive answer generation.
"""

import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOP_K = 5       # number of sentences to retrieve per query
MIN_SCORE = 0.13  # cosine-similarity floor; below this → "insufficient_context"


# ---------------------------------------------------------------------------
# Minimal stemming analyzer
# ---------------------------------------------------------------------------

_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "up", "about", "into",
    "through", "during", "before", "after", "and", "but", "or", "nor",
    "not", "no", "it", "its", "this", "that", "these", "those", "than",
    "then", "when", "where", "who", "what", "which", "how", "all", "each",
    "every", "some", "such", "only", "also", "just", "if", "any", "there",
    "their", "our", "your", "his", "her", "they", "them", "we", "you",
    "he", "she", "my", "i", "so", "as", "both", "either", "very", "per",
    "new",
})

_SUFFIXES: Tuple[str, ...] = (
    "ations", "ation",
    "ating",  "ated",
    "ates",   "ate",
    "ions",   "ing",   "ion",
    "ment",   "ness",
    "iers",   "ies",
    "ers",    "ed",    "er",   "es",
    "ly",
)

_SAFE_PRESUFFIX: FrozenSet[str] = frozenset("ntrlkpdfgbm")


def _stem(word: str) -> str:
    for sfx in _SUFFIXES:
        if word.endswith(sfx) and len(word) - len(sfx) >= 5:
            return word[: -len(sfx)]
    if len(word) >= 5 and word.endswith("s") and word[-2] in _SAFE_PRESUFFIX:
        return word[:-1]
    return word


def _analyze(text: str) -> List[str]:
    tokens = [
        _stem(tok)
        for tok in re.findall(r"[a-z]+", text.lower())
        if tok not in _STOP_WORDS and len(tok) > 2
    ]
    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split a document into individual sentences, dropping very short ones."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 30]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class DocumentIndex:
    def __init__(self) -> None:
        self._sentences: List[str] = []
        self._sources: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    def clear(self) -> None:
        self._sentences = []
        self._sources = []
        self._vectorizer = None
        self._matrix = None
        self._ready = False

    def build(self, docs_dir: str) -> Dict[str, Any]:
        """
        Read all .txt files from *docs_dir*, split them into sentences, and
        fit a TF-IDF index where every sentence is its own indexed unit.
        """
        path = Path(docs_dir)
        if not path.exists():
            raise FileNotFoundError(f"docs folder not found: '{docs_dir}'")

        txt_files = sorted(path.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in '{docs_dir}'")

        sentences, sources = [], []
        for fp in txt_files:
            text = fp.read_text(encoding="utf-8")
            file_sentences = _split_sentences(text)
            sentences.extend(file_sentences)
            sources.extend([fp.name] * len(file_sentences))

        vectorizer = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True, min_df=1)
        matrix = vectorizer.fit_transform(sentences)

        self._sentences = sentences
        self._sources = sources
        self._vectorizer = vectorizer
        self._matrix = matrix
        self._ready = True

        return {
            "documents_indexed": len(txt_files),
            "sentences_indexed": len(sentences),
            "files": [fp.name for fp in txt_files],
        }

    def answer(self, question: str) -> Dict[str, Any]:
        """Retrieve the most relevant individual sentences and return them as the answer."""
        retrieved = self._retrieve(question)

        best_score = retrieved[0][2] if retrieved else 0.0
        if best_score < MIN_SCORE:
            return {
                "answer": (
                    "The provided documents do not contain enough information "
                    "to answer this question confidently."
                ),
                "sources": [],
                "confidence": "insufficient_context",
            }

        matched = [(sent, src, score) for sent, src, score in retrieved if score >= MIN_SCORE]

        # Answer is only the specific matched sentences — nothing more
        answer_text = " ".join(sent for sent, _, _ in matched)

        sources = [
            {"file": src, "sentence": sent, "score": round(score, 4)}
            for sent, src, score in matched
        ]

        return {
            "answer": answer_text,
            "sources": sources,
            "confidence": "answered_from_docs",
        }

    def _retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, str, float]]:
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self._sentences[i], self._sources[i], float(scores[i]))
            for i in top_indices
        ]
