"""
retriever.py
------------
Loads source documents, splits them into citable chunks, and ranks chunks
against a question using TF-IDF cosine similarity.

Design note: retrieval is done with classic TF-IDF (scikit-learn), not an
LLM call and not an embeddings API. This keeps retrieval deterministic,
free, fast, and fully local -- the LLM is only used for the synthesis step,
which reduces cost and makes it easy to reason about why a given chunk was
retrieved.
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str        # e.g. "safety_and_regulation.txt#3"
    source: str    # e.g. "safety_and_regulation.txt"
    text: str


def _read_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "Reading PDF sources requires pypdf. Install with: pip install pypdf"
        ) from e
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _split_into_chunks(text: str, max_words: int = 180) -> List[str]:
    """Split on blank lines (paragraphs); further split very long
    paragraphs into ~max_words pieces on sentence boundaries so no single
    chunk is too large to cite cleanly."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current: List[str] = []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if current and current_len + sent_len > max_words:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
    return chunks


def load_sources(folder: str) -> List[Chunk]:
    """Load every .txt and .pdf file in `folder` and return a flat list
    of citable Chunks."""
    chunks: List[Chunk] = []
    paths = sorted(glob.glob(os.path.join(folder, "*.txt"))) + sorted(
        glob.glob(os.path.join(folder, "*.pdf"))
    )
    if not paths:
        raise FileNotFoundError(f"No .txt or .pdf source files found in {folder}")

    for path in paths:
        name = os.path.basename(path)
        text = _read_pdf(path) if path.lower().endswith(".pdf") else _read_txt(path)
        for i, chunk_text in enumerate(_split_into_chunks(text), start=1):
            chunks.append(Chunk(id=f"{name}#{i}", source=name, text=chunk_text))
    return chunks


class Retriever:
    """TF-IDF retriever over a fixed set of Chunks."""

    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("Retriever received an empty chunk list")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(
        self, query: str, top_k: int = 6, min_score: float = 0.03
    ) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (Chunk, score) pairs above min_score,
        ranked by cosine similarity to the query. An empty result means
        the sources likely do not cover the question."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunks, sims), key=lambda pair: pair[1], reverse=True)
        return [(c, float(s)) for c, s in ranked if s >= min_score][:top_k]
