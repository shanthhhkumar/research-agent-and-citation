"""
research_agent.py
------------------
Research Agent (with Citations) — single-file version.

Takes a question and a folder of source documents, retrieves the most
relevant passages with TF-IDF, asks Claude to synthesize a cited answer
from ONLY those passages, and prints/saves a structured result that shows
exactly which source passages backed each claim.

Design summary (see README.md for full tradeoff notes):
- Retrieval is real computation (TF-IDF + cosine similarity via
  scikit-learn), not an LLM guess. If nothing scores above the similarity
  floor, the agent reports "not in sources" WITHOUT calling the model.
- Synthesis is the LLM's only job, and it's constrained hard: every claim
  must carry a [S#] citation tag, outside knowledge is forbidden, and the
  model must self-report COVERAGE: FULL / PARTIAL / NONE.

Usage
-----
Interactive single question:
    python research_agent.py --sources sources/ --ask "What does the
    balloon operator charge per seat?"

Batch mode over a question set, saving results to JSON:
    python research_agent.py --sources sources/ --questions questions.json \
        --out sample_outputs/results.json

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...

Run `python research_agent.py --help` for all options.
"""

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import anthropic


# ============================================================================
# Retrieval: chunking + TF-IDF ranking
# ============================================================================

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


# ============================================================================
# Synthesis: Anthropic API call with strict citation instructions
# ============================================================================

DEFAULT_MODEL = os.environ.get("RESEARCH_AGENT_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a careful research assistant. You answer questions \
using ONLY the numbered source passages provided to you in the user message. \
You are not allowed to use outside knowledge, even if you are confident it is \
correct.

Rules:
1. Every factual claim in your answer must be immediately followed by a \
citation tag referencing the passage(s) it came from, in the form [S#] \
where # is the passage number given in the prompt (e.g. "Ticket prices \
start around $450,000 [S2].").
2. If a claim draws on more than one passage, cite all of them, e.g. [S2][S5].
3. Do not invent, guess, or extrapolate numbers, dates, or facts that are not \
stated in the passages, even if the question asks for something more \
precise than the passages actually say.
4. If the passages do not contain enough information to answer the question \
(fully or partially), you must clearly say so. If they answer PART of the \
question, answer that part with citations and explicitly state which part \
is not covered.
5. If the passages disagree with each other, note the disagreement rather \
than silently picking one.
6. Keep the answer concise and directly responsive to the question. Do not \
pad with generic commentary.

Respond in this exact format:

ANSWER: <your cited answer, or a clear statement that the sources do not \
contain the answer>
COVERAGE: <one of: FULL, PARTIAL, NONE -- how much of the question the \
provided passages actually answer>
"""


def build_user_prompt(question: str, passages: list) -> str:
    """passages: list of (label, chunk) tuples, label like 'S1'."""
    lines = [f"Question: {question}", "", "Source passages:"]
    for label, chunk in passages:
        lines.append(f"\n[{label}] (from {chunk.source})\n{chunk.text}")
    lines.append(
        "\nAnswer the question using only the passages above, following the "
        "citation rules in your instructions."
    )
    return "\n".join(lines)


def synthesize(question: str, passages: list, model: str = DEFAULT_MODEL) -> str:
    """Call Claude to synthesize a cited answer from retrieved passages.
    Raises anthropic.APIError subclasses on failure -- callers should
    handle/report these rather than silently swallowing them."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    user_prompt = build_user_prompt(question, passages)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ============================================================================
# Agent orchestration + CLI
# ============================================================================

def answer_question(retriever: Retriever, question: str, top_k: int, model: str) -> dict:
    retrieved = retriever.retrieve(question, top_k=top_k)

    if not retrieved:
        # Real computation says nothing is relevant -- don't even call the
        # model, just report it directly. This is cheaper and more honest
        # than asking the LLM to judge relevance itself.
        return {
            "question": question,
            "answer": "The provided sources do not contain information relevant to this question.",
            "coverage": "NONE",
            "citations": {},
            "retrieved_passages": [],
        }

    labels = [f"S{i+1}" for i in range(len(retrieved))]
    passages = list(zip(labels, [c for c, _score in retrieved]))

    raw = synthesize(question, passages, model=model)
    answer_text, coverage = _parse_response(raw)

    citation_map = {
        label: {"source": chunk.source, "chunk_id": chunk.id, "score": round(score, 4)}
        for label, (chunk, score) in zip(labels, retrieved)
    }

    return {
        "question": question,
        "answer": answer_text,
        "coverage": coverage,
        "citations": citation_map,
        "retrieved_passages": [
            {"label": label, "chunk_id": chunk.id, "source": chunk.source, "score": round(score, 4)}
            for label, (chunk, score) in zip(labels, retrieved)
        ],
    }


def _parse_response(raw: str) -> tuple:
    """Pull ANSWER and COVERAGE out of the model's structured response.
    Falls back gracefully if the model didn't follow the format exactly."""
    answer, coverage = raw.strip(), "UNKNOWN"
    if "ANSWER:" in raw:
        after_answer = raw.split("ANSWER:", 1)[1]
        if "COVERAGE:" in after_answer:
            answer, cov_part = after_answer.split("COVERAGE:", 1)
            coverage = cov_part.strip().splitlines()[0].strip()
        else:
            answer = after_answer
        answer = answer.strip()
    return answer, coverage


def run_batch(retriever: Retriever, questions: list, top_k: int, model: str) -> list:
    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q}", file=sys.stderr)
        try:
            results.append(answer_question(retriever, q, top_k, model))
        except Exception as e:
            results.append({"question": q, "error": str(e)})
    return results


def _print_result(result: dict):
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    print(f"\n{result['answer']}")
    print(f"\nCoverage: {result['coverage']}")
    if result["citations"]:
        print("Sources cited:")
        for label, info in result["citations"].items():
            print(f"  [{label}] {info['chunk_id']} (similarity={info['score']})")


def _save(results: list, path: str):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(results)} result(s) to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Research Agent (with Citations)")
    parser.add_argument("--sources", default="sources", help="Folder of .txt/.pdf source documents")
    parser.add_argument("--ask", help="Ask a single question interactively")
    parser.add_argument("--questions", help="Path to a JSON file: a list of question strings")
    parser.add_argument("--out", help="Path to save JSON results (batch mode)")
    parser.add_argument("--top-k", type=int, default=6, help="Max passages retrieved per question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model ID to use")
    args = parser.parse_args()

    if not args.ask and not args.questions:
        parser.error("Provide --ask \"question\" or --questions path/to/questions.json")

    print(f"Loading sources from {args.sources}/ ...", file=sys.stderr)
    chunks = load_sources(args.sources)
    print(f"Indexed {len(chunks)} passages from source documents.", file=sys.stderr)
    retriever = Retriever(chunks)

    if args.ask:
        result = answer_question(retriever, args.ask, args.top_k, args.model)
        _print_result(result)
        if args.out:
            _save([result], args.out)
        return

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)
    results = run_batch(retriever, questions, args.top_k, args.model)
    for r in results:
        _print_result(r)
    if args.out:
        _save(results, args.out)


if __name__ == "__main__":
    main()
