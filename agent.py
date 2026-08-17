"""
agent.py
--------
Research Agent (with Citations)

Takes a question and a folder of source documents, retrieves the most
relevant passages with TF-IDF, asks Claude to synthesize a cited answer
from ONLY those passages, and prints/saves a structured result that shows
exactly which source passages backed each claim.

Usage
-----
Interactive single question:
    python agent.py --sources sources/ --ask "What does the balloon
    operator charge per seat?"

Batch mode over a question set, saving results to JSON:
    python agent.py --sources sources/ --questions questions.json \
        --out sample_outputs/results.json

Run `python agent.py --help` for all options.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from retriever import Retriever, load_sources
from llm import synthesize, DEFAULT_MODEL


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
            "citations": [],
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


if __name__ == "__main__":
    main()
