# Research Agent (with Citations)

**One sentence:** this agent takes a question and a folder of source documents, and produces an answer where every claim is tagged with the exact source passage it came from — or a clear statement that the sources don't cover the question.

Built for the Rooman AI Challenge — Category 4, Research Agent (Advanced).

---

## How it works (30 seconds)

```
question ──► TF-IDF retrieval over chunked sources ──► top-k relevant passages
                                                              │
                                                              ▼
                                          Claude, told to answer ONLY from
                                          those passages and cite [S1][S2]...
                                                              │
                                                              ▼
                                     answer + coverage label (FULL/PARTIAL/NONE)
                                          + which exact chunks were cited
```

Retrieval and synthesis are deliberately separate steps:

- **Retrieval is real computation, not an LLM guess.** Documents are split into paragraph-level chunks, each given a stable ID (`filename.txt#3`), and ranked against the question with TF-IDF + cosine similarity (scikit-learn). If nothing scores above the similarity threshold, the agent reports "not in sources" **without even calling the model** — cheaper and more honest than asking an LLM to judge its own relevance.
- **Synthesis is the LLM's only job**, and it's constrained hard: the system prompt requires a citation tag after every claim, forbids outside knowledge, and requires an explicit `COVERAGE: FULL / PARTIAL / NONE` verdict so a reviewer (or a downstream system) can programmatically tell when the answer is incomplete.

---

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd research_agent
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your Anthropic API key
export $(cat .env | xargs)  # or use python-dotenv / your shell's env loading
```

Get an Anthropic API key at <https://console.anthropic.com/> if you don't have one.

## Running it

**Ask one question interactively:**

```bash
python agent.py --sources sources --ask "How much does a suborbital seat cost with Aurelia Spaceworks compared to Halcyon Voyages?"
```

**Batch mode over a question set (used to produce `sample_outputs/`):**

```bash
python agent.py --sources sources --questions questions.json --out sample_outputs/results.json
```

**Useful flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--sources` | `sources` | Folder of `.txt`/`.pdf` source documents |
| `--top-k` | `6` | Max passages retrieved per question |
| `--model` | `claude-sonnet-5` | Claude model ID (override with `--model claude-opus-5` for harder synthesis) |
| `--out` | — | Save full JSON results (question, answer, coverage, citations, retrieved passages) |

Drop your own `.txt` or `.pdf` files into `sources/` (or point `--sources` at a different folder) to run this against a different corpus — nothing else needs to change.

---

## What's included (deliverables)

- **`questions.json`** — 10 sample questions, deliberately mixed:
  - Directly answerable from one source (e.g. Aurelia's 2014 accident cause)
  - Answerable by combining multiple sources (e.g. balloon vs. rocket pricing)
  - Explicitly *not* in the sources (e.g. "projected market size in 2035" — the sources give qualitative forecasts, not a 2035 figure)
- **`sources/`** — 5 original source documents (~30 chunks total) covering company overviews, pricing, safety/regulation, environmental impact, and passenger training for a fictional commercial space-tourism industry. Deliberately includes gaps and cross-document facts so citation and "insufficient information" behavior are both exercised.
- **`sample_outputs/results.json`** — will be populated by running the batch command above with a real API key (see *Testing notes* below for why it isn't pre-populated with live model output).
- **This README** — setup, usage, and the tradeoff notes below.

---

## Design choices & tradeoffs

**TF-IDF over embeddings.** For a corpus this size (dozens of chunks), TF-IDF retrieval is essentially free, needs no extra API calls, and is trivial to reason about (a reviewer can see exactly why a passage scored the way it did). At real-world scale (thousands+ of documents) I'd switch to embedding-based retrieval (e.g. Voyage or OpenAI embeddings + a vector index) since TF-IDF misses synonyms and paraphrases that don't share surface words with the question.

**Paragraph-level chunking, not sentence-level.** Paragraphs keep enough context for the model to cite accurately without pulling in a whole document. Very long paragraphs are further split on sentence boundaries at ~180 words so no single citation is unreasonably large. Tradeoff: a fact split across two paragraphs (e.g. a claim in one paragraph, a caveat in the next) can occasionally be retrieved separately rather than together — mitigated by retrieving the top 6 chunks rather than just 1.

**Score-based short-circuit before calling the LLM.** If TF-IDF returns zero passages above the similarity floor, the agent skips the API call entirely and reports no-coverage directly. This only catches questions that are *lexically* unrelated to the corpus (e.g. asking about nitrogen's boiling point). For questions that are topically related but ask for a specific fact the sources don't contain (e.g. "market size in 2035"), retrieval still returns plausible-looking passages, so declaring `COVERAGE: PARTIAL/NONE` is left to the LLM under the citation-discipline system prompt. I tested this explicitly (see `questions.json` items on 2035 market size and Meridian Deep Space lunar passengers) — it's the main place where synthesis quality, not retrieval, does the honesty work, and it's the part I'd stress-test hardest with more time.

**Structured `ANSWER:` / `COVERAGE:` response format**, parsed in `agent.py`, rather than asking for JSON directly. In testing, forcing strict JSON out of a model while also demanding inline citation tags tends to produce malformed brackets or truncated answers; a simple two-field text format is easier for the model to produce reliably and still easy to parse.

**One LLM call per question, not one call for the whole batch.** Keeps each question's context window focused on only its own retrieved passages (no cross-question contamination) and keeps failures isolated — one bad question doesn't take down the whole batch (see the `try/except` per-question in `run_batch`).

### What I'd improve with more time

- Add an optional live web-search fallback (`--search`) for questions the local corpus doesn't cover at all, rather than only ever saying "not in sources."
- Add a lightweight eval harness that checks whether every `[S#]` tag in an answer corresponds to a real citation index, and flags "orphan" claims with no citation at all — right now that's checked only by prompt instruction, not verified in code.
- Swap TF-IDF for embedding retrieval + re-ranking once the corpus is large enough that lexical overlap stops being a good enough relevance signal.
- Cache retrieval + synthesis results per (question, source-folder hash) so re-running a batch after a small doc edit doesn't reprocess everything.

---

## Testing notes (honesty section)

This project was built and tested in a sandboxed environment with **no outbound network access**, so the Anthropic API itself could not be called live from that environment. What *was* verified end-to-end there:

- Source loading, chunking, and TF-IDF retrieval (`retriever.py`) — run directly against all 5 sample source docs and all 10 sample questions; retrieval scores and chunk IDs shown above are real output, not fabricated.
- The full CLI flow (`agent.py`) — argument parsing, retrieval, response parsing (`ANSWER:`/`COVERAGE:` extraction), citation-map construction, and JSON output — verified using a stub in place of the `anthropic` client to isolate agent logic from the network call.
- The no-coverage short-circuit path — confirmed it returns `COVERAGE: NONE` without invoking the model at all for an out-of-corpus question.

**Not yet verified in that environment:** an actual live call to Claude, since that requires network access and an API key. The prompt in `llm.py` and the parsing logic in `agent.py` are written and unit-tested against a stub response in the same `ANSWER:`/`COVERAGE:` shape the real model is instructed to produce — running `python agent.py --sources sources --questions questions.json --out sample_outputs/results.json` with a real `ANTHROPIC_API_KEY` set is the one remaining step, and it should work without code changes. I'm flagging this rather than pasting in fabricated "sample" model output.
