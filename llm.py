"""
llm.py
------
Thin wrapper around the Anthropic Messages API for the synthesis step.
Retrieval (see retriever.py) already did the real computation of "which
passages are relevant"; the model's only job here is to read those
passages and write a grounded, cited answer -- or say the sources don't
cover the question.
"""

import os

import anthropic

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
