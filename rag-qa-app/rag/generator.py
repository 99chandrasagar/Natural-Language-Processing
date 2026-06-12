"""
rag/generator.py
────────────────
Takes retrieved context chunks + a user question and calls an LLM to
produce a grounded, cited answer.

Supports:
  - OpenAI  (GPT-4o, GPT-3.5-turbo)  ← default
  - Anthropic Claude                  ← set LLM_PROVIDER=anthropic
  - Ollama (local LLaMA/Mistral)      ← set LLM_PROVIDER=ollama

Environment variables:
  OPENAI_API_KEY      — required for OpenAI
  ANTHROPIC_API_KEY   — required for Anthropic
  LLM_PROVIDER        — "openai" | "anthropic" | "ollama"  (default: openai)
  LLM_MODEL           — override the default model name
  OLLAMA_BASE_URL     — Ollama server URL (default: http://localhost:11434)
"""

import os
import logging
from typing import List

log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
PROVIDER      = os.getenv("LLM_PROVIDER", "openai").lower()
MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "512"))
TEMPERATURE   = float(os.getenv("LLM_TEMPERATURE", "0.2"))   # low = more factual

_DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "ollama":    "llama3",
}
MODEL = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, "gpt-4o-mini"))


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly \
based on the provided context. 

Rules:
- Only use information present in the context below.
- If the answer is not in the context, say "I don't have enough information \
in the document to answer that."
- Be concise and accurate.
- Do not make up facts."""


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Build a RAG prompt and call the configured LLM.

    Args:
        question:       The user's question.
        context_chunks: Top-k retrieved text chunks from the vector store.

    Returns:
        The LLM's answer string.
    """
    if not context_chunks:
        return "No relevant content was found in the document for your question."

    prompt = _build_prompt(question, context_chunks)
    log.info("Calling %s / %s", PROVIDER, MODEL)

    if PROVIDER == "openai":
        return _call_openai(prompt)
    elif PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    elif PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{PROVIDER}'. Use openai/anthropic/ollama.")


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(question: str, chunks: List[str]) -> str:
    """
    Combine retrieved chunks into a numbered context block.
    Numbered sections help the model reference sources clearly.
    """
    context_block = "\n\n".join(
        f"[{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ── LLM callers ────────────────────────────────────────────────────────────────

def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(prompt: str) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _call_ollama(prompt: str) -> str:
    """
    Call a local Ollama server (e.g. running LLaMA 3 or Mistral).
    Start Ollama with: ollama run llama3
    """
    import requests

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
    }
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["response"].strip()
