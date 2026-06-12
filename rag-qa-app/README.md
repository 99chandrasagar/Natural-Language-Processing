# RAG Document Q&A

A production-style **Retrieval-Augmented Generation** pipeline built with FastAPI, FAISS, and sentence-transformers. Upload any PDF or text file and ask questions — answers are grounded in your document's content, not hallucinated.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐    embed query    ┌──────────────────────┐
│  FastAPI /ask   │ ────────────────► │  FAISS Index         │
│  endpoint       │                   │  (cosine similarity) │
└─────────────────┘ ◄──── top-k ───── └──────────────────────┘
      │                chunks                    ▲
      │                                          │ embed chunks
      │                                          │
      │                               ┌──────────────────────┐
      │                               │  sentence-transformers│
      │                               │  all-MiniLM-L6-v2    │
      │                               └──────────────────────┘
      │                                          ▲
      │                               ┌──────────────────────┐
      │                               │  Ingestion pipeline  │
      │                               │  load → clean → chunk│
      │                               └──────────────────────┘
      │                                          ▲
      │                                     PDF / TXT
      │
      ▼
┌─────────────────┐
│  LLM Generator  │  GPT-4o / Claude / LLaMA (Ollama)
│  prompt + ctx   │
└─────────────────┘
      │
      ▼
   Answer
```

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS (IndexFlatIP, cosine similarity) |
| Chunking | Recursive character splitting (custom) |
| LLM | OpenAI GPT-4o-mini / Anthropic Claude / Ollama |
| Frontend | Vanilla JS SPA |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourname/rag-qa-app
cd rag-qa-app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
# Option A: OpenAI (default)
export OPENAI_API_KEY=sk-...

# Option B: Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=anthropic

# Option C: Local Ollama (no API key needed)
ollama run llama3
export LLM_PROVIDER=ollama
```

### 3. Run

```bash
uvicorn app:app --reload
# Open http://localhost:8000
```

---

## API Reference

### `POST /upload`
Upload and index a document.

**Request:** `multipart/form-data` with `file` field (`.pdf`, `.txt`, `.md`)

**Response:**
```json
{
  "session_id": "uuid",
  "filename": "research_paper.pdf",
  "chunks_indexed": 142,
  "message": "Document indexed successfully."
}
```

---

### `POST /ask`
Ask a question against an indexed document.

**Request:**
```json
{
  "session_id": "uuid",
  "question": "What is the main conclusion of the paper?",
  "top_k": 4
}
```

**Response:**
```json
{
  "answer": "The paper concludes that...",
  "sources": ["chunk 1 text...", "chunk 2 text..."],
  "session_id": "uuid"
}
```

---

### `GET /health`
```json
{ "status": "ok", "active_sessions": 3 }
```

---

## Key Design Decisions

### Chunking Strategy
Uses **recursive character splitting** — tries to split on paragraph breaks first, then sentences, then words. This keeps semantic units together better than fixed-size splitting.

### Embedding Model
`all-MiniLM-L6-v2` is chosen because:
- Runs fully **locally** (no API key / cost)
- 384 dimensions — fast indexing and search
- Strong performance on semantic similarity benchmarks
- ~90MB download, cached after first run

### FAISS Index Type
`IndexFlatIP` (inner product) with L2-normalized vectors = **exact cosine similarity**. For larger corpora (>100k chunks), swap for `IndexIVFFlat` or `IndexHNSWFlat` for approximate search.

### Multi-provider LLM
The `LLM_PROVIDER` env var lets you switch between OpenAI, Anthropic, and local Ollama without changing code — useful for cost/latency tradeoffs.

---

## Project Structure

```
rag-qa-app/
├── app.py              # FastAPI app, endpoints
├── rag/
│   ├── ingestion.py    # Document loading + chunking
│   ├── retriever.py    # Embedding + FAISS index
│   └── generator.py    # Prompt building + LLM call
├── static/
│   └── index.html      # Frontend SPA
├── uploads/            # Saved document files
├── requirements.txt
└── README.md
```

---

## Interview Talking Points

- **Why RAG over fine-tuning?** Fine-tuning is expensive and static. RAG keeps the knowledge base dynamic and updatable without retraining.
- **Chunking tradeoffs:** Smaller chunks = more precise retrieval but lose context. Larger chunks = more context but noisier. Overlap bridges the gap.
- **Why FAISS over a managed vector DB?** For a portfolio project, FAISS is zero-cost and zero-latency. In production I'd use Pinecone or Weaviate for persistence, filtering, and scale.
- **Hallucination mitigation:** The system prompt instructs the LLM to only use provided context and explicitly say "I don't know" when the answer isn't there.
- **Scalability:** Swap `IndexFlatIP` for `IndexIVFFlat`, persist indexes to disk, add Redis session store, put behind an async worker queue.
