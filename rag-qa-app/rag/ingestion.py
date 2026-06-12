"""
rag/ingestion.py
────────────────
Handles document loading and text chunking.

Supported formats: .pdf, .txt, .md

Chunking strategy:
  - Recursive character splitting (respects sentence/paragraph boundaries)
  - Configurable chunk_size and overlap
  - Each chunk is a plain string — ready for embedding
"""

import re
import logging
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)

# ── Chunking config ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 500        # target characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks (preserves context)
MIN_CHUNK_LEN = 50      # discard chunks shorter than this (headers, page numbers)


def ingest_document(path: Path) -> List[str]:
    """
    Load a document and return a list of text chunks.

    Pipeline:
        load → clean → split → filter
    """
    log.info("Ingesting document: %s", path)

    # 1. Load raw text
    raw_text = _load(path)
    log.info("Loaded %d characters", len(raw_text))

    # 2. Clean (remove extra whitespace, fix encoding artifacts)
    cleaned = _clean(raw_text)

    # 3. Split into chunks
    chunks = _recursive_split(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # 4. Filter out noise
    chunks = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LEN]

    log.info("Produced %d chunks", len(chunks))
    return chunks


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Falls back to pypdf if fitz is not installed.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError(
            "Install a PDF library: pip install pymupdf  OR  pip install pypdf"
        )


# ── Cleaning ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalize whitespace and remove common PDF artifacts."""
    # Collapse multiple blank lines → double newline (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove hyphenation at line breaks (common in PDFs)
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Normalize other whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ── Recursive splitter ─────────────────────────────────────────────────────────

def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively split text trying each separator in order.
    Mirrors LangChain's RecursiveCharacterTextSplitter logic.

    Separator priority:
        paragraph → sentence → word → character
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    sep = separators[0]
    remaining_seps = separators[1:]

    # Split on the current separator
    if sep:
        splits = text.split(sep)
    else:
        splits = list(text)

    chunks: List[str] = []
    current = ""

    for split in splits:
        candidate = (current + sep + split).strip() if current else split.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current chunk
            if current:
                chunks.append(current)

            # If this split is itself too long, recurse with a finer separator
            if len(split) > chunk_size and remaining_seps:
                sub_chunks = _recursive_split(split, chunk_size, overlap, remaining_seps)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = split

    if current:
        chunks.append(current)

    # Apply overlap: each chunk includes the tail of the previous one
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = overlapped[-1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks
