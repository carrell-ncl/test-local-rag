#!/usr/bin/env python3
"""
Redis-backed ingestion worker for arXiv PDFs.

Queue item (JSON) should look like:
{
  "arxiv_id": "2401.12345",
  "version": "v2",
  "pdf_url": "https://arxiv.org/pdf/2401.12345v2.pdf",
  "title": "...",
  "authors": ["A. Smith", "B. Jones"],
  "categories": ["cs.LG", "cs.AI"],
  "published": "2024-01-22T00:00:00Z",
  "metadata": {"anything": "extra"}
}

If you want to “focus on a subject” (e.g., graph neural networks),
you can enqueue only those categories/queries upstream, or filter here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import redis
import requests
from opensearchpy import OpenSearch

from src.parse_pdf import extract_arxiv_text


# -------------------------
# Config
# -------------------------

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "ingest:arxiv")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_INDEX = os.getenv("OS_INDEX", "rag_chunks")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))      # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # chars

# Optional: enforce subject focus at ingest-time
# Example: set SUBJECT_FILTER_REGEX="graph neural network|gnn|message passing"
SUBJECT_FILTER_REGEX = os.getenv("SUBJECT_FILTER_REGEX", "").strip()


# -------------------------
# Helpers
# -------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def redis_client() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def os_client() -> OpenSearch:
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def embed(text: str, model: str = OLLAMA_EMBED_MODEL) -> list[float]:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def download_pdf(pdf_url: str) -> str:
    r = requests.get(pdf_url, timeout=120)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".pdf")
    os.write(fd, r.content)
    os.close(fd)
    return path


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# Very lightweight section splitter for papers (works decently on arXiv PDFs).
# You can later replace with a better parser (Grobid, etc.) if desired.
SECTION_RE = re.compile(
    r"\n\s*(abstract|introduction|related work|background|method|methods|approach|"
    r"experiments|results|discussion|conclusion|conclusions)\s*\n",
    re.IGNORECASE,
)


def chunk_with_sections(text: str) -> list[tuple[str, int, str]]:
    """
    Returns: list of (section_name, local_chunk_index, chunk_text)
    """
    # If no matches, fall back to generic chunking
    if not SECTION_RE.search(text):
        return [("Body", i, c) for i, c in enumerate(chunk_text(text))]

    parts = SECTION_RE.split(text)
    # split() gives: [pre, section1, body1, section2, body2, ...]
    # "pre" is whatever comes before first matched header; treat as "Front"
    out: list[tuple[str, int, str]] = []

    front = parts[0].strip()
    if front:
        for i, c in enumerate(chunk_text(front)):
            out.append(("Front", i, c))

    for i in range(1, len(parts), 2):
        section = parts[i].strip().title()
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        if not body:
            continue
        for j, c in enumerate(chunk_text(body)):
            out.append((section, j, c))

    return out


def passes_subject_filter(title: str, fulltext: str) -> bool:
    """
    Optional ingest-time filter to focus corpus on a subject.
    If SUBJECT_FILTER_REGEX is empty, always passes.
    """
    if not SUBJECT_FILTER_REGEX:
        return True
    rx = re.compile(SUBJECT_FILTER_REGEX, re.IGNORECASE)
    return bool(rx.search(title or "") or rx.search(fulltext or ""))


@dataclass
class ArxivJob:
    arxiv_id: str
    pdf_url: str
    version: str = ""
    title: str = ""
    authors: list[str] | None = None
    categories: list[str] | None = None
    published: str = ""  # ISO string preferred
    metadata: dict[str, Any] | None = None


def parse_job(raw: str) -> ArxivJob:
    data = json.loads(raw)
    if "arxiv_id" not in data or "pdf_url" not in data:
        raise ValueError("Job must include 'arxiv_id' and 'pdf_url'.")

    return ArxivJob(
        arxiv_id=data["arxiv_id"],
        pdf_url=data["pdf_url"],
        version=data.get("version", ""),
        title=data.get("title", "") or "",
        authors=data.get("authors") or [],
        categories=data.get("categories") or [],
        published=data.get("published", "") or "",
        metadata=data.get("metadata") or {},
    )


def index_chunks(client: OpenSearch, index_name: str, job: ArxivJob, chunks: list[tuple[str, int, str]]) -> None:
    doc_id = job.arxiv_id or sha1(job.pdf_url)

    for section, local_idx, chunk in chunks:
        vec = embed(chunk)

        doc = {
            "doc_id": doc_id,
            "chunk_id": f"{section}:{local_idx}",
            "source": job.pdf_url,
            "arxiv_id": job.arxiv_id,
            "version": job.version,
            "title": job.title,
            "authors": job.authors or [],
            "categories": job.categories or [],
            "published": job.published or None,
            "section": section,
            "text": chunk,
            "created_at": now_iso(),
            "embedding": vec,
            "metadata": job.metadata or {},
        }

        _id = f"{doc_id}:{section}:{local_idx}"
        client.index(index=index_name, id=_id, body=doc)

    client.indices.refresh(index=index_name)


def process_one(job: ArxivJob, index_name: str) -> None:
    client = os_client()

    tmp_pdf = download_pdf(job.pdf_url)
    try:
        text = extract_arxiv_text(tmp_pdf, strip_references=True)

        if not text.strip():
            print(f"[WARN] No text extracted for {job.arxiv_id} ({job.pdf_url})")
            return

        if not passes_subject_filter(job.title, text):
            print(f"[SKIP] Subject filter rejected {job.arxiv_id} ({job.title})")
            return

        chunks = chunk_with_sections(text)
        if not chunks:
            print(f"[WARN] No chunks produced for {job.arxiv_id}")
            return

        print(f"[INFO] {job.arxiv_id} -> {len(chunks)} chunks")
        index_chunks(client, index_name, job, chunks)
        print(f"[OK] Indexed {len(chunks)} chunks for {job.arxiv_id}")

    finally:
        try:
            os.remove(tmp_pdf)
        except Exception:
            pass


def run_worker(index_name: str) -> None:
    r = redis_client()
    print(f"[INFO] Worker started. Queue={REDIS_QUEUE} Index={index_name}")

    while True:
        item = r.blpop(REDIS_QUEUE, timeout=5)
        if not item:
            continue

        _, raw = item
        try:
            job = parse_job(raw)
            process_one(job, index_name=index_name)
        except Exception as e:
            print(f"[ERROR] Failed job: {raw}\n  -> {e}")


def enqueue_job(job: dict[str, Any]) -> None:
    r = redis_client()
    r.rpush(REDIS_QUEUE, json.dumps(job))
    print(f"[INFO] Enqueued {job.get('arxiv_id')} -> {job.get('pdf_url')}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=OS_INDEX)
    ap.add_argument("--enqueue", default=None, help="JSON string job to enqueue (then exit).")
    args = ap.parse_args()

    if args.enqueue:
        enqueue_job(json.loads(args.enqueue))
    else:
        run_worker(args.index)
