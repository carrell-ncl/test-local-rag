#!/usr/bin/env python3
"""
Build the rag_docs index from existing rag_chunks.

Strategy:
- For each arxiv_id in rag_chunks, create one doc in rag_docs.
- Prefer text from section == "Abstract" (case-insensitive).
- If no explicit abstract chunk exists, fall back to the first chunk(s).
- Embed the abstract text with Ollama and upsert into rag_docs.

Robustness:
- Handles 'source' being either a dict or a string (some pipelines store a URL string).

Usage:
  python -m src.build_rag_docs --chunks-index rag_chunks --docs-index rag_docs --embed-model nomic-embed-text
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional, Tuple

import requests
from opensearchpy import OpenSearch


OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def os_client() -> OpenSearch:
    """Create an OpenSearch client for local (no-auth) OpenSearch."""
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def embed(text: str, model: str) -> list[float]:
    """Embed text using Ollama embeddings API."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def iter_arxiv_ids(client: OpenSearch, chunks_index: str, size: int = 500) -> list[str]:
    """Return all arxiv_ids present in chunks index via terms agg."""
    resp = client.search(
        index=chunks_index,
        body={"size": 0, "aggs": {"papers": {"terms": {"field": "arxiv_id", "size": size}}}},
    )
    buckets = resp.get("aggregations", {}).get("papers", {}).get("buckets", [])
    return [b["key"] for b in buckets]


def get_best_abstract_text(
    client: OpenSearch,
    chunks_index: str,
    arxiv_id: str,
    max_chars: int = 3500,
) -> str:
    """Select an abstract-like text for a paper from its chunks.

    Preference order:
    1) chunks where section matches "Abstract"
    2) fallback: first few chunks for the paper

    Args:
        client: OpenSearch client.
        chunks_index: rag_chunks index name.
        arxiv_id: Paper ID.
        max_chars: Max length of abstract text used for rag_docs.

    Returns:
        Abstract text (best effort).
    """
    # Try to fetch abstract chunks
    resp_abs = client.search(
        index=chunks_index,
        body={
            "size": 5,
            "_source": ["text", "section"],
            "query": {
                "bool": {
                    "filter": [{"term": {"arxiv_id": arxiv_id}}],
                    "must": [{"match": {"section": "Abstract"}}],
                }
            },
        },
    )
    hits_abs = resp_abs.get("hits", {}).get("hits", [])
    if hits_abs:
        txt = "\n\n".join([(h["_source"].get("text") or "").strip() for h in hits_abs]).strip()
        return txt[:max_chars]

    # Fallback: take first few chunks for the paper
    resp_any = client.search(
        index=chunks_index,
        body={
            "size": 5,
            "_source": ["text", "section", "chunk_id", "title", "authors", "categories", "published", "pdf_url", "abs_url", "source"],
            "query": {"term": {"arxiv_id": arxiv_id}},
            "sort": [{"chunk_id.keyword": {"order": "asc", "unmapped_type": "keyword"}}],
        },
    )
    hits_any = resp_any.get("hits", {}).get("hits", [])
    txt = "\n\n".join([(h["_source"].get("text") or "").strip() for h in hits_any]).strip()
    return txt[:max_chars]


def get_metadata_from_any_chunk(client: OpenSearch, chunks_index: str, arxiv_id: str) -> dict[str, Any]:
    """Pull representative metadata fields for a paper from any chunk."""
    resp = client.search(
        index=chunks_index,
        body={
            "size": 1,
            "_source": ["title", "authors", "categories", "published", "pdf_url", "abs_url", "source", "version"],
            "query": {"term": {"arxiv_id": arxiv_id}},
        },
    )
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        return {}
    return hits[0].get("_source", {}) or {}


def parse_source_urls(meta: dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract abs_url and pdf_url from metadata safely.

    Handles:
    - meta["abs_url"] / meta["pdf_url"] directly
    - meta["source"] as dict: {"abs_url": ..., "pdf_url": ...}
    - meta["source"] as string URL (stored by some pipelines)

    Args:
        meta: Metadata dict from a chunk.

    Returns:
        (abs_url, pdf_url)
    """
    abs_url = meta.get("abs_url")
    pdf_url = meta.get("pdf_url")

    src = meta.get("source")

    if isinstance(src, dict):
        abs_url = abs_url or src.get("abs_url")
        pdf_url = pdf_url or src.get("pdf_url")
    elif isinstance(src, str):
        # If a URL string is stored, guess whether it's abs or pdf
        if (abs_url is None) and ("/abs/" in src):
            abs_url = src
        if (pdf_url is None) and ("/pdf/" in src or src.endswith(".pdf")):
            pdf_url = src

    return abs_url, pdf_url


def upsert_doc(client: OpenSearch, docs_index: str, arxiv_id: str, doc: dict[str, Any]) -> None:
    """Upsert one rag_docs document, keyed by arxiv_id."""
    client.index(index=docs_index, id=arxiv_id, body=doc, refresh=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-index", default="rag_chunks")
    ap.add_argument("--docs-index", default="rag_docs")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--max-papers", type=int, default=500)
    args = ap.parse_args()

    client = os_client()

    arxiv_ids = iter_arxiv_ids(client, args.chunks_index, size=args.max_papers)
    if not arxiv_ids:
        print("No arxiv_id values found in chunks index.")
        return

    print(f"Found {len(arxiv_ids)} papers in '{args.chunks_index}'. Building '{args.docs_index}'...")

    for i, aid in enumerate(arxiv_ids, start=1):
        meta = get_metadata_from_any_chunk(client, args.chunks_index, aid)
        abstract = get_best_abstract_text(client, args.chunks_index, aid)

        if not abstract.strip():
            print(f"[SKIP] {aid}: no text found")
            continue

        abs_url, pdf_url = parse_source_urls(meta)

        emb = embed(abstract, args.embed_model)

        doc = {
            "arxiv_id": aid,
            "version": meta.get("version", "") or "",
            "title": meta.get("title", "") or "",
            "authors": meta.get("authors", []) or [],
            "categories": meta.get("categories", []) or [],
            "published": meta.get("published", None),
            "abs_url": abs_url,
            "pdf_url": pdf_url,
            "abstract": abstract,
            "embedding": emb,
        }

        upsert_doc(client, args.docs_index, aid, doc)
        print(f"[OK] ({i}/{len(arxiv_ids)}) Upserted rag_docs: {aid}")

    print("[DONE] rag_docs build complete.")


if __name__ == "__main__":
    main()
