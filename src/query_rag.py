#!/usr/bin/env python3
"""
Hybrid RAG query script (OpenSearch + Ollama).

Implements a robust hybrid retrieval strategy:
1) BM25 keyword retrieval (multi_match over title + text)
2) Vector retrieval (kNN over embedding)
3) Fuse rankings using Reciprocal Rank Fusion (RRF)
4) Balance context across papers (cap chunks per arxiv_id)
5) Ask an Ollama chat model using retrieved context

Run:
  python -m src.query_rag --question "What is the main contribution?"

Example (GNN-focused):
  python -m src.query_rag \
    --question "How does the paper compute attention coefficients?" \
    --categories cs.LG,cs.AI,stat.ML \
    --k 8 --candidates 40 --max-docs 3 --max-chunks-per-doc 3 \
    --show-top-papers --show-context

Notes:
- Requires Ollama embedding model (e.g. nomic-embed-text) and chat model (e.g. llama3.1).
- Works with OpenSearch 2.14 using knn_vector mapping.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import requests
from opensearchpy import OpenSearch


# -------------------------
# Config (override via env)
# -------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_INDEX = os.getenv("OS_INDEX", "rag_chunks")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


# -------------------------
# Clients
# -------------------------

def os_client() -> OpenSearch:
    """Create an OpenSearch client for a local (no-auth) instance."""
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def embed(text: str, model: str = OLLAMA_EMBED_MODEL) -> list[float]:
    """Embed text using Ollama embeddings API.

    Args:
        text: Text to embed.
        model: Ollama embedding model name.

    Returns:
        Embedding vector as a list of floats.
    """
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def ask_llm(question: str, context: str, model: str = OLLAMA_CHAT_MODEL) -> str:
    """Ask an Ollama chat model using retrieved context.

    Args:
        question: User question.
        context: Retrieved context.
        model: Ollama chat model name.

    Returns:
        The model's answer text.
    """
    system = (
        "You are a helpful research assistant. Answer using ONLY the provided context. "
        "Cite chunk numbers like [1], [2] when making claims. "
        "If the context is insufficient, say what is missing and what you'd need."
    )

    user = textwrap.dedent(
        f"""
        Context:
        {context}

        Question:
        {question}
        """
    ).strip()

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


# -------------------------
# Retrieval + Fusion
# -------------------------

def _build_filters(
    *,
    arxiv_id: Optional[str],
    categories: list[str],
    published_after: Optional[str],
    published_before: Optional[str],
) -> list[dict[str, Any]]:
    """Build OpenSearch filter clauses."""
    filters: list[dict[str, Any]] = []

    if arxiv_id:
        filters.append({"term": {"arxiv_id": arxiv_id}})

    if categories:
        # categories is mapped as keyword, so terms filter is appropriate
        filters.append({"terms": {"categories": categories}})

    if published_after or published_before:
        rng: dict[str, Any] = {}
        if published_after:
            rng["gte"] = published_after
        if published_before:
            rng["lte"] = published_before
        filters.append({"range": {"published": rng}})

    return filters


def bm25_search(
    client: OpenSearch,
    *,
    index_name: str,
    question: str,
    size: int,
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run BM25 keyword retrieval over title + text.

    Args:
        client: OpenSearch client.
        index_name: Index name.
        question: User question.
        size: Number of hits.
        filters: Filter clauses.

    Returns:
        List of OpenSearch hits.
    """
    body = {
        "size": size,
        "_source": ["arxiv_id", "title", "section", "chunk_id", "text", "source", "published", "categories"],
        "query": {
            "bool": {
                "filter": filters,
                "must": [
                    {
                        "multi_match": {
                            "query": question,
                            "fields": ["title^3", "text"],
                            "type": "best_fields",
                            "operator": "or",
                        }
                    }
                ],
            }
        },
    }
    resp = client.search(index=index_name, body=body)
    return resp.get("hits", {}).get("hits", [])


def knn_search(
    client: OpenSearch,
    *,
    index_name: str,
    qvec: list[float],
    size: int,
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run vector retrieval using kNN over embedding.

    Args:
        client: OpenSearch client.
        index_name: Index name.
        qvec: Query embedding vector.
        size: Number of hits.
        filters: Filter clauses.

    Returns:
        List of OpenSearch hits.
    """
    # Wrap knn inside bool so we can apply filters consistently.
    body = {
        "size": size,
        "_source": ["arxiv_id", "title", "section", "chunk_id", "text", "source", "published", "categories"],
        "query": {
            "bool": {
                "filter": filters,
                "must": [
                    {
                        "knn": {
                            "embedding": {
                                "vector": qvec,
                                "k": size,
                            }
                        }
                    }
                ],
            }
        },
    }
    resp = client.search(index=index_name, body=body)
    return resp.get("hits", {}).get("hits", [])


def rrf_fuse(
    bm25_hits: list[dict[str, Any]],
    knn_hits: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        bm25_hits: Hits from BM25 search (ranked).
        knn_hits: Hits from kNN search (ranked).
        rrf_k: RRF constant (typical values 20–100; 60 is common).
        limit: Max fused hits to return.

    Returns:
        List of hits sorted by fused RRF score (added to hit["_rrf_score"]).
    """
    # Map doc _id -> hit
    by_id: dict[str, dict[str, Any]] = {}
    scores: defaultdict[str, float] = defaultdict(float)

    def add_list(hits: list[dict[str, Any]]) -> None:
        for rank, h in enumerate(hits, start=1):
            _id = h.get("_id")
            if not _id:
                continue
            if _id not in by_id:
                by_id[_id] = h
            scores[_id] += 1.0 / (rrf_k + rank)

    add_list(bm25_hits)
    add_list(knn_hits)

    fused = []
    for _id, h in by_id.items():
        h["_rrf_score"] = float(scores.get(_id, 0.0))
        fused.append(h)

    fused.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return fused[:limit]


def balance_hits_by_doc(
    hits: list[dict[str, Any]],
    *,
    max_docs: int,
    max_chunks_per_doc: int,
) -> list[dict[str, Any]]:
    """Balance context so no single paper dominates.

    Strategy:
    - Group by arxiv_id
    - Rank papers by their best hit score (RRF score)
    - Take up to `max_docs` papers
    - Take up to `max_chunks_per_doc` chunks per paper

    Args:
        hits: Fused hits.
        max_docs: Max distinct papers to include.
        max_chunks_per_doc: Max chunks per paper.

    Returns:
        A balanced list of hits (still sorted by fused score).
    """
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for h in hits:
        src = h.get("_source", {})
        doc_id = src.get("arxiv_id") or "unknown"
        by_doc[doc_id].append(h)

    # Sort within each doc by fused score
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)

    # Rank docs by best score
    ranked_docs = sorted(
        by_doc.items(),
        key=lambda kv: kv[1][0].get("_rrf_score", 0.0),
        reverse=True,
    )

    chosen: list[dict[str, Any]] = []
    for doc_id, doc_hits in ranked_docs[:max_docs]:
        chosen.extend(doc_hits[:max_chunks_per_doc])

    chosen.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return chosen


def summarize_top_papers(hits: list[dict[str, Any]], top_n: int = 5) -> list[str]:
    """Create a human-readable summary of top candidate papers.

    Args:
        hits: Hits (preferably fused, before balancing).
        top_n: Number of papers to summarize.

    Returns:
        List of summary lines.
    """
    # Compute per-paper best score + count
    stats: dict[str, dict[str, Any]] = {}
    for h in hits:
        src = h.get("_source", {})
        aid = src.get("arxiv_id") or "unknown"
        title = src.get("title") or ""
        score = h.get("_rrf_score", 0.0)

        if aid not in stats:
            stats[aid] = {"best": score, "count": 1, "title": title}
        else:
            stats[aid]["count"] += 1
            if score > stats[aid]["best"]:
                stats[aid]["best"] = score
            # keep the first non-empty title
            if not stats[aid]["title"] and title:
                stats[aid]["title"] = title

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["best"], reverse=True)[:top_n]
    lines = []
    for i, (aid, s) in enumerate(ranked, start=1):
        lines.append(f"{i}) {aid}  (best={s['best']:.4f}, hits={s['count']})  {s['title']}")
    return lines


def build_context(hits: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    """Build a context string from retrieved chunks.

    Args:
        hits: Selected hits (post balancing).
        max_chars: Max size of context to send to the LLM.

    Returns:
        Context string.
    """
    parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        title = src.get("title", "")
        arxiv_id = src.get("arxiv_id", "")
        section = src.get("section", "")
        chunk_id = src.get("chunk_id", "")
        text = (src.get("text", "") or "").strip()

        parts.append(
            f"[{i}] arXiv: {arxiv_id}\n"
            f"Title: {title}\n"
            f"Section: {section}\n"
            f"Chunk: {chunk_id}\n"
            f"Text:\n{text}\n"
        )

    context = "\n---\n".join(parts)
    return context[:max_chars]


# -------------------------
# Main
# -------------------------

def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Question to ask.")
    ap.add_argument("--index", default=OS_INDEX, help="OpenSearch index name.")

    ap.add_argument("--k", type=int, default=8, help="Final number of chunks to use in context (after balancing).")
    ap.add_argument(
        "--candidates",
        type=int,
        default=40,
        help="Candidates per retriever (BM25 and kNN). Higher = better recall, slower.",
    )
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant (typical ~60).")

    ap.add_argument("--max-docs", type=int, default=3, help="Max distinct papers to include in context.")
    ap.add_argument("--max-chunks-per-doc", type=int, default=3, help="Max chunks per paper in context.")

    ap.add_argument("--arxiv-id", default=None, help="Optional: restrict retrieval to a single paper.")
    ap.add_argument(
        "--categories",
        default="",
        help="Optional: comma-separated arXiv categories to filter (e.g. cs.LG,cs.AI,stat.ML).",
    )
    ap.add_argument("--published-after", default=None, help="Optional: ISO date/time lower bound (gte).")
    ap.add_argument("--published-before", default=None, help="Optional: ISO date/time upper bound (lte).")

    ap.add_argument("--show-top-papers", action="store_true", help="Print top candidate papers before answering.")
    ap.add_argument("--show-context", action="store_true", help="Print retrieved context before answering.")

    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    filters = _build_filters(
        arxiv_id=args.arxiv_id,
        categories=categories,
        published_after=args.published_after,
        published_before=args.published_before,
    )

    client = os_client()

    # Embed once (used for kNN)
    qvec = embed(args.question)

    # Retrieve candidates from both retrievers
    bm25_hits = bm25_search(
        client,
        index_name=args.index,
        question=args.question,
        size=args.candidates,
        filters=filters,
    )
    knn_hits = knn_search(
        client,
        index_name=args.index,
        qvec=qvec,
        size=args.candidates,
        filters=filters,
    )

    if not bm25_hits and not knn_hits:
        print("No hits returned. Is the index empty, or do your filters exclude everything?")
        return

    # Fuse using RRF
    fused = rrf_fuse(bm25_hits, knn_hits, rrf_k=args.rrf_k, limit=max(args.candidates, args.k) * 2)

    if args.show_top_papers:
        print("\n=== Top candidate papers (pre-balance) ===")
        for line in summarize_top_papers(fused, top_n=8):
            print(line)

    # Balance across papers
    balanced = balance_hits_by_doc(
        fused,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )

    # Keep only top-k chunks after balancing
    selected = balanced[: args.k]

    context = build_context(selected)

    if args.show_context:
        print("\n=== Retrieved Context (post-balance) ===\n")
        print(context)
        print("\n=== End Context ===\n")

    answer = ask_llm(args.question, context)
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
