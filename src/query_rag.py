#!/usr/bin/env python3
"""
Two-stage RAG query:
1) Retrieve top papers from rag_docs (hybrid BM25 + kNN + RRF)
2) Retrieve top chunks from rag_chunks filtered to those papers (hybrid + RRF)
3) Ask Ollama chat model with retrieved chunk context

Run:
  python -m src.query_rag --question "..." --categories cs.LG,stat.ML --show-top
"""

from __future__ import annotations

import argparse
import os
import textwrap
from collections import defaultdict
from typing import Any, Optional

import requests
from opensearchpy import OpenSearch


OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

DOCS_INDEX = os.getenv("DOCS_INDEX", "rag_docs")
CHUNKS_INDEX = os.getenv("CHUNKS_INDEX", "rag_chunks")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


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


def _filters(
    *,
    categories: list[str],
    published_after: Optional[str],
    published_before: Optional[str],
    arxiv_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    f: list[dict[str, Any]] = []
    if categories:
        f.append({"terms": {"categories": categories}})
    if published_after or published_before:
        rng: dict[str, Any] = {}
        if published_after:
            rng["gte"] = published_after
        if published_before:
            rng["lte"] = published_before
        f.append({"range": {"published": rng}})
    if arxiv_ids:
        f.append({"terms": {"arxiv_id": arxiv_ids}})
    return f


def bm25_multi_match(fields: list[str], query: str) -> dict[str, Any]:
    return {
        "multi_match": {
            "query": query,
            "fields": fields,
            "type": "best_fields",
            "operator": "or",
        }
    }


def knn_query(field: str, vector: list[float], k: int) -> dict[str, Any]:
    return {"knn": {field: {"vector": vector, "k": k}}}


def search_bm25(
    client: OpenSearch,
    index: str,
    query: str,
    size: int,
    filters: list[dict[str, Any]],
    fields: list[str],
    source_fields: list[str],
) -> list[dict[str, Any]]:
    body = {
        "size": size,
        "_source": source_fields,
        "query": {"bool": {"filter": filters, "must": [bm25_multi_match(fields, query)]}},
    }
    resp = client.search(index=index, body=body)
    return resp.get("hits", {}).get("hits", [])


def search_knn(
    client: OpenSearch,
    index: str,
    vector: list[float],
    size: int,
    filters: list[dict[str, Any]],
    embedding_field: str,
    source_fields: list[str],
) -> list[dict[str, Any]]:
    body = {
        "size": size,
        "_source": source_fields,
        "query": {"bool": {"filter": filters, "must": [knn_query(embedding_field, vector, size)]}},
    }
    resp = client.search(index=index, body=body)
    return resp.get("hits", {}).get("hits", [])


def rrf_fuse(a: list[dict[str, Any]], b: list[dict[str, Any]], rrf_k: int = 60, limit: int = 100) -> list[dict[str, Any]]:
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

    add_list(a)
    add_list(b)

    fused = []
    for _id, h in by_id.items():
        h["_rrf_score"] = float(scores.get(_id, 0.0))
        fused.append(h)

    fused.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return fused[:limit]


def top_papers_summary(hits: list[dict[str, Any]], top_n: int = 8) -> list[str]:
    lines = []
    for i, h in enumerate(hits[:top_n], start=1):
        src = h.get("_source", {})
        lines.append(f"{i}) {src.get('arxiv_id')}  (rrf={h.get('_rrf_score', 0.0):.4f})  {src.get('title','')}")
    return lines


def balance_chunks(hits: list[dict[str, Any]], max_docs: int, max_chunks_per_doc: int) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hits:
        aid = (h.get("_source", {}) or {}).get("arxiv_id") or "unknown"
        by_doc[aid].append(h)

    for aid in by_doc:
        by_doc[aid].sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)

    ranked_docs = sorted(by_doc.items(), key=lambda kv: kv[1][0].get("_rrf_score", 0.0), reverse=True)

    chosen: list[dict[str, Any]] = []
    for aid, doc_hits in ranked_docs[:max_docs]:
        chosen.extend(doc_hits[:max_chunks_per_doc])

    chosen.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return chosen


def build_context(chunk_hits: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    parts: list[str] = []
    for i, h in enumerate(chunk_hits, start=1):
        s = h.get("_source", {}) or {}
        parts.append(
            f"[{i}] arXiv: {s.get('arxiv_id','')}\n"
            f"Title: {s.get('title','')}\n"
            f"Section: {s.get('section','')}\n"
            f"Chunk: {s.get('chunk_id','')}\n"
            f"Text:\n{(s.get('text','') or '').strip()}\n"
        )
    ctx = "\n---\n".join(parts)
    return ctx[:max_chars]


def ask_llm(question: str, context: str) -> str:
    system = (
        "You are a helpful research assistant. Answer using ONLY the provided context. "
        "Cite chunk numbers like [1], [2]. If context is insufficient, say what is missing."
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
            "model": OLLAMA_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)

    ap.add_argument("--categories", default="cs.LG,cs.AI,stat.ML")
    ap.add_argument("--published-after", default=None)
    ap.add_argument("--published-before", default=None)

    ap.add_argument("--paper-k", type=int, default=5, help="How many papers to shortlist from rag_docs.")
    ap.add_argument("--paper-candidates", type=int, default=30, help="Candidates per retriever for rag_docs.")

    ap.add_argument("--chunk-k", type=int, default=8, help="Final chunks for LLM context.")
    ap.add_argument("--chunk-candidates", type=int, default=60, help="Candidates per retriever for rag_chunks.")
    ap.add_argument("--max-docs", type=int, default=3)
    ap.add_argument("--max-chunks-per-doc", type=int, default=3)

    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--show-top", action="store_true")
    ap.add_argument("--show-context", action="store_true")

    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    client = os_client()
    qvec = embed(args.question)

    # ---- Stage 1: paper selection from rag_docs
    doc_filters = _filters(
        categories=categories,
        published_after=args.published_after,
        published_before=args.published_before,
        arxiv_ids=None,
    )

    docs_bm25 = search_bm25(
        client,
        index=DOCS_INDEX,
        query=args.question,
        size=args.paper_candidates,
        filters=doc_filters,
        fields=["title^3", "abstract"],
        source_fields=["arxiv_id", "title", "categories", "published"],
    )
    docs_knn = search_knn(
        client,
        index=DOCS_INDEX,
        vector=qvec,
        size=args.paper_candidates,
        filters=doc_filters,
        embedding_field="embedding",
        source_fields=["arxiv_id", "title", "categories", "published"],
    )
    docs_fused = rrf_fuse(docs_bm25, docs_knn, rrf_k=args.rrf_k, limit=200)

    if not docs_fused:
        print("No papers found in rag_docs (check filters, or build rag_docs).")
        return

    top_docs = docs_fused[: args.paper_k]
    top_arxiv_ids = [(h.get("_source", {}) or {}).get("arxiv_id") for h in top_docs]
    top_arxiv_ids = [x for x in top_arxiv_ids if x]

    if args.show_top:
        print("\n=== Top candidate papers (rag_docs) ===")
        for line in top_papers_summary(top_docs, top_n=min(8, len(top_docs))):
            print(line)

    # ---- Stage 2: chunk retrieval from rag_chunks filtered to shortlisted papers
    chunk_filters = _filters(
        categories=categories,
        published_after=args.published_after,
        published_before=args.published_before,
        arxiv_ids=top_arxiv_ids,
    )

    chunks_bm25 = search_bm25(
        client,
        index=CHUNKS_INDEX,
        query=args.question,
        size=args.chunk_candidates,
        filters=chunk_filters,
        fields=["title^3", "text"],
        source_fields=["arxiv_id", "title", "section", "chunk_id", "text", "categories", "published"],
    )
    chunks_knn = search_knn(
        client,
        index=CHUNKS_INDEX,
        vector=qvec,
        size=args.chunk_candidates,
        filters=chunk_filters,
        embedding_field="embedding",
        source_fields=["arxiv_id", "title", "section", "chunk_id", "text", "categories", "published"],
    )
    chunks_fused = rrf_fuse(chunks_bm25, chunks_knn, rrf_k=args.rrf_k, limit=400)

    if not chunks_fused:
        print("No chunks found in rag_chunks for shortlisted papers.")
        return

    balanced = balance_chunks(chunks_fused, max_docs=args.max_docs, max_chunks_per_doc=args.max_chunks_per_doc)
    selected = balanced[: args.chunk_k]

    context = build_context(selected)
    if args.show_context:
        print("\n=== Context ===\n")
        print(context)
        print("\n=== End Context ===\n")

    answer = ask_llm(args.question, context)
    print("\n=== Answer ===\n")
    print(answer)


if __name__ == "__main__":
    main()
