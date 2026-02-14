# app.py
"""
Minimal Streamlit frontend for your local RAG (OpenSearch + Ollama).

UI goals:
- Uncluttered: sidebar for settings/filters, main search bar + answer.
- Hybrid retrieval: BM25 + kNN + RRF fusion + doc-balanced context.
- Shows "Top candidate papers" and (optional) retrieved context.

Run:
  streamlit run app.py
"""

from __future__ import annotations

import os
import textwrap
from collections import defaultdict
from typing import Any, Optional

import requests
import streamlit as st
from opensearchpy import OpenSearch


# -------------------------
# Config (env overrides)
# -------------------------

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OS_INDEX = os.getenv("OS_INDEX", "rag_chunks")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")


# -------------------------
# Backend helpers
# -------------------------

@st.cache_resource(show_spinner=False)
def get_os_client() -> OpenSearch:
    """Create and cache OpenSearch client."""
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def embed(text: str, model: str = OLLAMA_EMBED_MODEL) -> list[float]:
    """Embed text via Ollama."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _build_filters(
    arxiv_id: Optional[str],
    categories: list[str],
    published_after: Optional[str],
    published_before: Optional[str],
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if arxiv_id:
        filters.append({"term": {"arxiv_id": arxiv_id}})
    if categories:
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
    question: str,
    size: int,
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    resp = client.search(index=OS_INDEX, body=body)
    return resp.get("hits", {}).get("hits", [])


def knn_search(
    client: OpenSearch,
    qvec: list[float],
    size: int,
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    resp = client.search(index=OS_INDEX, body=body)
    return resp.get("hits", {}).get("hits", [])


def rrf_fuse(
    bm25_hits: list[dict[str, Any]],
    knn_hits: list[dict[str, Any]],
    rrf_k: int = 60,
    limit: int = 80,
) -> list[dict[str, Any]]:
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
    max_docs: int,
    max_chunks_per_doc: int,
) -> list[dict[str, Any]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hits:
        src = h.get("_source", {})
        aid = src.get("arxiv_id") or "unknown"
        by_doc[aid].append(h)

    for aid in by_doc:
        by_doc[aid].sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)

    ranked_docs = sorted(
        by_doc.items(),
        key=lambda kv: kv[1][0].get("_rrf_score", 0.0),
        reverse=True,
    )

    chosen: list[dict[str, Any]] = []
    for aid, doc_hits in ranked_docs[:max_docs]:
        chosen.extend(doc_hits[:max_chunks_per_doc])

    chosen.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return chosen


def summarize_top_papers(hits: list[dict[str, Any]], top_n: int = 5) -> list[str]:
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
            stats[aid]["best"] = max(stats[aid]["best"], score)
            if not stats[aid]["title"] and title:
                stats[aid]["title"] = title

    ranked = sorted(stats.items(), key=lambda kv: kv[1]["best"], reverse=True)[:top_n]
    return [f"{i}) {aid}  (best={s['best']:.4f}, hits={s['count']})  {s['title']}"
            for i, (aid, s) in enumerate(ranked, start=1)]


def build_context(hits: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        parts.append(
            f"[{i}] arXiv: {src.get('arxiv_id','')}\n"
            f"Title: {src.get('title','')}\n"
            f"Section: {src.get('section','')}\n"
            f"Chunk: {src.get('chunk_id','')}\n"
            f"Text:\n{(src.get('text','') or '').strip()}\n"
        )
    ctx = "\n---\n".join(parts)
    return ctx[:max_chars]


def ask_llm(question: str, context: str, model: str = OLLAMA_CHAT_MODEL) -> str:
    system = (
        "You are a helpful research assistant. Answer using ONLY the provided context. "
        "Cite chunk numbers like [1], [2] when making claims. "
        "If the context is insufficient, say what is missing."
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
    return r.json()["message"]["content"]


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Local RAG", page_icon="🔎", layout="centered")

st.title("🔎 Local RAG")
st.caption("Hybrid retrieval (BM25 + vectors) over your indexed arXiv papers.")

with st.sidebar:
    st.header("Settings")

    st.text_input("OpenSearch index", value=OS_INDEX, key="index_name", disabled=True)
    st.text_input("Embedding model", value=OLLAMA_EMBED_MODEL, key="embed_model", disabled=True)
    st.text_input("Chat model", value=OLLAMA_CHAT_MODEL, key="chat_model", disabled=True)

    st.divider()
    st.subheader("Filters (optional)")
    arxiv_id = st.text_input("arXiv ID (restrict to one paper)", value="").strip() or None
    cats_raw = st.text_input("Categories (comma-separated)", value="cs.LG,cs.AI,stat.ML").strip()
    categories = [c.strip() for c in cats_raw.split(",") if c.strip()]

    published_after = st.text_input("Published after (ISO)", value="").strip() or None
    published_before = st.text_input("Published before (ISO)", value="").strip() or None

    st.divider()
    st.subheader("Retrieval")
    candidates = st.slider("Candidates per retriever", min_value=10, max_value=200, value=40, step=10)
    rrf_k = st.slider("RRF k", min_value=10, max_value=200, value=60, step=10)
    max_docs = st.slider("Max papers in context", min_value=1, max_value=10, value=3, step=1)
    max_chunks_per_doc = st.slider("Max chunks per paper", min_value=1, max_value=8, value=3, step=1)
    k_final = st.slider("Final chunks to send to LLM", min_value=2, max_value=20, value=8, step=1)

    st.divider()
    show_top = st.checkbox("Show top candidate papers", value=True)
    show_context = st.checkbox("Show retrieved context", value=False)


question = st.text_input("Ask a question", placeholder="e.g., How are attention coefficients computed in GAT?")
ask = st.button("Search", type="primary", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        client = get_os_client()

        with st.spinner("Embedding question..."):
            qvec = embed(question.strip())

        filters = _build_filters(
            arxiv_id=arxiv_id,
            categories=categories,
            published_after=published_after,
            published_before=published_before,
        )

        with st.spinner("Retrieving (BM25 + vector) ..."):
            bm25_hits = bm25_search(client, question=question, size=candidates, filters=filters)
            knn_hits = knn_search(client, qvec=qvec, size=candidates, filters=filters)
            fused = rrf_fuse(bm25_hits, knn_hits, rrf_k=rrf_k, limit=max(candidates, k_final) * 2)

        if not fused:
            st.info("No results found. Try removing filters or increasing candidates.")
            st.stop()

        if show_top:
            st.subheader("Top candidate papers")
            for line in summarize_top_papers(fused, top_n=8):
                st.write(line)

        balanced = balance_hits_by_doc(fused, max_docs=max_docs, max_chunks_per_doc=max_chunks_per_doc)
        selected = balanced[:k_final]
        context = build_context(selected)

        if show_context:
            with st.expander("Retrieved context", expanded=False):
                st.code(context)

        with st.spinner("Generating answer..."):
            answer = ask_llm(question.strip(), context)

        st.subheader("Answer")
        st.write(answer)

    except requests.exceptions.ConnectionError as e:
        st.error(
            "Connection error. Is OpenSearch and Ollama running?\n\n"
            f"- OpenSearch: http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}\n"
            f"- Ollama: {OLLAMA_URL}\n\n"
            f"Details: {e}"
        )
    except Exception as e:
        st.error(f"Error: {e}")
