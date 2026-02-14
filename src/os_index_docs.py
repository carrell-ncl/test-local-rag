#!/usr/bin/env python3
"""
Create the OpenSearch index for document-level retrieval (rag_docs).

This index stores one document per paper (typically abstract/summary + metadata),
with a knn_vector field for embeddings.

Usage:
  python -m src.os_index_docs --index rag_docs --embed-model nomic-embed-text
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import requests
from opensearchpy import OpenSearch


OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def os_client() -> OpenSearch:
    """Create an OpenSearch client for local (no-auth) OpenSearch."""
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def get_embed_dim(embed_model: str) -> int:
    """Probe Ollama embeddings endpoint to infer vector dimension."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": embed_model, "prompt": "dimension probe"},
        timeout=120,
    )
    r.raise_for_status()
    emb = r.json()["embedding"]
    return len(emb)


def create_rag_docs_index(index_name: str, dim: int) -> None:
    """Create rag_docs index with knn_vector mapping.

    Args:
        index_name: Name of index to create.
        dim: Embedding dimensionality.
    """
    client = os_client()

    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
        return

    body: dict[str, Any] = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "arxiv_id": {"type": "keyword"},
                "version": {"type": "keyword"},
                "title": {"type": "text"},
                "authors": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "published": {"type": "date"},
                "abs_url": {"type": "keyword"},
                "pdf_url": {"type": "keyword"},
                # Document-level text used for BM25
                "abstract": {"type": "text"},
                # Vector embedding used for kNN
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    # defaults are fine for local; OpenSearch will use HNSW under the hood
                },
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    print(f"Created index '{index_name}' (dim={dim}).")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="rag_docs")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    args = ap.parse_args()

    dim = get_embed_dim(args.embed_model)
    print(f"Embedding model '{args.embed_model}' dimension: {dim}")
    create_rag_docs_index(args.index, dim)


if __name__ == "__main__":
    main()
