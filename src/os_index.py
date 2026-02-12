#!/usr/bin/env python3
"""
Create an OpenSearch index for arXiv RAG chunks with vector embeddings + metadata.

- Infers embedding dimension from Ollama embeddings endpoint.
- Adds arXiv-specific metadata fields (arxiv_id, authors, categories, version, published).
"""

from __future__ import annotations

import os
import requests
from opensearchpy import OpenSearch


def get_opensearch_client() -> OpenSearch:
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", "9200"))
    return OpenSearch([{"host": host, "port": port}])


def ollama_embedding_dim(model: str) -> int:
    base = os.getenv("OLLAMA_URL", "http://localhost:11434")
    r = requests.post(
        f"{base}/api/embeddings",
        json={"model": model, "prompt": "dimension probe"},
        timeout=60,
    )
    r.raise_for_status()
    vec = r.json()["embedding"]
    return len(vec)


def create_index(index_name: str, embedding_model: str) -> None:
    client = get_opensearch_client()

    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
        return

    dim = ollama_embedding_dim(embedding_model)
    print(f"Embedding model '{embedding_model}' dimension: {dim}")

    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                # Optional: tune refresh interval for bulk ingest performance
                # "refresh_interval": "30s",
            }
        },
        "mappings": {
            "properties": {
                # Identifiers
                "doc_id": {"type": "keyword"},   # often same as arxiv_id (or deterministic hash)
                "chunk_id": {"type": "keyword"}, # e.g. "Methods:3"
                "source": {"type": "keyword"},   # e.g. pdf_url
                "arxiv_id": {"type": "keyword"},
                "version": {"type": "keyword"},

                # Searchable metadata
                "title": {"type": "text"},
                "authors": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "published": {"type": "date"},
                "section": {"type": "keyword"},

                # Main content
                "text": {"type": "text"},
                "created_at": {"type": "date"},

                # Vector
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 16},
                    },
                },

                "metadata": {"type": "object", "enabled": True},
            }
        },
    }

    client.indices.create(index=index_name, body=body)
    print(f"Created index '{index_name}'.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.getenv("OS_INDEX", "rag_chunks"))
    ap.add_argument("--embed-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    args = ap.parse_args()

    create_index(args.index, args.embed_model)
