#!/usr/bin/env python3
"""
Enqueue arXiv papers into Redis for ingestion, with OpenSearch-based deduplication.

What it does:
- Queries arXiv Atom API for papers matching a query and optional category filters.
- Builds one "job" dict per paper (your existing job.json schema).
- Dedupes by arxiv_id against an OpenSearch index (rag_chunks or rag_docs).
- Pushes jobs onto Redis list queue (default: ingest:arxiv).

Why this is useful:
- Avoids hand-editing jobs.json.
- Lets you scale ingestion by running multiple ingest workers.
- Guarantees you don't re-ingest papers you already indexed.

Ensure Redis container running:
docker compose up -d redis 

Usage examples:
  # Enqueue 50 papers about graph neural networks from cs.LG/cs.AI/stat.ML, deduping against rag_chunks
  python -m src.enqueue_arxiv \
    --query "graph neural network" \
    --categories cs.LG,cs.AI,stat.ML \
    --max-results 50 \
    --dedupe \
    --dedupe-index rag_chunks

    make worker
    # or
    python -m src.ingest_worker --index rag_chunks


  # Enqueue by category only (no keyword query)
  python -m src.enqueue_arxiv --query "cat:cs.LG" --max-results 100 --dedupe

  # Write jobs to a file (JSON Lines) as well
  python -m src.enqueue_arxiv --query "graph attention" --max-results 25 --out jobs.jsonl --dedupe

Notes:
- arXiv API is eventually consistent; some queries may return fewer than requested.
- arXiv IDs may appear like "http://arxiv.org/abs/1710.10903v1" in the feed; we normalize.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests
import redis
from opensearchpy import OpenSearch


# -------------------------
# Environment defaults
# -------------------------

ARXIV_API = "https://export.arxiv.org/api/query"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "ingest:arxiv")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))

DEFAULT_DEDUPE_INDEX = os.getenv("OS_INDEX", "rag_chunks")


# -------------------------
# Helpers
# -------------------------

_ARXIV_ID_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/)?(?P<id>\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?$", re.I)


def normalize_arxiv_id(raw: str) -> str:
    """Normalize various arXiv ID formats to plain 'YYYY.NNNNN'.

    Args:
        raw: Raw ID or URL from arXiv feed.

    Returns:
        Normalized arXiv id.

    Raises:
        ValueError: if the id can't be normalized.
    """
    raw = (raw or "").strip()
    raw = raw.replace("http://", "https://")

    # Common feed form: https://arxiv.org/abs/1710.10903v1
    m = _ARXIV_ID_RE.search(raw)
    if m:
        return m.group("id")

    # Sometimes the <id> is like "oai:arXiv.org:1710.10903"
    if raw.lower().startswith("oai:arxiv.org:"):
        candidate = raw.split(":")[-1]
        m2 = _ARXIV_ID_RE.search(candidate)
        if m2:
            return m2.group("id")

    # If it's already plain
    if re.fullmatch(r"\d{4}\.\d{4,5}", raw):
        return raw

    raise ValueError(f"Could not normalize arXiv id from: {raw}")


def arxiv_abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def build_search_query(query: str, categories: list[str]) -> str:
    """Build an arXiv API search_query string.

    If query already contains arXiv operators (cat:, all:, ti:, au:, abs:),
    we assume the user knows what they are doing and we don't wrap it.

    Otherwise we use: all:"<query>"

    Categories (if provided) are added as: (cat:cs.LG OR cat:stat.ML ...)

    Args:
        query: Keyword query or full arXiv search expression.
        categories: arXiv categories to filter.

    Returns:
        arXiv API search_query string.
    """
    q = (query or "").strip()
    has_ops = any(op in q for op in ("cat:", "all:", "ti:", "au:", "abs:", "id:"))
    if not has_ops:
        q = f'all:"{q}"' if q else "all:*"

    if categories:
        cat_expr = " OR ".join([f"cat:{c.strip()}" for c in categories if c.strip()])
        if cat_expr:
            q = f"({q}) AND ({cat_expr})"

    return q


def os_client() -> OpenSearch:
    """Create OpenSearch client (local, no auth)."""
    return OpenSearch([{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


def redis_client() -> redis.Redis:
    """Create Redis client."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)


def os_has_arxiv_id(client: OpenSearch, index: str, arxiv_id: str) -> bool:
    """Check if an arxiv_id already exists in the given index.

    Uses a lightweight count query.

    Args:
        client: OpenSearch client.
        index: Index name (rag_chunks or rag_docs).
        arxiv_id: Normalized arXiv id.

    Returns:
        True if already present; else False.
    """
    try:
        resp = client.count(index=index, body={"query": {"term": {"arxiv_id": arxiv_id}}})
        return int(resp.get("count", 0)) > 0
    except Exception:
        # If index doesn't exist or any error, treat as not present
        return False


@dataclass
class ArxivEntry:
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    published: str  # ISO-like string from feed
    abs_url: str
    pdf_url: str


def parse_arxiv_feed(xml_text: str) -> list[ArxivEntry]:
    """Parse arXiv Atom feed XML into entries.

    Args:
        xml_text: Atom feed response body.

    Returns:
        List of ArxivEntry.
    """
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    entries = []

    for e in root.findall("atom:entry", ns):
        raw_id = (e.findtext("atom:id", default="", namespaces=ns) or "").strip()
        try:
            aid = normalize_arxiv_id(raw_id)
        except ValueError:
            # Try arxiv:doi or arxiv:journal_ref not useful for id; skip
            continue

        title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
        title = re.sub(r"\s+", " ", title)

        published = (e.findtext("atom:published", default="", namespaces=ns) or "").strip()

        authors = []
        for a in e.findall("atom:author", ns):
            name = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append(name)

        categories = []
        for c in e.findall("atom:category", ns):
            term = c.attrib.get("term", "").strip()
            if term:
                categories.append(term)

        # Links: abs + pdf
        abs_url = arxiv_abs_url(aid)
        pdf_url = arxiv_pdf_url(aid)
        for link in e.findall("atom:link", ns):
            href = link.attrib.get("href", "")
            ltype = link.attrib.get("type", "")
            rel = link.attrib.get("rel", "")
            title_attr = link.attrib.get("title", "")

            # Usually: rel="alternate" is abs page
            if rel == "alternate" and href:
                abs_url = href.replace("http://", "https://")
            # Usually: title="pdf" or type="application/pdf"
            if (title_attr.lower() == "pdf" or ltype == "application/pdf") and href:
                pdf_url = href.replace("http://", "https://")

        entries.append(
            ArxivEntry(
                arxiv_id=aid,
                title=title,
                authors=authors,
                categories=sorted(set(categories)),
                published=published,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )

    return entries


def fetch_arxiv_entries(
    *,
    search_query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
    polite_delay_s: float,
) -> list[ArxivEntry]:
    """Fetch a page of arXiv results."""
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    r = requests.get(ARXIV_API, params=params, timeout=60)
    r.raise_for_status()
    time.sleep(max(0.0, polite_delay_s))
    return parse_arxiv_feed(r.text)


def to_job(entry: ArxivEntry, manual_test: bool = False) -> dict[str, Any]:
    """Convert an arXiv entry into your ingestion job schema."""
    return {
        "arxiv_id": entry.arxiv_id,
        "version": "",
        "pdf_url": entry.pdf_url,
        "title": entry.title,
        "authors": entry.authors,
        "categories": entry.categories,
        "published": entry.published,
        "metadata": {
            "abs_url": entry.abs_url,
            "manual_test": bool(manual_test),
        },
    }


def enqueue_jobs(
    r: redis.Redis,
    queue: str,
    jobs: Iterable[dict[str, Any]],
) -> int:
    """Push jobs onto Redis list queue."""
    n = 0
    for job in jobs:
        r.rpush(queue, json.dumps(job))
        n += 1
    return n


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help='Keyword query, or full arXiv search expression (e.g. all:"graph neural network")')
    ap.add_argument("--categories", default="", help="Comma-separated arXiv categories to filter (e.g. cs.LG,cs.AI,stat.ML)")
    ap.add_argument("--max-results", type=int, default=50, help="Max number of papers to enqueue")
    ap.add_argument("--page-size", type=int, default=50, help="Page size per arXiv API call (<= 2000; recommend 50-200)")
    ap.add_argument("--start", type=int, default=0, help="Start offset for arXiv API")
    ap.add_argument("--sort-by", default="submittedDate", choices=["relevance", "lastUpdatedDate", "submittedDate"], help="arXiv sortBy")
    ap.add_argument("--sort-order", default="descending", choices=["ascending", "descending"], help="arXiv sortOrder")
    ap.add_argument("--polite-delay", type=float, default=0.25, help="Delay between API calls (seconds)")

    ap.add_argument("--queue", default=REDIS_QUEUE, help="Redis list name to enqueue jobs into")
    ap.add_argument("--dedupe", action="store_true", help="Dedupe against OpenSearch by arxiv_id")
    ap.add_argument("--dedupe-index", default=DEFAULT_DEDUPE_INDEX, help="OpenSearch index to check for dedupe (rag_chunks or rag_docs)")

    ap.add_argument("--out", default="", help="Optional: write enqueued jobs to a JSONL file")
    ap.add_argument("--manual-test", action="store_true", help="Set metadata.manual_test=true on jobs")

    args = ap.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    search_q = build_search_query(args.query, categories)

    print(f"[INFO] arXiv search_query: {search_q}")
    print(f"[INFO] Target enqueue count: {args.max_results}")
    print(f"[INFO] Redis queue: {args.queue}")
    if args.dedupe:
        print(f"[INFO] Dedupe enabled: OpenSearch index '{args.dedupe_index}'")
    if args.out:
        print(f"[INFO] Writing enqueued jobs to: {args.out} (JSONL)")

    os_cli = os_client() if args.dedupe else None
    r = redis_client()

    remaining = args.max_results
    start = args.start
    page_size = max(1, min(args.page_size, 2000))

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    enqueued = 0
    seen_ids: set[str] = set()

    try:
        while remaining > 0:
            batch_n = min(page_size, remaining)
            entries = fetch_arxiv_entries(
                search_query=search_q,
                start=start,
                max_results=batch_n,
                sort_by=args.sort_by,
                sort_order=args.sort_order,
                polite_delay_s=args.polite_delay,
            )
            if not entries:
                print("[INFO] No more results from arXiv.")
                break

            jobs_to_push: list[dict[str, Any]] = []
            for e in entries:
                if e.arxiv_id in seen_ids:
                    continue
                seen_ids.add(e.arxiv_id)

                if args.dedupe and os_cli is not None:
                    if os_has_arxiv_id(os_cli, args.dedupe_index, e.arxiv_id):
                        print(f"[SKIP] {e.arxiv_id} (already in {args.dedupe_index})  {e.title}")
                        continue

                job = to_job(e, manual_test=args.manual_test)
                jobs_to_push.append(job)

            if jobs_to_push:
                n = enqueue_jobs(r, args.queue, jobs_to_push)
                enqueued += n
                for job in jobs_to_push:
                    print(f"[ENQ] {job['arxiv_id']}  {job['title']}")
                    if out_f:
                        out_f.write(json.dumps(job, ensure_ascii=False) + "\n")
                remaining -= len(jobs_to_push)
            else:
                print("[INFO] Batch produced no new jobs (all duplicates or filtered).")

            # Next page
            start += batch_n

            # Safety: if we keep getting nothing new, don't loop forever
            if start > args.start + 10_000 and enqueued == 0:
                print("[WARN] Large start offset with no enqueues; stopping.")
                break

        print(f"[DONE] Enqueued {enqueued} jobs to Redis list '{args.queue}'.")
        if out_f:
            out_f.flush()

    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
