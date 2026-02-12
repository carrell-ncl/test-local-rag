#!/usr/bin/env python3
"""
Enqueue arXiv papers into Redis for downstream ingestion.

This script queries the official arXiv API (Atom feed), extracts metadata,
and pushes jobs into a Redis queue compatible with `ingest_worker.py`.

Typical use cases:
- Build a subject-focused RAG corpus (e.g. Graph Neural Networks).
- Periodically ingest recent arXiv papers.
- Mirror an AWS-style fetch → queue → worker pipeline locally.

Example:
    python enqueue_arxiv.py \
        --query "graph neural network" \
        --category cs.LG \
        --max-results 50 \
        --dedupe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import redis
import requests


ARXIV_API = os.getenv("ARXIV_API", "https://export.arxiv.org/api/query")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "ingest:arxiv")

# Atom namespace used by arXiv API
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}

# Be polite to arXiv (recommended delay between requests)
DEFAULT_THROTTLE_S = float(os.getenv("ARXIV_THROTTLE_SECONDS", "3.0"))


@dataclass
class ArxivEntry:
    """Container for a parsed arXiv API entry."""

    arxiv_id: str
    version: str
    pdf_url: str
    title: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    summary: str
    link_abs: str


def redis_client() -> redis.Redis:
    """Create a Redis client.

    Returns:
        redis.Redis: Connected Redis client.
    """
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def build_search_query(query: str, category: Optional[str]) -> str:
    """Build an arXiv API search_query string.

    Args:
        query: Keyword or phrase to search for (searched in title or abstract).
        category: Optional arXiv category (e.g. "cs.LG", "cs.AI").

    Returns:
        str: A valid arXiv API search_query string.
    """
    parts: list[str] = []

    if query:
        phrase = query.replace('"', '\\"')
        parts.append(f'(ti:"{phrase}" OR abs:"{phrase}")')

    if category:
        parts.append(f"cat:{category}")

    return " AND ".join(parts) if parts else "all:*"


def arxiv_api_call(
    search_query: str,
    start: int,
    max_results: int,
    sort_by: str,
    sort_order: str,
    timeout_s: int = 60,
) -> str:
    """Call the arXiv API and return raw XML.

    Args:
        search_query: arXiv search_query expression.
        start: Offset into result set.
        max_results: Number of results to return.
        sort_by: Sort field (e.g. submittedDate).
        sort_order: Sort order (ascending or descending).
        timeout_s: HTTP timeout in seconds.

    Returns:
        str: Raw Atom XML response.
    """
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    url = ARXIV_API + "?" + urllib.parse.urlencode(params)
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.text


def parse_id_and_version(id_text: str) -> tuple[str, str]:
    """Extract arXiv ID and version from an entry ID URL.

    Args:
        id_text: Full arXiv ID URL (e.g. http://arxiv.org/abs/2401.12345v2).

    Returns:
        tuple[str, str]: (arxiv_id, version), e.g. ("2401.12345", "v2").
    """
    m = re.search(r"/abs/([^v]+)(v\d+)?$", id_text.strip())
    if not m:
        tail = id_text.rstrip("/").split("/")[-1]
        m2 = re.match(r"^(.+?)(v\d+)?$", tail)
        if m2:
            return m2.group(1), (m2.group(2) or "")
        return tail, ""
    return m.group(1), (m.group(2) or "")


def entry_pdf_link(entry: ET.Element) -> tuple[str, str]:
    """Extract PDF and abstract URLs from an arXiv entry.

    Args:
        entry: XML element representing an arXiv entry.

    Returns:
        tuple[str, str]: (pdf_url, abstract_url)
    """
    pdf_url = ""
    abs_url = ""

    for link in entry.findall("a:link", ATOM_NS):
        href = link.attrib.get("href", "")
        title = link.attrib.get("title", "")
        rel = link.attrib.get("rel", "")
        typ = link.attrib.get("type", "")

        if rel == "alternate" and (typ == "text/html" or "/abs/" in href):
            abs_url = href

        if title.lower() == "pdf" or href.endswith(".pdf"):
            pdf_url = href

    return pdf_url, abs_url


def parse_feed(xml_text: str) -> list[ArxivEntry]:
    """Parse an arXiv Atom feed into structured entries.

    Args:
        xml_text: Raw Atom XML returned by arXiv API.

    Returns:
        list[ArxivEntry]: Parsed arXiv entries.
    """
    root = ET.fromstring(xml_text)
    entries: list[ArxivEntry] = []

    for e in root.findall("a:entry", ATOM_NS):
        id_text = e.findtext("a:id", "", ATOM_NS).strip()
        title = e.findtext("a:title", "", ATOM_NS).strip()
        summary = e.findtext("a:summary", "", ATOM_NS).strip()
        published = e.findtext("a:published", "", ATOM_NS).strip()
        updated = e.findtext("a:updated", "", ATOM_NS).strip()

        authors = [
            a.findtext("a:name", "", ATOM_NS).strip()
            for a in e.findall("a:author", ATOM_NS)
            if a.findtext("a:name", "", ATOM_NS)
        ]

        categories = [
            c.attrib.get("term", "").strip()
            for c in e.findall("a:category", ATOM_NS)
            if c.attrib.get("term")
        ]

        pdf_url, abs_url = entry_pdf_link(e)
        arxiv_id, version = parse_id_and_version(id_text)

        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}{version}.pdf"
        if not abs_url and arxiv_id:
            abs_url = f"https://arxiv.org/abs/{arxiv_id}{version}"

        entries.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                version=version,
                pdf_url=pdf_url,
                title=" ".join(title.split()),
                authors=authors,
                categories=categories,
                published=published,
                updated=updated,
                summary=summary,
                link_abs=abs_url,
            )
        )

    return entries


def to_job(entry: ArxivEntry) -> dict[str, Any]:
    """Convert an ArxivEntry into a Redis ingestion job.

    Args:
        entry: Parsed arXiv entry.

    Returns:
        dict[str, Any]: Job payload compatible with ingest_worker.py.
    """
    return {
        "arxiv_id": entry.arxiv_id,
        "version": entry.version,
        "pdf_url": entry.pdf_url,
        "title": entry.title,
        "authors": entry.authors,
        "categories": entry.categories,
        "published": entry.published,
        "metadata": {
            "updated": entry.updated,
            "summary": entry.summary,
            "abs_url": entry.link_abs,
        },
    }


def enqueue_jobs(jobs: Iterable[dict[str, Any]], dedupe: bool) -> int:
    """Push jobs into Redis.

    Args:
        jobs: Iterable of job dictionaries.
        dedupe: Whether to deduplicate by arxiv_id + version.

    Returns:
        int: Number of jobs enqueued.
    """
    r = redis_client()
    count = 0

    for job in jobs:
        if dedupe:
            key = f"seen:{job.get('arxiv_id')}:{job.get('version')}"
            if r.setnx(key, "1") == 0:
                continue
            r.expire(key, 30 * 24 * 3600)

        r.rpush(REDIS_QUEUE, json.dumps(job))
        count += 1

    return count


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="graph neural network")
    ap.add_argument("--category", default=None)
    ap.add_argument("--max-results", type=int, default=25)
    ap.add_argument("--page-size", type=int, default=25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sort", default="submittedDate",
                    choices=["relevance", "lastUpdatedDate", "submittedDate"])
    ap.add_argument("--order", default="descending",
                    choices=["ascending", "descending"])
    ap.add_argument("--throttle-seconds", type=float, default=DEFAULT_THROTTLE_S)
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    search_query = build_search_query(args.query, args.category)
    total_target = max(0, args.max_results)
    page_size = max(1, min(args.page_size, 2000))

    print(f"[INFO] search_query: {search_query}")
    print(f"[INFO] target={total_target}, page_size={page_size}")

    enqueued = 0
    start = args.start

    while enqueued < total_target:
        batch = min(page_size, total_target - enqueued)

        xml_text = arxiv_api_call(
            search_query=search_query,
            start=start,
            max_results=batch,
            sort_by=args.sort,
            sort_order=args.order,
        )

        entries = parse_feed(xml_text)
        if not entries:
            break

        jobs = [to_job(e) for e in entries]
        added = enqueue_jobs(jobs, dedupe=args.dedupe)

        enqueued += added
        start += batch

        print(f"[OK] Enqueued {added} (total {enqueued}/{total_target})")
        time.sleep(max(0.0, args.throttle_seconds))

    print(f"[DONE] Total enqueued: {enqueued}")


if __name__ == "__main__":
    main()
