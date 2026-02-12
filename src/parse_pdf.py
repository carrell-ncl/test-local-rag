#!/usr/bin/env python3
"""
arXiv-friendly PDF parsing (born-digital PDFs; no OCR).

- Extracts text with pypdf.
- Light cleanup.
- Optional: strip references/bibliography tail (huge retrieval win for papers).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

_ws_re = re.compile(r"[ \t]+")
_nl_re = re.compile(r"\n{3,}")

# Common headings that mark the start of references in papers.
_REF_CUTOFF = re.compile(
    r"\n\s*(references|bibliography|acknowledg(e)?ments)\s*\n",
    re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: str | Path, max_pages: Optional[int] = None) -> str:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    parts: list[str] = []
    n_pages = len(reader.pages)
    limit = min(n_pages, max_pages) if max_pages else n_pages

    for i in range(limit):
        page = reader.pages[i]
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)

    text = "\n".join(parts)

    # Basic cleanup
    text = text.replace("\r", "\n")
    text = _ws_re.sub(" ", text)
    text = _nl_re.sub("\n\n", text)
    return text.strip()


def extract_arxiv_text(
    pdf_path: str | Path,
    *,
    strip_references: bool = True,
    max_pages: Optional[int] = None,
) -> str:
    """
    Extract arXiv paper text and optionally strip references tail.
    """
    text = extract_text_from_pdf(pdf_path, max_pages=max_pages)
    if strip_references:
        m = _REF_CUTOFF.search(text)
        if m:
            text = text[: m.start()].rstrip()
    return text


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to a PDF file")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--keep-references", action="store_true", help="Do not strip references/bibliography.")
    args = ap.parse_args()

    print(
        extract_arxiv_text(
            args.pdf,
            strip_references=not args.keep_references,
            max_pages=args.max_pages,
        )
    )
