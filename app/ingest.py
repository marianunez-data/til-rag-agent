"""
ingest.py
---------
Data ingestion pipeline for GitHub repositories.
Downloads markdown documentation, applies chunking, and prepares
documents for indexing into a search engine.
"""

import io
import re
import zipfile

import frontmatter
import requests


def read_repo_data(repo_owner: str, repo_name: str, branch: str = "main") -> list[dict]:
    """
    Download and parse all markdown files from a GitHub repository.

    Args:
        repo_owner: GitHub username or organization.
        repo_name:  Repository name.
        branch:     Branch to download (default: main).

    Returns:
        List of dicts with keys from frontmatter metadata + 'content' + 'filename'.
    """
    url = (
        f"https://codeload.github.com/{repo_owner}/{repo_name}"
        f"/zip/refs/heads/{branch}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    documents = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for file_info in zf.infolist():
            filename = file_info.filename
            if not (filename.lower().endswith(".md") or filename.lower().endswith(".mdx")):
                continue
            try:
                with zf.open(file_info) as f:
                    content = f.read().decode("utf-8", errors="ignore")
                    post = frontmatter.loads(content)
                    doc = post.to_dict()
                    # Strip the repo root prefix (e.g. "til-master/") from filename
                    _, clean_filename = filename.split("/", maxsplit=1)
                    doc["filename"] = clean_filename
                    documents.append(doc)
            except Exception:
                continue

    return documents


def sliding_window(text: str, size: int = 2000, step: int = 1000) -> list[dict]:
    """
    Split a text string into overlapping fixed-size chunks.

    Args:
        text: Input text to chunk.
        size: Maximum characters per chunk.
        step: Step size between chunk start positions.

    Returns:
        List of dicts with keys 'start' and 'chunk'.
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive integers")

    n = len(text)
    chunks = []
    for i in range(0, n, step):
        chunks.append({"start": i, "chunk": text[i : i + size]})
        if i + size >= n:
            break
    return chunks


def split_by_headers(text: str, level: int = 1) -> list[str]:
    """
    Split a markdown document by header level.

    Args:
        text:  Markdown text.
        level: Header level to split on (1 = '#', 2 = '##').

    Returns:
        List of section strings, each beginning with its header.
    """
    pattern = re.compile(
        r"^(#{" + str(level) + r"} )(.+)$", re.MULTILINE
    )
    parts = pattern.split(text)
    sections = []
    for i in range(1, len(parts), 3):
        header = (parts[i] + parts[i + 1]).strip()
        body = parts[i + 2].strip() if i + 2 < len(parts) else ""
        sections.append(f"{header}\n\n{body}" if body else header)
    return sections


def chunk_documents(
    documents: list[dict],
    strategy: str = "sliding_window",
    size: int = 2000,
    step: int = 1000,
    header_level: int = 1,
) -> list[dict]:
    """
    Apply a chunking strategy to a list of documents.

    Args:
        documents:    List of document dicts (must include 'content').
        strategy:     'sliding_window' or 'by_headers'.
        size:         Chunk size in characters (sliding window only).
        step:         Step size in characters (sliding window only).
        header_level: Markdown header level (by_headers only).

    Returns:
        List of chunk dicts preserving document metadata.
    """
    chunks = []
    for doc in documents:
        doc_copy = doc.copy()
        content = doc_copy.pop("content", "")

        if strategy == "sliding_window":
            raw_chunks = sliding_window(content, size=size, step=step)
            for c in raw_chunks:
                c.update(doc_copy)
                chunks.append(c)

        elif strategy == "by_headers":
            sections = split_by_headers(content, level=header_level)
            if not sections:
                doc_copy["section"] = content.strip()
                chunks.append(doc_copy)
            else:
                for section in sections:
                    entry = doc_copy.copy()
                    entry["section"] = section
                    chunks.append(entry)

    return chunks


def filter_documents(
    documents: list[dict],
    exclude_filenames: list[str] | None = None,
    min_content_length: int = 100,
) -> list[dict]:
    """
    Filter documents by filename patterns and minimum content length.

    Args:
        documents:           List of document dicts.
        exclude_filenames:   Substrings to match against filename (case-insensitive).
        min_content_length:  Minimum character count for content field.

    Returns:
        Filtered list of document dicts.
    """
    exclude = [e.upper() for e in (exclude_filenames or [])]
    result = []
    for doc in documents:
        filename_upper = doc.get("filename", "").upper()
        if any(ex in filename_upper for ex in exclude):
            continue
        content = str(doc.get("content", doc.get("chunk", doc.get("section", ""))))
        if len(content) < min_content_length:
            continue
        result.append(doc)
    return result
