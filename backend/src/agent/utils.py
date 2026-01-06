from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


# File types supported by local search (targeted for markdown-based docs)
ALLOWED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".json",
    ".yaml",
    ".yml",
}


def get_research_topic(messages: List[AnyMessage]) -> str:
    # Extract the research question from the conversation state.
    if len(messages) == 1:
        return messages[-1].content

    research_topic = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            research_topic += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            research_topic += f"Assistant: {message.content}\n"
    return research_topic


def _tokenize(text: str) -> List[str]:
    # Lightweight tokenizer used for cheap relevance scoring.
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _read_text_file(path: Path) -> str:
    # Best-effort text read for local files.
    return path.read_text(encoding="utf-8", errors="ignore")


def list_text_files(directory: str) -> List[str]:
    # Recursively collect all text-like files from the local directory.
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []

    files: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() in ALLOWED_TEXT_EXTENSIONS or p.name in {"LICENSE", "COPYING", "README"}:
            files.append(str(p))
    return files


def extract_markdown_headings(text: str, max_lines: int = 120) -> str:
    # Extract early markdown headings for fast structural profiling.
    lines = (text or "").splitlines()
    out: List[str] = []
    for line in lines[:max_lines]:
        s = line.strip()
        if s.startswith("#"):
            out.append(s)
    return "\n".join(out).strip()


def build_file_profile(file_path: str, profile_chars: int) -> str:
    # Cheap scan phase:
    # build a lightweight profile (headings + preview) without full-text processing.
    path = Path(file_path)
    try:
        text = _read_text_file(path)
    except Exception:
        return ""

    preview = text[: max(0, profile_chars)]
    if path.suffix.lower() == ".md":
        heads = extract_markdown_headings(text)
        if heads:
            return f"HEADINGS:\n{heads}\n\nPREVIEW:\n{preview}".strip()
    return f"PREVIEW:\n{preview}".strip()


def score_profile(question: str, pending_queries: List[str], file_path: str, profile: str) -> int:
    # Cheap relevance scoring used during local scan and re-ranking.
    # Combines filename match, token overlap, and literal phrase bonus.
    base_text = (question or "").strip()
    for q in pending_queries or []:
        base_text += " " + (q or "")

    tokens = set(_tokenize(base_text))
    if not tokens:
        return 0

    name = Path(file_path).name.lower()
    score = 0
    for t in tokens:
        if t in name:
            score += 5

    prof_tokens = set(_tokenize(profile))
    score += len(tokens & prof_tokens)

    if question and question.lower() in (profile or "").lower():
        score += 10

    return score


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    # Split text into overlapping chunks for deep local reading.
    if chunk_size <= 0:
        return [(0, len(text), text)]

    overlap = max(0, min(overlap, chunk_size - 1))
    chunks: List[Tuple[int, int, str]] = []

    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunks.append((i, end, text[i:end]))
        if end == n:
            break
        i = end - overlap

    return chunks


def top_k_chunks(
    question: str,
    pending_queries: List[str],
    chunks: List[Tuple[int, int, str]],
    k: int,
) -> List[Tuple[int, int, str]]:
    # Select the most relevant chunks based on token overlap.
    base_text = (question or "").strip()
    for q in pending_queries or []:
        base_text += " " + (q or "")

    q_tokens = set(_tokenize(base_text))
    if not q_tokens or not chunks:
        return []

    scored: List[Tuple[int, Tuple[int, int, str]]] = []
    for ch in chunks:
        _, _, txt = ch
        tset = set(_tokenize(txt))
        sc = len(q_tokens & tset)
        if sc > 0:
            scored.append((sc, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[: max(1, k)]]


def pick_excerpts_for_file(
    file_path: str,
    question: str,
    pending_queries: List[str],
    *,
    chunk_size: int,
    chunk_overlap: int,
    top_chunks: int,
) -> List[str]:
    # Deep local read strategy:
    # 1) prefer top-k relevant chunks
    # 2) fallback to headings + early chunks to avoid missing content
    path = Path(file_path)
    try:
        text = _read_text_file(path)
    except Exception:
        return []

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    selected = top_k_chunks(question, pending_queries, chunks, top_chunks)

    if selected:
        return [txt for _, _, txt in selected]

    excerpts: List[str] = []
    if path.suffix.lower() == ".md":
        heads = extract_markdown_headings(text)
        if heads:
            excerpts.append("HEADINGS:\n" + heads)

    if chunks:
        excerpts.append(chunks[0][2])
        if len(chunks) > 1:
            excerpts.append(chunks[1][2])

    return excerpts[: max(1, min(2, top_chunks))]


def sources_from_excerpts(file_path: str, file_id: int, excerpts: List[str]) -> List[Dict[str, str]]:
    # Convert local excerpts into citation objects compatible with the agent prompts.
    out: List[Dict[str, str]] = []
    for idx, snippet in enumerate(excerpts):
        out.append(
            {
                "label": Path(file_path).name,
                "short_url": f"local://{file_id}-{idx}",
                "value": file_path,
                "title": f"{Path(file_path).name} (excerpt {idx + 1})",
                "snippet": (snippet or "").strip(),
            }
        )
    return out


def format_search_results_for_prompt(short_sources: List[Dict[str, str]]) -> str:
    # Format local excerpts into a prompt-ready citation block.
    lines: List[str] = []
    for i, s in enumerate(short_sources, start=1):
        title = s.get("title", "").strip()
        snippet = s.get("snippet", "").strip()
        short_url = s.get("short_url", "").strip()

        lines.append(f"S{i}: {title}")
        if snippet:
            lines.append(f"Snippet: {snippet}")
        lines.append(f"Use this citation exactly: [S{i}]({short_url})")
        lines.append("")
    return "\n".join(lines).strip()
