from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
import operator

from langgraph.graph import add_messages
from typing_extensions import Annotated


class OverallState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]

    # Local search root (replaces web search input)
    search_dir: str

    # Local search limits
    max_files: int          # files kept after directory scan
    deep_k: int             # max files to read deeply
    min_files: int          # minimum files before early stop
    max_research_loops: int # read+reflect loop limit

    # Chunking for local files
    top_chunks: int
    chunk_size: int
    chunk_overlap: int

    # Cheap scan configuration
    profile_chars: int     # chars used for file profiling (no full read)

    # Scan results
    ranked_files: list[str]       # ordered local candidates
    file_profiles: dict[str, str] # path -> lightweight profile
    file_scores: dict[str, int]   # path -> initial relevance score

    # Read progress
    read_files: Annotated[list, operator.add]
    read_files_count: int

    # Reflection-driven refinement
    pending_queries: Annotated[list, operator.add]

    # Accumulated local summaries and sources
    web_research_result: Annotated[list, operator.add]  # kept name for graph compatibility
    sources_gathered: Annotated[list, operator.add]

    # Loop control / models
    research_loop_count: int
    reasoning_model: str

    # Reflection output
    is_sufficient: bool
    knowledge_gap: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)
