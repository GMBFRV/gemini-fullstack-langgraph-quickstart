import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local directory research agent")
    parser.add_argument("question", help="Research question")
    # Local corpus root for search (web search replacement)
    parser.add_argument("--dir", default=".", help="Local directory to scan recursively")

    parser.add_argument("--max-files", type=int, default=30, help="Max files to consider after scan")
    parser.add_argument("--deep-k", type=int, default=8, help="Max files to deep-read")
    parser.add_argument("--min-files", type=int, default=1, help="Min files to read before allowing stop")
    parser.add_argument("--max-loops", type=int, default=10, help="Max read+reflect loops")

    parser.add_argument("--top-chunks", type=int, default=5, help="Top chunks per file")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size (chars)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap (chars)")
    parser.add_argument("--profile-chars", type=int, default=2500, help="Chars for quick file profile preview")

    parser.add_argument("--reasoning-model", default="llama-3.3-70b-versatile", help="Model for reflection/final answer")
    parser.add_argument("--recursion-limit", type=int, default=None, help="LangGraph recursion limit override")

    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
        "search_dir": args.dir,
        "max_files": args.max_files,
        "deep_k": args.deep_k,
        "min_files": args.min_files,
        "max_research_loops": args.max_loops,
        "top_chunks": args.top_chunks,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "profile_chars": args.profile_chars,
        "research_loop_count": 0,
        "read_files_count": 0,
        "pending_queries": [],
        "reasoning_model": args.reasoning_model,
    }

    # Ensure recursion_limit is always higher than the configured loop budget.
    recursion_limit = args.recursion_limit or (1 + 2 * int(args.max_loops) + 1 + 20)

    result = graph.invoke(state, config={"recursion_limit": recursion_limit})

    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
