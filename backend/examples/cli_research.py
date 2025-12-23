import argparse
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from langchain_core.messages import HumanMessage
from agent.graph import graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument("--initial-queries", type=int, default=3)
    parser.add_argument("--max-loops", type=int, default=2)
    parser.add_argument(
        "--reasoning-model",
        default="llama-3.3-70b-versatile",
        help="Model for the final answer",
    )
    args = parser.parse_args()

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print(messages[-1].content)


if __name__ == "__main__":
    main()
