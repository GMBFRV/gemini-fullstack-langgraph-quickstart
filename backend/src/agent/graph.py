import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.prompts import (
    answer_instructions,
    get_current_date,
    local_reader_instructions,
    reflection_instructions,
)
from agent.state import OverallState
from agent.tools_and_schemas import Reflection
from agent.utils import (
    build_file_profile,
    format_search_results_for_prompt,
    get_research_topic,
    list_text_files,
    pick_excerpts_for_file,
    score_profile,
    sources_from_excerpts,
)

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")


def scan_directory(state: OverallState, config: RunnableConfig) -> dict:
    # Phase 1: local search replacement for web search.
    # Build cheap profiles for all files and rank them without reading full content.
    question = get_research_topic(state["messages"])
    directory = state["search_dir"]

    files = list_text_files(directory)
    profiles: dict[str, str] = {}
    scores: dict[str, int] = {}

    profile_chars = int(state.get("profile_chars", 2500))

    for fp in files:
        prof = build_file_profile(fp, profile_chars=profile_chars)
        profiles[fp] = prof
        scores[fp] = score_profile(question, [], fp, prof)

    # Keep score==0 files to support broad / directory-level questions
    ordered = sorted(files, key=lambda p: scores.get(p, 0), reverse=True)
    ranked = ordered[: int(state.get("max_files", 30))]

    return {
        "ranked_files": ranked,
        "file_profiles": profiles,
        "file_scores": scores,
        "read_files": [],
        "read_files_count": 0,
        "pending_queries": [],
        "web_research_result": [],
        "sources_gathered": [],
        "research_loop_count": 0,
    }


def read_next_file(state: OverallState, config: RunnableConfig) -> dict:
    # Phase 2: iterative deep read of local files instead of remote retrieval.
    # Selection is dynamic and can change based on reflection feedback.
    question = get_research_topic(state["messages"])
    ranked = state.get("ranked_files", [])
    already = set(state.get("read_files", []))
    pending = state.get("pending_queries", []) or []

    remaining = [fp for fp in ranked if fp not in already]
    if not remaining:
        return {"web_research_result": ["No relevant local sources found."], "sources_gathered": []}

    profiles = state.get("file_profiles", {}) or {}

    rescored = []
    for fp in remaining:
        prof = profiles.get(fp, "")
        rescored.append((score_profile(question, pending, fp, prof), fp))
    rescored.sort(key=lambda x: x[0], reverse=True)

    next_file = rescored[0][1]

    out = {
        "read_files": [next_file],
        "read_files_count": int(state.get("read_files_count", 0)) + 1,
    }

    excerpts = pick_excerpts_for_file(
        next_file,
        question,
        pending,
        chunk_size=int(state.get("chunk_size", 1200)),
        chunk_overlap=int(state.get("chunk_overlap", 150)),
        top_chunks=int(state.get("top_chunks", 5)),
    )

    if not excerpts:
        out.update(
            {
                "web_research_result": [f"Skipped unreadable or empty file: {next_file}"],
                "sources_gathered": [],
            }
        )
        return out

    file_id = int(state.get("read_files_count", 0))
    short_sources = sources_from_excerpts(next_file, file_id=file_id, excerpts=excerpts)
    excerpts_block = format_search_results_for_prompt(short_sources)

    configurable = Configuration.from_runnable_config(config)
    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = local_reader_instructions.format(
        current_date=get_current_date(),
        question=question,
        file_path=next_file,
        excerpts=excerpts_block,
    )
    summary = llm.invoke(prompt).content

    out.update({"web_research_result": [summary], "sources_gathered": short_sources})
    return out


def reflection(state: OverallState, config: RunnableConfig) -> dict:
    # Reflection decides whether local coverage is sufficient
    # and produces follow-up queries for further local search if needed.
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    next_count = int(state.get("research_loop_count", 0)) + 1
    summaries = "\n\n---\n\n".join(state.get("web_research_result", []))
    question = get_research_topic(state["messages"])

    prompt = reflection_instructions.format(question=question, summaries=summaries)

    llm = ChatGroq(
        model=reasoning_model,
        temperature=0.4,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(prompt)

    return {
        "is_sufficient": bool(result.is_sufficient),
        "knowledge_gap": result.knowledge_gap,
        "pending_queries": result.follow_up_queries or [],
        "research_loop_count": next_count,
        "read_files_count": int(state.get("read_files_count", 0)),
        "min_files": int(state.get("min_files", 1)),
        "max_research_loops": int(state.get("max_research_loops", 10)),
    }


def route_after_reflection(ref_out: dict) -> str:
    if ref_out.get("research_loop_count", 0) >= ref_out.get("max_research_loops", 10):
        return "finalize_answer"
    if ref_out.get("is_sufficient") and ref_out.get("read_files_count", 0) >= ref_out.get("min_files", 1):
        return "finalize_answer"
    return "read_next_file"


def finalize_answer(state: OverallState, config: RunnableConfig) -> dict:
    # Final answer is synthesized strictly from local summaries and sources.
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    question = get_research_topic(state["messages"])
    summaries = "\n---\n\n".join(state.get("web_research_result", []))

    prompt = answer_instructions.format(
        current_date=get_current_date(),
        question=question,
        summaries=summaries,
    )

    llm = ChatGroq(
        model=reasoning_model,
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.invoke(prompt)

    sources = state.get("sources_gathered", [])
    unique_sources = []
    content = result.content
    for source in sources:
        short = source.get("short_url")
        if short and short in content:
            content = content.replace(short, source["value"])
            unique_sources.append(source)

    return {"messages": [AIMessage(content=content)], "sources_gathered": unique_sources}


builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("scan_directory", scan_directory)
builder.add_node("read_next_file", read_next_file)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "scan_directory")
builder.add_edge("scan_directory", "read_next_file")
builder.add_edge("read_next_file", "reflection")
builder.add_conditional_edges("reflection", route_after_reflection, ["read_next_file", "finalize_answer"])
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="universal-local-agent")
