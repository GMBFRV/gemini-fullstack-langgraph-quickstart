import os
from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)

from langchain_groq import ChatGroq

from agent.utils import (
    get_research_topic,
    searxng_search,
    shorten_sources,
    format_search_results_for_prompt,
)

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")

# Public SearXNG instances (no key).
DEFAULT_SEARXNG_INSTANCES = [
    "https://searxng.site",
    "https://search.projectsegfau.lt",
    "https://searx.tiekoetter.com",
]


def _get_searx_instances() -> list[str]:
    env_val = os.getenv("SEARXNG_INSTANCES", "").strip()
    if env_val:
        return [x.strip() for x in env_val.split(",") if x.strip()]
    return DEFAULT_SEARXNG_INSTANCES


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generate search queries based on the user's question."""
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Spawn a web_research node per query."""
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    Perform web research using SearXNG (no API key), then summarize with citations.
    """
    configurable = Configuration.from_runnable_config(config)

    instances = _get_searx_instances()
    results = searxng_search(
        state["search_query"],
        instances=instances,
        max_results=8,
        timeout=12.0,
        lang="en",
    )

    # If no results, still return something (so reflection can decide to follow up)
    short_sources = shorten_sources(results, state["id"]) if results else []
    search_results_block = format_search_results_for_prompt(short_sources) if short_sources else "No results."

    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
        search_results=search_results_block,
    )

    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    summary = llm.invoke(formatted_prompt).content

    # sources_gathered should be a list of dicts (label, short_url, value, ...)
    return {
        "sources_gathered": short_sources,
        "search_query": [state["search_query"]],
        "web_research_result": [summary],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Identify knowledge gaps and generate follow-up queries."""
    configurable = Configuration.from_runnable_config(config)

    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatGroq(
        model=reasoning_model,
        temperature=0.7,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """Decide whether to continue searching or finalize."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"

    return [
        Send(
            "web_research",
            {
                "search_query": follow_up_query,
                "id": state["number_of_ran_queries"] + int(idx),
            },
        )
        for idx, follow_up_query in enumerate(state["follow_up_queries"])
    ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Finalize answer and replace short URLs with original URLs."""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatGroq(
        model=reasoning_model,
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and keep only used urls
    unique_sources = []
    sources = state.get("sources_gathered", [])

    for source in sources:
        short = source.get("short_url")
        if short and short in result.content:
            result.content = result.content.replace(short, source["value"])
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
