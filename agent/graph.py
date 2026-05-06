# =============================================================
# agent/graph.py
# =============================================================

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from agent.state import DebateState
from agent.tools import ALL_TOOLS
from agent.prompts import DEBATE_SYSTEM, COMPARE_SYSTEM, RECOMMEND_SYSTEM
from agent.mode_router import detect_mode_and_films


# ------------------------------------------------------------------
# Helper — pilih system prompt berdasarkan mode
# ------------------------------------------------------------------

def _get_system_prompt(mode: str, film_a: str | None, film_b: str | None) -> str:
    fa = film_a or "Film A"
    fb = film_b or "Film B"
    if mode == "DEBATE":
        return DEBATE_SYSTEM.format(film_a=fa, film_b=fb, context="{context}")
    if mode == "COMPARE":
        return COMPARE_SYSTEM.format(film_a=fa, film_b=fb, context="{context}")
    return RECOMMEND_SYSTEM.format(context="{context}")


# ------------------------------------------------------------------
# NODE 1: route_mode
# ------------------------------------------------------------------

def route_mode(state: DebateState) -> dict:
    if state.get("mode"):
        return {}

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_human:
        return {"mode": "RECOMMEND", "film_a": None, "film_b": None}

    detected = detect_mode_and_films(last_human.content)
    print(f"[graph:route_mode] mode={detected['mode']} | film_a={detected['film_a']} | film_b={detected['film_b']}")

    return {
        "mode":   detected["mode"],
        "film_a": detected["film_a"],
        "film_b": detected["film_b"],
    }


# ------------------------------------------------------------------
# Edge condition
# ------------------------------------------------------------------

def should_continue(state: DebateState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tools"
    return END


# ------------------------------------------------------------------
# Build & compile graph — LLM dibuat di sini agar baca key terbaru
# ------------------------------------------------------------------

def build_graph(db_path: str = "debate_history.db"):
    from imdb_search.config import OPENAI_API_KEY, LLM_MODEL

    # LLM dibuat di dalam fungsi agar OPENAI_API_KEY sudah terisi
    llm = ChatOpenAI(
        model       = LLM_MODEL,
        api_key     = OPENAI_API_KEY,
        temperature = 0.4,
        streaming   = True,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    # NODE 2: generate — pakai llm_with_tools dari closure
    def generate(state: DebateState) -> dict:
        system_prompt = _get_system_prompt(
            mode   = state.get("mode", "RECOMMEND"),
            film_a = state.get("film_a"),
            film_b = state.get("film_b"),
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        print(f"[graph:generate] Calling LLM | mode={state.get('mode')} | messages={len(messages)}")
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(DebateState)

    graph.add_node("route_mode", route_mode)
    graph.add_node("call_tools", tool_node)
    graph.add_node("generate",   generate)

    graph.set_entry_point("route_mode")
    graph.add_edge("route_mode", "generate")
    graph.add_conditional_edges(
        "generate",
        should_continue,
        {"call_tools": "call_tools", END: END},
    )
    graph.add_edge("call_tools", "generate")

    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    print(f"[graph] Graph compiled | checkpointer={db_path}")
    return compiled