# =============================================================
# agent/graph.py
# Definisi LangGraph StateGraph untuk IMDB Movie Debate.
#
# Alur graph:
#   route_mode → call_tools → generate → END
#
# Setiap node adalah fungsi Python biasa yang menerima
# DebateState dan mengembalikan dict untuk update state.
#
# Checkpointer (SqliteSaver) otomatis menyimpan state tiap
# langkah — history sesi tidak perlu dikelola manual.
# =============================================================

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from imdb_search.config import OPENAI_API_KEY, LLM_MODEL
from agent.state import DebateState
from agent.tools import ALL_TOOLS
from agent.prompts import DEBATE_SYSTEM, COMPARE_SYSTEM, RECOMMEND_SYSTEM
from agent.mode_router import detect_mode_and_films



# ------------------------------------------------------------------
# LLM — dibuat sekali, dipakai semua node yang butuh LLM
# ------------------------------------------------------------------

_llm = ChatOpenAI(
    model       = LLM_MODEL,
    api_key     = OPENAI_API_KEY,
    temperature = 0.4,
    streaming   = True,
)

_llm_with_tools = _llm.bind_tools(ALL_TOOLS)


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
# Deteksi mode & film dari pesan user terakhir.
# Hanya jalan di pesan pertama sesi (mode masih kosong).
# ------------------------------------------------------------------

def route_mode(state: DebateState) -> dict:
    """
    Deteksi mode (DEBATE/COMPARE/RECOMMEND) dan nama film
    dari query user. Hasil disimpan ke state.
    """
    # Jika mode sudah ada (sesi lanjutan), skip deteksi
    if state.get("mode"):
        return {}

    # Ambil pesan user terakhir
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
# NODE 2: call_tools (via ToolNode dari LangGraph prebuilt)
# LangGraph ToolNode otomatis memanggil tool yang diminta LLM
# dan menambahkan ToolMessage ke state["messages"].
# ------------------------------------------------------------------

tool_node = ToolNode(ALL_TOOLS)


# ------------------------------------------------------------------
# NODE 3: generate
# LLM menghasilkan jawaban akhir berdasarkan konteks tools
# dan system prompt sesuai mode.
# ------------------------------------------------------------------

def generate(state: DebateState) -> dict:
    """
    Panggil LLM dengan system prompt sesuai mode dan
    semua pesan (termasuk hasil tool) sebagai konteks.
    """
    system_prompt = _get_system_prompt(
        mode   = state.get("mode", "RECOMMEND"),
        film_a = state.get("film_a"),
        film_b = state.get("film_b"),
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    print(f"[graph:generate] Calling LLM | mode={state.get('mode')} | messages={len(messages)}")
    response = _llm_with_tools.invoke(messages)

    return {"messages": [response]}


# ------------------------------------------------------------------
# Edge condition: apakah LLM minta tool call lagi?
# Jika ya → kembali ke tool_node
# Jika tidak → selesai (END)
# ------------------------------------------------------------------

def should_continue(state: DebateState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tools"
    return END


# ------------------------------------------------------------------
# Build & compile graph
# ------------------------------------------------------------------

def build_graph(db_path: str = "debate_history.db"):
    """
    Membangun dan mengompilasi LangGraph StateGraph.

    Parameter:
        db_path: path file SQLite untuk checkpointer
                 (default: debate_history.db di working directory)

    Return: compiled graph siap dipanggil dengan .invoke() atau .stream()
    """
    graph = StateGraph(DebateState)

    # Daftarkan semua node
    graph.add_node("route_mode",  route_mode)
    graph.add_node("call_tools",  tool_node)
    graph.add_node("generate",    generate)

    # Entry point
    graph.set_entry_point("route_mode")

    # Edges
    graph.add_edge("route_mode", "generate")
    graph.add_conditional_edges(
        "generate",
        should_continue,
        {"call_tools": "call_tools", END: END},
    )
    graph.add_edge("call_tools", "generate")

    # Checkpointer — state otomatis persist ke SQLite
   
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    print(f"[graph] Graph compiled | checkpointer={db_path}")
    return compiled
