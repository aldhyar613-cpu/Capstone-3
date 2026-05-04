# =============================================================
# agent/debate_agent.py
# Entry point yang dipanggil oleh app.py (Streamlit).
#
# Sebelumnya: LangChain AgentExecutor + create_openai_tools_agent
# Sekarang  : LangGraph StateGraph (via agent/graph.py)
#
# Perubahan utama:
#   - build_agent() → build_graph() dari agent/graph.py
#   - run_agent() memanggil graph.invoke() bukan agent.invoke()
#   - History sesi dikelola checkpointer (SqliteSaver), bukan MySQL manual
#   - MySQL session_repo tetap dipakai untuk statistik & sidebar Streamlit
# =============================================================

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.state import DebateState
from database.session_repo import create_session, save_message


# ------------------------------------------------------------------
# Re-export build_graph agar app.py cukup import dari sini
# ------------------------------------------------------------------

__all__ = ["build_graph", "run_agent"]


# ------------------------------------------------------------------
# Main Function — dipanggil oleh app.py
# ------------------------------------------------------------------

def run_agent(
    query:      str,
    graph,
    session_id: int | None = None,
    thread_id:  str | None = None,
) -> dict:
    """
    Proses satu query user melalui LangGraph StateGraph.

    Parameter:
        query      : pertanyaan dari user
        graph      : compiled graph dari build_graph()
        session_id : ID sesi MySQL (None = buat baru setelah invoke)
        thread_id  : ID thread untuk LangGraph checkpointer
                     (None = generate otomatis dari session_id)

    Return dict:
        {
            "answer":     str,
            "mode":       str,
            "film_a":     str | None,
            "film_b":     str | None,
            "session_id": int,
            "thread_id":  str,
        }
    """
    # Step 1 — Siapkan config untuk checkpointer (thread per sesi)
    if thread_id is None:
        thread_id = f"session-{session_id}" if session_id else "session-new"

    config = {"configurable": {"thread_id": thread_id}}

    # Step 2 — Simpan pesan user ke MySQL (untuk sidebar & statistik)
    if session_id:
        save_message(session_id, "user", query)

    # Step 3 — Invoke graph
    print(f"[debate_agent] Invoke graph | thread={thread_id} | query='{query[:60]}'")
    try:
        initial_state: DebateState = {
            "messages":   [HumanMessage(content=query)],
            "mode":       "",        # akan diisi node route_mode
            "film_a":     None,
            "film_b":     None,
            "session_id": session_id,
        }

        result_state = graph.invoke(initial_state, config=config)

        # Ambil jawaban dari pesan terakhir
        last_message = result_state["messages"][-1]
        answer  = last_message.content if hasattr(last_message, "content") else str(last_message)
        mode    = result_state.get("mode", "RECOMMEND")
        film_a  = result_state.get("film_a")
        film_b  = result_state.get("film_b")

    except Exception as e:
        print(f"[debate_agent] ⚠ Error: {e}")
        answer = f"Terjadi kesalahan saat memproses query: {e}"
        mode   = "RECOMMEND"
        film_a = film_b = None

    # Step 4 — Buat sesi MySQL baru jika belum ada
    if session_id is None:
        session_id = create_session(
            film_a = film_a or "",
            film_b = film_b or "",
            mode   = mode,
        )
        # Simpan kedua pesan sekarang (user + assistant)
        save_message(session_id, "user",      query)
        save_message(session_id, "assistant", answer)
    else:
        # Simpan hanya jawaban assistant
        save_message(session_id, "assistant", answer)

    print(f"[debate_agent] Done | mode={mode} | session={session_id}")

    return {
        "answer":     answer,
        "mode":       mode,
        "film_a":     film_a,
        "film_b":     film_b,
        "session_id": session_id,
        "thread_id":  thread_id,
    }
