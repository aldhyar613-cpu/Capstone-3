# =============================================================
# app.py — Streamlit UI untuk IMDB Movie Debate & Comparison
#
# Cara jalankan:
#   cd imdb-debate
#   streamlit run app.py
# =============================================================

import streamlit as st
from langchain_core.messages import HumanMessage

from agent.debate_agent import build_graph, run_agent
from agent.state import DebateState
from database.session_repo import get_all_sessions, get_session_history, get_stats

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title = "🎬 IMDB Movie Debate",
    page_icon  = "🎬",
    layout     = "wide",
)

# ------------------------------------------------------------------
# CSS tambahan
# ------------------------------------------------------------------
st.markdown("""
<style>
.mode-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}
.mode-DEBATE   { background:#fee2e2; color:#991b1b; }
.mode-COMPARE  { background:#dbeafe; color:#1e3a8a; }
.mode-RECOMMEND{ background:#dcfce7; color:#14532d; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "graph"         not in st.session_state:
    st.session_state.graph        = None
if "session_id"    not in st.session_state:
    st.session_state.session_id   = None
if "thread_id"     not in st.session_state:
    st.session_state.thread_id    = None
if "messages"      not in st.session_state:
    st.session_state.messages     = []
if "current_mode"  not in st.session_state:
    st.session_state.current_mode = None

# ------------------------------------------------------------------
# Load graph (cached agar tidak rebuild setiap rerun)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat agent...")
def load_graph():
    return build_graph()

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 IMDB Debate")
    st.caption("Powered by LangGraph + Qdrant + MySQL")

    st.divider()

    if st.button("➕ Sesi Baru", use_container_width=True, type="primary"):
        st.session_state.session_id   = None
        st.session_state.thread_id    = None
        st.session_state.messages     = []
        st.session_state.current_mode = None
        st.rerun()

    st.subheader("📋 Riwayat Sesi")

    try:
        sessions = get_all_sessions(limit=20)
        if sessions:
            for s in sessions:
                fa    = s.get("film_a") or "?"
                fb    = s.get("film_b") or "?"
                label = f"[{s['mode']}] {fa[:15]} vs {fb[:15]}"
                if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                    st.session_state.session_id   = s["id"]
                    st.session_state.thread_id    = f"session-{s['id']}"
                    st.session_state.current_mode = s["mode"]
                    history = get_session_history(s["id"])
                    st.session_state.messages = [
                        {"role": h["role"], "content": h["message"]}
                        for h in history
                    ]
                    st.rerun()
        else:
            st.caption("Belum ada riwayat sesi")
    except Exception as e:
        st.caption(f"MySQL belum terkoneksi: {e}")

    st.divider()

    st.subheader("📊 Statistik")
    try:
        stats = get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Total Sesi",  stats["total_sessions"])
        col2.metric("Total Query", stats["total_queries"])
        by_mode = stats.get("by_mode", {})
        for mode, cnt in by_mode.items():
            st.caption(f"{mode}: {cnt} sesi")
    except Exception:
        st.caption("Statistik tidak tersedia")

    st.divider()

    with st.expander("💡 Cara Pakai"):
        st.markdown("""
**Mode DEBATE**
> "Debatkan: Inception vs Interstellar, mana yang lebih baik?"

**Mode COMPARE**
> "Bandingkan data The Dark Knight vs Avengers"

**Mode RECOMMEND**
> "Rekomendasikan film thriller psikologis rating di atas 8"

Mode terdeteksi **otomatis** dari kalimatmu!
        """)

# ------------------------------------------------------------------
# AREA UTAMA
# ------------------------------------------------------------------
st.title("🎬 IMDB Movie Debate & Comparison")

if st.session_state.current_mode:
    mode   = st.session_state.current_mode
    colors = {"DEBATE": "🔴", "COMPARE": "🔵", "RECOMMEND": "🟢"}
    icon   = colors.get(mode, "⚪")
    st.caption(f"{icon} Mode aktif: **{mode}**  |  Sesi #{st.session_state.session_id}")

# ------------------------------------------------------------------
# Tampilkan riwayat chat
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Placeholder contoh (muncul saat chat kosong)
# ------------------------------------------------------------------
if not st.session_state.messages:
    st.info(
        "👋 Mulai dengan mengetik pertanyaan debat atau perbandingan film!\n\n"
        "**Contoh:**\n"
        "- *Debatkan The Godfather vs Goodfellas*\n"
        "- *Bandingkan rating Parasite vs Joker*\n"
        "- *Rekomendasikan film perang terbaik*"
    )

# ------------------------------------------------------------------
# Input user
# ------------------------------------------------------------------
query = st.chat_input("Tanya tentang film, minta debat, atau minta rekomendasi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.graph is None:
        st.session_state.graph = load_graph()

    with st.chat_message("assistant"):
        with st.spinner("Agent sedang berpikir..."):
            try:
                result = run_agent(
                    query      = query,
                    graph      = st.session_state.graph,
                    session_id = st.session_state.session_id,
                    thread_id  = st.session_state.thread_id,
                )

                answer    = result["answer"]
                mode      = result["mode"]
                thread_id = result["thread_id"]

                st.session_state.session_id   = result["session_id"]
                st.session_state.thread_id    = thread_id
                st.session_state.current_mode = mode

                st.markdown(
                    f'<span class="mode-badge mode-{mode}">{mode}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                if result.get("film_a") or result.get("film_b"):
                    films = " vs ".join(filter(None, [result.get("film_a"), result.get("film_b")]))
                    st.caption(f"🎬 {films}")

            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
