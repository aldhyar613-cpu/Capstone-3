# =============================================================
# agent/state.py
# Definisi state untuk LangGraph StateGraph.
#
# DebateState adalah "memori" yang dibawa antar node:
#   messages  → riwayat percakapan (HumanMessage / AIMessage / ToolMessage)
#   mode      → DEBATE / COMPARE / RECOMMEND
#   film_a    → judul film pertama (bisa None)
#   film_b    → judul film kedua (bisa None)
#   session_id → ID sesi MySQL yang aktif
# =============================================================

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class DebateState(TypedDict):
    """
    State yang mengalir antar node di LangGraph.

    Field messages menggunakan `add_messages` sebagai reducer —
    setiap node cukup return pesan baru, LangGraph otomatis
    menggabungkannya ke list yang sudah ada (tidak menimpa).
    """
    messages:   Annotated[list, add_messages]
    mode:       str
    film_a:     str | None
    film_b:     str | None
    session_id: int | None
