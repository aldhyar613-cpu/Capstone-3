# =============================================================
# tests/test_agent.py
# Unit test dan integration test untuk IMDB Debate agent.
#
# Jalankan:
#   pytest tests/ -v
#
# Catatan:
#   - Test yang menyentuh LLM membutuhkan OPENAI_API_KEY di .env
#   - Test tool (test_compare_tool, test_filter_tool) membutuhkan
#     MySQL dan Qdrant yang sudah diisi data
# =============================================================

import pytest
from unittest.mock import patch, MagicMock


# ------------------------------------------------------------------
# Test 1: Mode detection — DEBATE
# ------------------------------------------------------------------

def test_mode_detection_debate():
    """LLM harus mendeteksi mode DEBATE dan dua nama film."""
    from agent.mode_router import detect_mode_and_films

    result = detect_mode_and_films("Debatkan Inception vs Interstellar")

    assert result["mode"] == "DEBATE", f"Expected DEBATE, got {result['mode']}"
    assert result["film_a"] is not None, "film_a seharusnya terisi"
    assert result["film_b"] is not None, "film_b seharusnya terisi"


# ------------------------------------------------------------------
# Test 2: Mode detection — COMPARE
# ------------------------------------------------------------------

def test_mode_detection_compare():
    """LLM harus mendeteksi mode COMPARE untuk query perbandingan data."""
    from agent.mode_router import detect_mode_and_films

    result = detect_mode_and_films("Bandingkan rating The Dark Knight vs Joker")

    assert result["mode"] == "COMPARE", f"Expected COMPARE, got {result['mode']}"


# ------------------------------------------------------------------
# Test 3: Mode detection — RECOMMEND
# ------------------------------------------------------------------

def test_mode_detection_recommend():
    """LLM harus mendeteksi mode RECOMMEND untuk query tanpa dua film."""
    from agent.mode_router import detect_mode_and_films

    result = detect_mode_and_films("Rekomendasikan film thriller psikologis")

    assert result["mode"] == "RECOMMEND", f"Expected RECOMMEND, got {result['mode']}"
    assert result["film_a"] is None, "film_a seharusnya None untuk RECOMMEND"


# ------------------------------------------------------------------
# Test 4: DebateState structure
# ------------------------------------------------------------------

def test_debate_state_structure():
    """DebateState harus punya semua field yang dibutuhkan."""
    from agent.state import DebateState
    from typing import get_type_hints

    hints = get_type_hints(DebateState)
    required_fields = {"messages", "mode", "film_a", "film_b", "session_id"}

    assert required_fields.issubset(hints.keys()), (
        f"Field yang hilang: {required_fields - hints.keys()}"
    )


# ------------------------------------------------------------------
# Test 5: Graph build (tidak invoke LLM)
# ------------------------------------------------------------------

def test_graph_build():
    """Graph harus bisa dicompile tanpa error."""
    from agent.graph import build_graph

    graph = build_graph(db_path=":memory:")  # SQLite in-memory untuk test
    assert graph is not None


# ------------------------------------------------------------------
# Test 6: Tool — compare_stats_tool (membutuhkan MySQL)
# ------------------------------------------------------------------

@pytest.mark.integration
def test_compare_tool_returns_data():
    """compare_stats_tool harus return string HEAD-TO-HEAD."""
    from agent.tools import compare_stats_tool

    result = compare_stats_tool.invoke("The Dark Knight, Joker")

    assert isinstance(result, str), "Output harus string"
    assert "HEAD-TO-HEAD" in result or "tidak ditemukan" in result, (
        f"Output tidak terduga: {result[:100]}"
    )


# ------------------------------------------------------------------
# Test 7: Tool — filter_movies_tool (membutuhkan MySQL)
# ------------------------------------------------------------------

@pytest.mark.integration
def test_filter_tool_drama():
    """filter_movies_tool harus return daftar film Drama."""
    from agent.tools import filter_movies_tool

    result = filter_movies_tool.invoke("genre:Drama, min_rating:8.0, limit:3")

    assert isinstance(result, str)
    assert "Error" not in result, f"Tool error: {result}"


# ------------------------------------------------------------------
# Test 8: Graph end-to-end (membutuhkan LLM + tools)
# ------------------------------------------------------------------

@pytest.mark.integration
def test_graph_invoke_recommend():
    """Graph harus return state dengan messages dari LLM."""
    from agent.graph import build_graph
    from langchain_core.messages import HumanMessage

    graph  = build_graph(db_path=":memory:")
    config = {"configurable": {"thread_id": "test-thread-1"}}

    state = graph.invoke(
        {
            "messages":   [HumanMessage(content="Rekomendasikan film noir klasik")],
            "mode":       "",
            "film_a":     None,
            "film_b":     None,
            "session_id": None,
        },
        config=config,
    )

    assert len(state["messages"]) > 1, "Harus ada minimal 2 pesan (user + AI)"
    assert state["mode"] in ("DEBATE", "COMPARE", "RECOMMEND")
