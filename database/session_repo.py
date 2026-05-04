# =============================================================
# database/session_repo.py
# Fungsi untuk:
#   - Buat sesi debate/comparison baru
#   - Simpan setiap pesan (user & assistant) ke history
#   - Ambil history sesi tertentu
#   - List semua sesi yang pernah ada (untuk sidebar Streamlit)
# =============================================================

from datetime import datetime
from database.connection import get_cursor


# ------------------------------------------------------------------
# SESSION
# ------------------------------------------------------------------

def create_session(film_a: str, film_b: str, mode: str) -> int:
    """
    Buat sesi debate baru.

    mode harus salah satu dari: 'DEBATE', 'COMPARE', 'RECOMMEND'
    Return: session_id (int) yang dipakai untuk simpan history
    """
    sql = """
        INSERT INTO debate_sessions (film_a, film_b, mode)
        VALUES (%s, %s, %s)
    """
    with get_cursor() as (conn, cur):
        cur.execute(sql, (film_a, film_b, mode.upper()))
        return cur.lastrowid


def get_all_sessions(limit: int = 50) -> list[dict]:
    """
    Ambil semua sesi debate, terbaru di atas.
    Dipakai oleh sidebar Streamlit untuk menampilkan riwayat.
    """
    sql = """
        SELECT
            id,
            film_a,
            film_b,
            mode,
            created_at,
            (SELECT COUNT(*) FROM debate_history WHERE session_id = debate_sessions.id) AS total_pesan
        FROM debate_sessions
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as (_, cur):
        cur.execute(sql, (limit,))
        return cur.fetchall()


def get_session_by_id(session_id: int) -> dict | None:
    """
    Ambil detail satu sesi by ID.
    """
    sql = "SELECT * FROM debate_sessions WHERE id = %s"
    with get_cursor() as (_, cur):
        cur.execute(sql, (session_id,))
        return cur.fetchone()


# ------------------------------------------------------------------
# HISTORY (PESAN)
# ------------------------------------------------------------------

def save_message(session_id: int, role: str, message: str):
    """
    Simpan satu pesan ke debate_history.

    role  : 'user' atau 'assistant'
    Dipanggil setelah setiap pertukaran pesan di agent.
    """
    sql = """
        INSERT INTO debate_history (session_id, role, message)
        VALUES (%s, %s, %s)
    """
    with get_cursor() as (_, cur):
        cur.execute(sql, (session_id, role, message))


def get_session_history(session_id: int) -> list[dict]:
    """
    Ambil semua pesan dalam satu sesi, urut dari awal.
    Return list of dict: [{"role": "user", "message": "..."}, ...]

    Dipakai untuk:
        1. Membangun ulang chat_history saat user kembali ke sesi lama
        2. Menampilkan riwayat percakapan di Streamlit
    """
    sql = """
        SELECT role, message, created_at
        FROM debate_history
        WHERE session_id = %s
        ORDER BY created_at ASC
    """
    with get_cursor() as (_, cur):
        cur.execute(sql, (session_id,))
        return cur.fetchall()


def get_recent_messages(session_id: int, n: int = 10) -> list[dict]:
    """
    Ambil N pesan terakhir dari sesi — untuk konteks agent.
    Lebih efisien dari ambil semua jika sesi sudah panjang.
    """
    sql = """
        SELECT role, message
        FROM debate_history
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_cursor() as (_, cur):
        cur.execute(sql, (session_id, n))
        rows = cur.fetchall()
    # Balik urutan agar kronologis
    return list(reversed(rows))


# ------------------------------------------------------------------
# STATISTIK (untuk halaman "Tentang" di Streamlit)
# ------------------------------------------------------------------

def get_stats() -> dict:
    """
    Ringkasan statistik penggunaan — ditampilkan di sidebar Streamlit.
    """
    with get_cursor() as (_, cur):
        cur.execute("SELECT COUNT(*) AS total FROM debate_sessions")
        total_sessions = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) AS total FROM debate_history WHERE role='user'")
        total_queries = cur.fetchone()["total"]

        cur.execute("""
            SELECT mode, COUNT(*) AS cnt
            FROM debate_sessions
            GROUP BY mode
        """)
        by_mode = {row["mode"]: row["cnt"] for row in cur.fetchall()}

    return {
        "total_sessions": total_sessions,
        "total_queries":  total_queries,
        "by_mode":        by_mode,
    }
