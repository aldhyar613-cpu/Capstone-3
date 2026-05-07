# =============================================================
# agent/tools.py
# Definisi 6 tools yang dipakai oleh debate_agent.py.
# Setiap tool adalah fungsi Python biasa yang dibungkus
# menjadi LangChain Tool menggunakan @tool decorator.
# Tools:
#   1. search_movies_tool       — semantic search via Qdrant
#   2. get_movie_details_tool   — data lengkap film dari MySQL
#   3. compare_stats_tool       — head-to-head dua film (MySQL)
#   4. filter_movies_tool       — filter eksak via MySQL
#   5. save_debate_history_tool — simpan pesan ke MySQL
#   6. get_history_tool         — ambil riwayat sesi dari MySQL
# =============================================================

from langchain.tools import tool
from qdrant_client import QdrantClient

from imdb_search.config import QDRANT_PATH
from imdb_search.embedder import embed_query
from imdb_search.vector_store import get_qdrant_client, search_vectors
from database.movie_repo import (
    get_movie_by_title,
    filter_movies,
    compare_two_movies,
)
from database.session_repo import (
    save_message,
    get_session_history,
    get_all_sessions,
)


# Client Qdrant — dibuat sekali, dipakai semua tool yang butuh
_qdrant_client: QdrantClient = None


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()   # pakai fungsi dari vector_store (cloud-aware)
    return _qdrant_client


# ------------------------------------------------------------------
# TOOL 1: Semantic Search via Qdrant
# PERBAIKAN: format output lebih terstruktur agar LLM bisa render
# dengan Rating, Sutradara, Deskripsi per film
# ------------------------------------------------------------------

@tool
def search_movies_tool(query: str) -> str:
    """
    Cari film yang relevan secara semantik dari Qdrant vector store.
    Gunakan tool ini saat butuh menemukan film berdasarkan deskripsi,
    tema, atau kata kunci — bukan judul eksak.

    Input  : query string (kalimat deskriptif)
    Output : string berisi daftar film relevan dengan metadata
    """
    try:
        client  = _get_qdrant_client()
        vector  = embed_query(query)
        results = search_vectors(client=client, query_vector=vector, top_k=5)

        if not results:
            return "Tidak ada film yang ditemukan untuk query ini."

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. **{r['title']}** ({r['year']})\n"
                f"   - Rating: {r['imdb_rating']}/10\n"
                f"   - Sutradara: {r['director']}\n"
                f"   - Genre: {r['genre']}\n"
                f"   - Deskripsi: {r['overview'][:200]}"
            )
        return "\n\n".join(lines)

    except Exception as e:
        return f"Error saat search: {e}"


# ------------------------------------------------------------------
# TOOL 2: Get Movie Details dari MySQL
# ------------------------------------------------------------------

@tool
def get_movie_details_tool(title: str) -> str:
    """
    Ambil data lengkap satu film dari MySQL berdasarkan judul eksak.
    Gunakan tool ini saat butuh detail spesifik: rating, gross,
    runtime, cast, metascore, overview.

    Input  : judul film (string)
    Output : string berisi semua metadata film
    """
    try:
        movie = get_movie_by_title(title)

        if not movie:
            return f"Film '{title}' tidak ditemukan di database MySQL."

        gross = f"${movie['gross_usd']:,}" if movie.get("gross_usd") else "Data tidak tersedia"
        meta  = movie.get("meta_score") or "N/A"
        stars = ", ".join(filter(None, [
            movie.get("star1"), movie.get("star2"),
            movie.get("star3"), movie.get("star4"),
        ]))

        return (
            f"Judul       : {movie['title']} ({movie.get('year', 'N/A')})\n"
            f"Genre       : {movie.get('genre', 'N/A')}\n"
            f"Sutradara   : {movie.get('director', 'N/A')}\n"
            f"Bintang     : {stars}\n"
            f"Certificate : {movie.get('certificate', 'N/A')}\n"
            f"IMDB Rating : {movie.get('imdb_rating', 'N/A')}/10\n"
            f"Metascore   : {meta}\n"
            f"Jumlah Vote : {movie.get('no_of_votes', 'N/A'):,}\n"
            f"Runtime     : {movie.get('runtime_min', 'N/A')} menit\n"
            f"Box Office  : {gross}\n"
            f"Sinopsis    : {movie.get('overview', 'N/A')}"
        )

    except Exception as e:
        return f"Error saat ambil detail film: {e}"


# ------------------------------------------------------------------
# TOOL 3: Compare Stats (head-to-head MySQL)
# PERBAIKAN: output markdown tabel agar LLM bisa render tabel
# perbandingan yang rapi di Streamlit
# ------------------------------------------------------------------

@tool
def compare_stats_tool(titles: str) -> str:
    """
    Bandingkan statistik dua film secara head-to-head dari MySQL.
    Gunakan tool ini untuk mode COMPARE agar dapat data akurat.

    Input  : dua judul dipisah koma, contoh: "Inception, Interstellar"
    Output : string perbandingan lengkap dengan pemenang per kategori
    """
    try:
        parts = [t.strip() for t in titles.split(",")]
        if len(parts) < 2:
            return "Masukkan dua judul film dipisah koma. Contoh: 'Inception, Interstellar'"

        title_a, title_b = parts[0], parts[1]
        result = compare_two_movies(title_a, title_b)

        if "error" in result:
            return result["error"]

        a = result["film_a"]
        b = result["film_b"]
        w = result["winners"]

        def fmt_gross(val):
            return f"${val:,}" if isinstance(val, (int, float)) and val else "N/A"

        def fmt_num(val):
            return f"{val:,}" if isinstance(val, (int, float)) and val else "N/A"

        # Output sebagai markdown tabel agar LLM bisa render langsung
        output = f"""DATA PERBANDINGAN: {a['title']} vs {b['title']}

| Kategori | {a['title']} | {b['title']} | Pemenang |
|----------|------------|------------|----------|
| Rating IMDB | {a.get('imdb_rating', 'N/A')}/10 | {b.get('imdb_rating', 'N/A')}/10 | {w.get('rating_tertinggi', 'N/A')} |
| Metascore | {a.get('meta_score', 'N/A')} | {b.get('meta_score', 'N/A')} | {w.get('metascore_tertinggi', 'N/A')} |
| Gross Box Office | {fmt_gross(a.get('gross_usd'))} | {fmt_gross(b.get('gross_usd'))} | {w.get('gross_terbesar', 'N/A')} |
| Jumlah Vote | {fmt_num(a.get('no_of_votes'))} | {fmt_num(b.get('no_of_votes'))} | {w.get('vote_terbanyak', 'N/A')} |
| Runtime | {a.get('runtime_min', 'N/A')} menit | {b.get('runtime_min', 'N/A')} menit | - |
| Genre | {a.get('genre', 'N/A')} | {b.get('genre', 'N/A')} | - |
| Tahun | {a.get('year', 'N/A')} | {b.get('year', 'N/A')} | - |

Film A - Sutradara: {a.get('director', 'N/A')}
Film A - Sinopsis: {a.get('overview', 'N/A')}

Film B - Sutradara: {b.get('director', 'N/A')}
Film B - Sinopsis: {b.get('overview', 'N/A')}
"""
        return output

    except Exception as e:
        return f"Error saat compare: {e}"


# ------------------------------------------------------------------
# TOOL 4: Filter Movies dari MySQL
# PERBAIKAN: format output lebih terstruktur agar RECOMMEND
# outputnya rapi dengan Rating, Sutradara, Deskripsi per film
# ------------------------------------------------------------------

@tool
def filter_movies_tool(criteria: str) -> str:
    """
    Filter film dari MySQL berdasarkan kriteria terstruktur.
    Gunakan tool ini untuk mode RECOMMEND saat user menyebut
    syarat spesifik: genre tertentu, rating minimum, tahun, dll.

    Input  : kriteria dalam format "key:value" dipisah koma
             Key yang valid: genre, min_rating, max_rating,
             min_year, max_year, min_metascore, director, limit
    Contoh : "genre:Drama, min_rating:8.0, min_year:2000"
    Output : daftar film yang memenuhi kriteria
    """
    try:
        # Parse kriteria dari string
        params = {}
        for part in criteria.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            params[key.strip().lower()] = val.strip()

        results = filter_movies(
            genre         = params.get("genre"),
            min_rating    = float(params["min_rating"])    if "min_rating"    in params else None,
            max_rating    = float(params["max_rating"])    if "max_rating"    in params else None,
            min_year      = int(params["min_year"])        if "min_year"      in params else None,
            max_year      = int(params["max_year"])        if "max_year"      in params else None,
            min_metascore = int(params["min_metascore"])   if "min_metascore" in params else None,
            director      = params.get("director"),
            limit         = int(params.get("limit", 8)),
        )

        if not results:
            return "Tidak ada film yang memenuhi kriteria tersebut."

        lines = [f"Ditemukan {len(results)} film:\n"]
        for i, m in enumerate(results, 1):
            gross = f" | Gross: ${m['gross_usd']:,}" if m.get("gross_usd") else ""
            lines.append(
                f"{i}. **{m['title']}** ({m.get('year','?')})\n"
                f"   - Rating: {m.get('imdb_rating','?')}/10\n"
                f"   - Sutradara: {m.get('director','?')}\n"
                f"   - Genre: {m.get('genre','?')}{gross}\n"
                f"   - Deskripsi: {m.get('overview','')[:200]}"
            )
        return "\n\n".join(lines)

    except Exception as e:
        return f"Error saat filter: {e}"


# ------------------------------------------------------------------
# TOOL 5: Save Debate History ke MySQL
# ------------------------------------------------------------------

@tool
def save_debate_history_tool(data: str) -> str:
    """
    Simpan pesan ke riwayat debate di MySQL.
    Dipanggil agent secara otomatis setelah setiap respons.

    Input  : "session_id:123, role:assistant, message:teks pesan"
    Output : konfirmasi tersimpan atau pesan error
    """
    try:
        params = {}
        for part in data.split(",", 2):
            if ":" in part:
                key, val = part.split(":", 1)
                params[key.strip()] = val.strip()

        session_id = int(params["session_id"])
        role       = params["role"]
        message    = params["message"]

        save_message(session_id, role, message)
        return f"✓ Pesan ({role}) tersimpan ke sesi #{session_id}"

    except Exception as e:
        return f"Error saat simpan history: {e}"


# ------------------------------------------------------------------
# TOOL 6: Get History dari MySQL
# ------------------------------------------------------------------

@tool
def get_history_tool(session_id: str) -> str:
    """
    Ambil riwayat percakapan dari sesi debate tertentu.
    Gunakan tool ini saat user ingin melanjutkan diskusi sebelumnya.

    Input  : session_id (string angka)
             Gunakan "all" untuk melihat semua sesi yang pernah ada
    Output : riwayat percakapan atau daftar semua sesi
    """
    try:
        if session_id.strip().lower() == "all":
            sessions = get_all_sessions(limit=20)
            if not sessions:
                return "Belum ada sesi debate yang tersimpan."

            lines = ["Sesi debate yang tersimpan:\n"]
            for s in sessions:
                fa = s.get("film_a") or "-"
                fb = s.get("film_b") or "-"
                lines.append(
                    f"[#{s['id']}] {s['mode']} | {fa} vs {fb} | "
                    f"{s['total_pesan']} pesan | {s['created_at']}"
                )
            return "\n".join(lines)

        history = get_session_history(int(session_id))
        if not history:
            return f"Sesi #{session_id} tidak ditemukan atau belum ada pesan."

        lines = [f"Riwayat sesi #{session_id}:\n"]
        for msg in history:
            prefix = "👤 User" if msg["role"] == "user" else "🤖 AI"
            lines.append(f"{prefix}: {msg['message'][:200]}...")
        return "\n".join(lines)

    except Exception as e:
        return f"Error saat ambil history: {e}"


# ------------------------------------------------------------------
# Export semua tools sebagai list (dipakai debate_agent.py)
# ------------------------------------------------------------------

ALL_TOOLS = [
    search_movies_tool,
    get_movie_details_tool,
    compare_stats_tool,
    filter_movies_tool,
    save_debate_history_tool,
    get_history_tool,
]
