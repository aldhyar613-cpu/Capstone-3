# =============================================================
# database/movie_repo.py
# Fungsi untuk:
#   - Insert / bulk insert film dari preprocessor
#   - Query film by title (exact & fuzzy)
#   - Filter film by genre, rating, tahun, dll
#   - Agregasi untuk compare_stats (gross, rating, dll)
#
# Tugas: kelola data film di MySQL.
# =============================================================

from database.connection import get_cursor


# ------------------------------------------------------------------
# INSERT
# ------------------------------------------------------------------

def bulk_insert_movies(docs: list[dict]):
    """
    Menyimpan semua film dari preprocessor ke tabel movies.
    Jika film sudah ada (title + year sama), update datanya (upsert).

    Dipanggil oleh preprocessor.build_documents() saat sync_mysql=True.
    """
    sql = """
        INSERT INTO movies
            (title, year, genre, director, star1, star2, star3, star4,
             certificate, imdb_rating, meta_score, no_of_votes,
             runtime_min, gross_usd, overview)
        VALUES
            (%(title)s, %(year)s, %(genre)s, %(director)s,
             %(star1)s, %(star2)s, %(star3)s, %(star4)s,
             %(certificate)s, %(imdb_rating)s, %(meta_score)s,
             %(no_of_votes)s, %(runtime_min)s, %(gross_usd)s, %(overview)s)
        ON DUPLICATE KEY UPDATE
            genre       = VALUES(genre),
            imdb_rating = VALUES(imdb_rating),
            meta_score  = VALUES(meta_score),
            no_of_votes = VALUES(no_of_votes),
            gross_usd   = VALUES(gross_usd)
    """

    rows = []
    for d in docs:
        stars = d.get("stars", [])
        rows.append({
            "title":       d["title"],
            "year":        d.get("year"),
            "genre":       d.get("genre"),
            "director":    d.get("director"),
            "star1":       stars[0] if len(stars) > 0 else None,
            "star2":       stars[1] if len(stars) > 1 else None,
            "star3":       stars[2] if len(stars) > 2 else None,
            "star4":       stars[3] if len(stars) > 3 else None,
            "certificate": d.get("certificate"),
            "imdb_rating": d.get("imdb_rating"),
            "meta_score":  d.get("meta_score") if d.get("meta_score", -1) != -1 else None,
            "no_of_votes": d.get("no_of_votes"),
            "runtime_min": d.get("runtime_min"),
            "gross_usd":   int(d["gross_usd"]) if d.get("gross_usd") else None,
            "overview":    d.get("overview", ""),
        })

    with get_cursor() as (conn, cur):
        cur.executemany(sql, rows)
    print(f"[movie_repo] {len(rows)} film di-upsert ke MySQL")


# ------------------------------------------------------------------
# GET BY TITLE
# ------------------------------------------------------------------

def get_movie_by_title(title: str) -> dict | None:
    """
    Cari film by judul (exact match, case-insensitive).
    Return dict metadata lengkap atau None jika tidak ditemukan.
    """
    sql = "SELECT * FROM movies WHERE LOWER(title) = LOWER(%s) LIMIT 1"
    with get_cursor() as (_, cur):
        cur.execute(sql, (title,))
        return cur.fetchone()


def search_movies_by_title(title: str, limit: int = 5) -> list[dict]:
    """
    Cari film by judul (fuzzy/partial match).
    Berguna saat user typo atau pakai nama pendek.
    """
    sql = "SELECT * FROM movies WHERE title LIKE %s ORDER BY imdb_rating DESC LIMIT %s"
    with get_cursor() as (_, cur):
        cur.execute(sql, (f"%{title}%", limit))
        return cur.fetchall()


# ------------------------------------------------------------------
# FILTER
# ------------------------------------------------------------------

def filter_movies(
    genre:         str   = None,
    min_rating:    float = None,
    max_rating:    float = None,
    min_year:      int   = None,
    max_year:      int   = None,
    min_metascore: int   = None,
    min_gross:     int   = None,
    director:      str   = None,
    limit:         int   = 10,
) -> list[dict]:
    """
    Filter film dengan kondisi fleksibel dari MySQL.
    Dipakai oleh tool filter_movies_tool (mode RECOMMEND).

    Semua parameter opsional — kombinasikan sesuai kebutuhan.
    Return list dict film yang memenuhi semua syarat.
    """
    conditions = []
    params     = []

    if genre:
        conditions.append("genre LIKE %s")
        params.append(f"%{genre}%")
    if min_rating is not None:
        conditions.append("imdb_rating >= %s")
        params.append(min_rating)
    if max_rating is not None:
        conditions.append("imdb_rating <= %s")
        params.append(max_rating)
    if min_year is not None:
        conditions.append("year >= %s")
        params.append(min_year)
    if max_year is not None:
        conditions.append("year <= %s")
        params.append(max_year)
    if min_metascore is not None:
        conditions.append("meta_score >= %s")
        params.append(min_metascore)
    if min_gross is not None:
        conditions.append("gross_usd >= %s")
        params.append(min_gross)
    if director:
        conditions.append("director LIKE %s")
        params.append(f"%{director}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql   = f"SELECT * FROM movies {where} ORDER BY imdb_rating DESC LIMIT %s"
    params.append(limit)

    with get_cursor() as (_, cur):
        cur.execute(sql, params)
        return cur.fetchall()


# ------------------------------------------------------------------
# COMPARE STATS
# ------------------------------------------------------------------

def compare_two_movies(title_a: str, title_b: str) -> dict:
    """
    Ambil data dua film sekaligus dan hitung perbandingan head-to-head.
    Dipakai oleh compare_stats_tool (mode COMPARE).

    Return dict berisi:
        film_a, film_b  : dict metadata masing-masing
        winner          : dict berisi pemenang per kategori
    """
    film_a = get_movie_by_title(title_a)
    film_b = get_movie_by_title(title_b)

    if not film_a or not film_b:
        missing = []
        if not film_a:
            missing.append(title_a)
        if not film_b:
            missing.append(title_b)
        return {"error": f"Film tidak ditemukan: {', '.join(missing)}"}

    def winner(a_val, b_val, a_name, b_name, label):
        if a_val is None and b_val is None:
            return {label: "Data tidak tersedia"}
        if a_val is None:
            return {label: b_name}
        if b_val is None:
            return {label: a_name}
        if a_val > b_val:
            return {label: a_name}
        if b_val > a_val:
            return {label: b_name}
        return {label: "Seri"}

    winners = {}
    winners.update(winner(film_a.get("imdb_rating"), film_b.get("imdb_rating"),
                          film_a["title"], film_b["title"], "rating_tertinggi"))
    winners.update(winner(film_a.get("meta_score"),  film_b.get("meta_score"),
                          film_a["title"], film_b["title"], "metascore_tertinggi"))
    winners.update(winner(film_a.get("gross_usd"),   film_b.get("gross_usd"),
                          film_a["title"], film_b["title"], "gross_terbesar"))
    winners.update(winner(film_a.get("no_of_votes"), film_b.get("no_of_votes"),
                          film_a["title"], film_b["title"], "vote_terbanyak"))

    return {
        "film_a":  film_a,
        "film_b":  film_b,
        "winners": winners,
    }
