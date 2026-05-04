# =============================================================
# database/models.py
# Tanggung jawab:
#   - Berisi SQL CREATE TABLE untuk semua tabel project
#   - Dipanggil sekali oleh scripts/init_db.py
#
# Tabel:
#   1. movies         — master data film dari CSV
#   2. debate_sessions — setiap sesi debate/comparison
#   3. debate_history  — riwayat pesan per sesi
# =============================================================

# ------------------------------------------------------------------
# Tabel 1: movies
# Menyimpan metadata lengkap semua film dari CSV.
# Dipakai oleh movie_repo.py untuk query & filter terstruktur.
# ------------------------------------------------------------------
CREATE_MOVIES = """
CREATE TABLE IF NOT EXISTS movies (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    title       VARCHAR(255)   NOT NULL,
    year        SMALLINT,
    genre       VARCHAR(255),
    director    VARCHAR(255),
    star1       VARCHAR(255),
    star2       VARCHAR(255),
    star3       VARCHAR(255),
    star4       VARCHAR(255),
    certificate VARCHAR(50),
    imdb_rating DECIMAL(3,1),
    meta_score  SMALLINT,
    no_of_votes INT,
    runtime_min SMALLINT,
    gross_usd   BIGINT,
    overview    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_title  (title),
    INDEX idx_year   (year),
    INDEX idx_rating (imdb_rating),
    INDEX idx_genre  (genre(50))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ------------------------------------------------------------------
# Tabel 2: debate_sessions
# Satu baris = satu sesi percakapan user dengan agent.
# mode: DEBATE | COMPARE | RECOMMEND
# ------------------------------------------------------------------
CREATE_DEBATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS debate_sessions (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    film_a     VARCHAR(255),
    film_b     VARCHAR(255),
    mode       ENUM('DEBATE', 'COMPARE', 'RECOMMEND') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_mode     (mode),
    INDEX idx_film_a   (film_a),
    INDEX idx_film_b   (film_b),
    INDEX idx_created  (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# ------------------------------------------------------------------
# Tabel 3: debate_history
# Satu baris = satu pesan (user atau assistant) dalam satu sesi.
# role: user | assistant
# ------------------------------------------------------------------
CREATE_DEBATE_HISTORY = """
CREATE TABLE IF NOT EXISTS debate_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT            NOT NULL,
    role       ENUM('user', 'assistant') NOT NULL,
    message    TEXT           NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES debate_sessions(id)
        ON DELETE CASCADE,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# List urutan untuk init_db.py — urut karena ada FK
ALL_TABLES = [
    ("movies",          CREATE_MOVIES),
    ("debate_sessions", CREATE_DEBATE_SESSIONS),
    ("debate_history",  CREATE_DEBATE_HISTORY),
]
