# =============================================================
# scripts/init_db.py
# Jalankan SEKALI untuk:
#   1. Membuat database 'imdb_debate' jika belum ada
#   2. Membuat semua tabel (movies, debate_sessions, debate_history)
#
# Cara pakai:
#   cd imdb-debate
#   python -m scripts.init_db
# =============================================================

import mysql.connector
from imdb_search.config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
)
from database.models import ALL_TABLES


def create_database_if_not_exists():
    """
    Buat database jika belum ada.
    Koneksi tanpa database= karena database belum ada.
    """
    conn = mysql.connector.connect(
        host     = MYSQL_HOST,
        port     = MYSQL_PORT,
        user     = MYSQL_USER,
        password = MYSQL_PASSWORD,
        charset  = "utf8mb4",
    )
    cur = conn.cursor()
    cur.execute(
        f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
        f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"[init_db] Database '{MYSQL_DATABASE}' siap")


def create_all_tables():
    """
    Buat semua tabel dari models.py.
    IF NOT EXISTS — aman dijalankan berkali-kali.
    """
    conn = mysql.connector.connect(
        host     = MYSQL_HOST,
        port     = MYSQL_PORT,
        user     = MYSQL_USER,
        password = MYSQL_PASSWORD,
        database = MYSQL_DATABASE,
        charset  = "utf8mb4",
    )
    cur = conn.cursor()

    for table_name, sql in ALL_TABLES:
        cur.execute(sql)
        print(f"[init_db] Tabel '{table_name}' siap")

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    print("=" * 50)
    print("  IMDB Debate — Inisialisasi MySQL")
    print("=" * 50)

    try:
        create_database_if_not_exists()
        create_all_tables()
        print("\n✅ MySQL berhasil diinisialisasi!")
        print("   Langkah berikutnya: python -m scripts.load_qdrant")
    except Exception as e:
        print(f"\n❌ Gagal: {e}")
        print("   Pastikan MySQL running dan kredensial di .env sudah benar.")
