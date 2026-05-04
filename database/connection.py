# =============================================================
# database/connection.py
# Fungsi untuk:
#   - Membuka koneksi ke MySQL
#   - Menyediakan context manager agar koneksi selalu ditutup
#   - Dipakai oleh semua file lain di folder database/
# =============================================================

import mysql.connector
from contextlib import contextmanager

from imdb_search.config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)


def get_connection():
    """
    Membuka koneksi baru ke MySQL.
    Selalu panggil .close() setelah selesai,
    atau gunakan context manager get_cursor() di bawah.
    """
    return mysql.connector.connect(
        host     = MYSQL_HOST,
        port     = MYSQL_PORT,
        user     = MYSQL_USER,
        password = MYSQL_PASSWORD,
        database = MYSQL_DATABASE,
        charset  = "utf8mb4",
    )


@contextmanager
def get_cursor(dictionary: bool = True):
    """
    Context manager yang otomatis commit & tutup koneksi.

    Cara pakai:
        with get_cursor() as (conn, cur):
            cur.execute("SELECT ...")
            rows = cur.fetchall()

    dictionary=True → hasil fetchall() berupa list of dict (default)
    dictionary=False → hasil berupa list of tuple
    """
    conn = get_connection()
    cur  = conn.cursor(dictionary=dictionary)
    try:
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()
