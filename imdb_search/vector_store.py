# =============================================================
# vector_store.py
# Tanggung jawab file ini:
#   1. Membuka & menutup koneksi ke Qdrant (local disk)
#   2. Mengecek apakah collection sudah ada & berisi data
#   3. Membuat collection baru jika belum ada
#   4. Menyimpan (upsert) vektor + metadata ke Qdrant
#
# Tugasnya : kelola koneksi & data di Qdrant.
# =============================================================

import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

# Import semua yang dibutuhkan dari config
from imdb_search.config import (
    QDRANT_PATH,
    QDRANT_MODE,
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION,
    VECTOR_SIZE,
    UPSERT_BATCH,
)


# -------------------------------------------------------------
# KONEKSI — buka & tutup client
# -------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """
    Membuka koneksi ke Qdrant — local disk atau cloud.

    Mode dikontrol oleh env var QDRANT_MODE:
      - "local" (default) : koneksi ke folder disk lokal (QDRANT_PATH)
      - "cloud"           : koneksi ke Qdrant Cloud (QDRANT_URL + QDRANT_API_KEY)

    Untuk cloud, isi di .env:
        QDRANT_MODE=cloud
        QDRANT_URL=https://xxxx.qdrant.io:6333
        QDRANT_API_KEY=your-api-key-here
    """
    if QDRANT_MODE == "cloud":
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_MODE=cloud tapi QDRANT_URL atau QDRANT_API_KEY belum diisi di .env"
            )
        print(f"[qdrant] Konek ke Qdrant Cloud → {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        print(f"[qdrant] Konek ke local disk → {QDRANT_PATH}")
        return QdrantClient(path=QDRANT_PATH)


# -------------------------------------------------------------
# CEK — apakah data sudah ada
# -------------------------------------------------------------

def collection_is_ready(client: QdrantClient) -> bool:
    """
    Cek apakah collection sudah ada DAN sudah berisi data.

    Fungsi ini yang mencegah proses embedding diulang
    setiap kali program dijalankan. Jika sudah ada data,
    scripts/load_qdrant.py akan skip langsung ke search.

    Return:
        True  → data sudah ada, skip embedding
        False → belum ada data, jalankan pipeline lengkap
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        print(f"[qdrant] Collection '{COLLECTION}' belum ada")
        return False

    info  = client.get_collection(COLLECTION)
    count = info.points_count
    print(f"[qdrant] Collection '{COLLECTION}' ditemukan — {count} points")
    return count > 0


# -------------------------------------------------------------
# SETUP — buat collection baru
# -------------------------------------------------------------

def setup_collection(client: QdrantClient):
    """
    Membuat collection baru di Qdrant.
    Dipanggil hanya jika collection_is_ready() mengembalikan False.

    Konfigurasi:
        - Ukuran vektor : VECTOR_SIZE (1536 untuk text-embedding-3-small)
        - Distance      : COSINE (cocok untuk semantic search)
    """
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"[qdrant] Collection '{COLLECTION}' dibuat "
          f"(dim={VECTOR_SIZE}, cosine)")


# -------------------------------------------------------------
# UPSERT — simpan vektor + metadata ke Qdrant
# -------------------------------------------------------------

def upsert_documents(
    client: QdrantClient,
    docs: list[dict],
    vectors: list[list[float]],
):
    """
    Menyimpan semua dokumen + vektor ke Qdrant.

    Setiap point berisi:
        - id     : nomor urut (0, 1, 2, ...)
        - vector : vektor float dari embedder.py
        - payload: semua metadata film (title, year, genre, dsb)
                   kecuali doc_id yang sudah tidak diperlukan

    Proses dilakukan per batch (UPSERT_BATCH dari config)
    agar tidak terlalu berat untuk Google Drive / disk.
    """
    print(f"\n[qdrant] Mulai upsert {len(docs)} points ...")

    # Buat list PointStruct
    points = [
        PointStruct(
            id      = i,
            vector  = vec,
            payload = {k: v for k, v in doc.items() if k != "doc_id"},
        )
        for i, (doc, vec) in enumerate(zip(docs, vectors))
    ]

    # Upsert per batch
    for start in range(0, len(points), UPSERT_BATCH):
        batch = points[start : start + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION, points=batch)
        done = min(start + UPSERT_BATCH, len(points))
        print(f"[qdrant] {done:>4}/{len(points)} points tersimpan")

    # Validasi hasil upsert
    info     = client.get_collection(COLLECTION)
    actual   = info.points_count
    expected = len(docs)

    if actual == expected:
        print(f"[qdrant] ✓ Validasi OK — {actual} points tersimpan di disk")
    else:
        print(f"[qdrant] ⚠ Validasi GAGAL — expected {expected}, actual {actual}")


# -------------------------------------------------------------
# SEARCH — cari vektor yang mirip (dipakai oleh retriever.py)
# -------------------------------------------------------------

def search_vectors(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    genre: str = None,
    min_rating: float = None,
    min_year: int = None,
    max_year: int = None,
    min_metascore: int = None,
) -> list[dict]:
    """
    Mencari dokumen yang paling mirip dengan query_vector di Qdrant.
    Mendukung filter metadata: genre, rating, tahun, metascore.

    Input  : vektor query dari embedder.embed_query()
    Output : list of dict berisi metadata film yang relevan

    Fungsi ini dipanggil oleh retriever.py,
    bukan langsung oleh user atau main.py.
    """
    # Bangun kondisi filter (opsional)
    conditions = []
    if genre:
        conditions.append(
            FieldCondition(key="genre_list", match=MatchValue(value=genre))
        )
    if min_rating:
        conditions.append(
            FieldCondition(key="imdb_rating", range=Range(gte=min_rating))
        )
    yr = {}
    if min_year:
        yr["gte"] = min_year
    if max_year:
        yr["lte"] = max_year
    if yr:
        conditions.append(
            FieldCondition(key="year", range=Range(**yr))
        )
    if min_metascore:
        conditions.append(
            FieldCondition(key="meta_score", range=Range(gte=min_metascore))
        )

    # Query ke Qdrant
    result = client.query_points(
        collection_name = COLLECTION,
        query           = query_vector,
        query_filter    = Filter(must=conditions) if conditions else None,
        limit           = top_k,
        with_payload    = True,
    )

    # Format output
    return [
        {
            "score":       round(h.score, 4),
            "title":       h.payload.get("title"),
            "year":        h.payload.get("year"),
            "genre":       h.payload.get("genre"),
            "imdb_rating": h.payload.get("imdb_rating"),
            "director":    h.payload.get("director"),
            "stars":       h.payload.get("stars", []),
            "overview":    h.payload.get("text", ""),
        }
        for h in result.points
    ]
