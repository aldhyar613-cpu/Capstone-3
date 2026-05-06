# =============================================================
# vector_store.py
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
from imdb_search.config import COLLECTION, VECTOR_SIZE, UPSERT_BATCH


def get_qdrant_client() -> QdrantClient:
    from imdb_search.config import QDRANT_MODE, QDRANT_URL, QDRANT_API_KEY, QDRANT_PATH

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


def collection_is_ready(client: QdrantClient) -> bool:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        print(f"[qdrant] Collection '{COLLECTION}' belum ada")
        return False
    info  = client.get_collection(COLLECTION)
    count = info.points_count
    print(f"[qdrant] Collection '{COLLECTION}' ditemukan — {count} points")
    return count > 0


def setup_collection(client: QdrantClient):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[qdrant] Collection '{COLLECTION}' dibuat (dim={VECTOR_SIZE}, cosine)")


def upsert_documents(client: QdrantClient, docs: list[dict], vectors: list[list[float]]):
    print(f"\n[qdrant] Mulai upsert {len(docs)} points ...")
    points = [
        PointStruct(
            id      = i,
            vector  = vec,
            payload = {k: v for k, v in doc.items() if k != "doc_id"},
        )
        for i, (doc, vec) in enumerate(zip(docs, vectors))
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        batch = points[start : start + UPSERT_BATCH]
        client.upsert(collection_name=COLLECTION, points=batch)
        done = min(start + UPSERT_BATCH, len(points))
        print(f"[qdrant] {done:>4}/{len(points)} points tersimpan")
    info = client.get_collection(COLLECTION)
    actual = info.points_count
    if actual == len(docs):
        print(f"[qdrant] ✓ Validasi OK — {actual} points tersimpan")
    else:
        print(f"[qdrant] ⚠ Validasi GAGAL — expected {len(docs)}, actual {actual}")


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
    conditions = []
    if genre:
        conditions.append(FieldCondition(key="genre_list", match=MatchValue(value=genre)))
    if min_rating:
        conditions.append(FieldCondition(key="imdb_rating", range=Range(gte=min_rating)))
    yr = {}
    if min_year:
        yr["gte"] = min_year
    if max_year:
        yr["lte"] = max_year
    if yr:
        conditions.append(FieldCondition(key="year", range=Range(**yr)))
    if min_metascore:
        conditions.append(FieldCondition(key="meta_score", range=Range(gte=min_metascore)))

    result = client.query_points(
        collection_name = COLLECTION,
        query           = query_vector,
        query_filter    = Filter(must=conditions) if conditions else None,
        limit           = top_k,
        with_payload    = True,
    )
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