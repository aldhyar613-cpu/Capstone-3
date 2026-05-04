# =============================================================
# scripts/load_qdrant.py
# Pipeline lengkap: CSV → MySQL + Qdrant
#
# Cara pakai (jalankan setelah init_db.py):
#   cd imdb-debate
#   python -m scripts.load_qdrant
# =============================================================

from imdb_search.preprocessor import load_and_clean, build_documents
from imdb_search.embedder import embed_texts, save_token_report
from imdb_search.vector_store import (
    get_qdrant_client,
    collection_is_ready,
    setup_collection,
    upsert_documents,
)


def main():
    print("=" * 60)
    print("  IMDB Debate — Load Data ke Qdrant + MySQL")
    print("=" * 60)

    client = get_qdrant_client()

    if collection_is_ready(client):
        print("\n[load] Collection Qdrant sudah ada — skip embedding")
        print("[load] Tip: Hapus folder qdrant_imdb/ untuk reload dari awal")
        client.close()
        return

    # Step 1 — Baca & bersihkan CSV, sync ke MySQL otomatis
    df   = load_and_clean()
    docs = build_documents(df, sync_mysql=True)

    # Step 2 — Embed teks
    texts   = [d["text"] for d in docs]
    vectors = embed_texts(texts)

    # Step 3 — Setup collection & upsert ke Qdrant
    setup_collection(client)
    upsert_documents(client, docs, vectors)

    # Step 4 — Simpan laporan token
    save_token_report()

    client.close()
    print("\n✅ Pipeline selesai! Jalankan: streamlit run app.py")


if __name__ == "__main__":
    main()
