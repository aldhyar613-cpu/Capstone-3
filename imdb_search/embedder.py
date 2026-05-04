# =============================================================
# embedder.py
# Tanggung jawab file ini:
#   1. Menerima list teks dari preprocessor.py
#   2. Mengirim teks ke OpenAI API per batch
#   3. Mengembalikan list vektor siap disimpan ke Qdrant
#   4. Mencatat pemakaian token ke laporan
#
# Tugasnya : teks masuk → vektor keluar.
# =============================================================

import os
import csv
import numpy as np
import openai
from datetime import datetime
from typing import Optional

# Import semua yang dibutuhkan dari config
from imdb_search.config import (
    OPENAI_API_KEY,
    EMBED_MODEL_OPENAI,
    BATCH_SIZE,
    TOKEN_REPORT_DIR,
    TOKEN_TXT_PATH,
    TOKEN_CSV_PATH,
)


# -------------------------------------------------------------
# TOKEN TRACKER
# Mencatat pemakaian token setiap kali ada request ke OpenAI.
# Dipakai untuk monitoring biaya API.
# -------------------------------------------------------------

_token_log: list[dict] = []


def _log_token(jenis: str, deskripsi: str, prompt_tokens: int, total_tokens: int):
    """
    Catat satu transaksi token ke log global.
    Dipanggil otomatis setiap kali ada response dari OpenAI.

    jenis     : "embedding" atau "search"
    deskripsi : keterangan batch / teks query
    """
    _token_log.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "jenis":         jenis,
        "deskripsi":     deskripsi,
        "prompt_tokens": prompt_tokens,
        "total_tokens":  total_tokens,
    })


# -------------------------------------------------------------
# STEP 1 — Embed Teks via OpenAI
# -------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Mengubah list teks menjadi list vektor float menggunakan OpenAI API.

    Proses:
        1. Kirim teks per batch ke OpenAI (BATCH_SIZE dari config)
        2. Terima vektor kembali dari OpenAI
        3. Normalisasi vektor (penting untuk cosine similarity di Qdrant)
        4. Catat token yang dipakai ke _token_log

    Input  : list of str  (teks deskriptif dari preprocessor)
    Output : list of list[float]  (vektor siap upsert ke Qdrant)
    """
    assert OPENAI_API_KEY, (
        "OPENAI_API_KEY tidak ditemukan! "
        "Pastikan file .env sudah dibuat dan berisi OPENAI_API_KEY=sk-..."
    )

    client     = openai.OpenAI(api_key=OPENAI_API_KEY)
    all_vecs   = []
    total_batch = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"[embed] Mulai embedding {len(texts)} teks via OpenAI ...")
    print(f"[embed] Model : {EMBED_MODEL_OPENAI}")
    print(f"[embed] Batch : {BATCH_SIZE} teks/request → total {total_batch} batch")
    print(f"[embed] {'-'*50}")

    for i in range(0, len(texts), BATCH_SIZE):
        batch     = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        # Kirim ke OpenAI, terima vektor
        resp = client.embeddings.create(
            model=EMBED_MODEL_OPENAI,
            input=batch,
        )
        all_vecs.extend([item.embedding for item in resp.data])

        # Catat token
        _log_token(
            jenis         = "embedding",
            deskripsi     = f"batch {batch_num}/{total_batch} ({len(batch)} film)",
            prompt_tokens = resp.usage.prompt_tokens,
            total_tokens  = resp.usage.total_tokens,
        )

        # Progress di terminal
        done = min(i + BATCH_SIZE, len(texts))
        print(f"[embed] {done:>4}/{len(texts)} film "
              f"— batch {batch_num}/{total_batch} "
              f"— token: {resp.usage.total_tokens:,}")

    # Normalisasi vektor — wajib untuk cosine similarity
    arr = np.array(all_vecs)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)

    print(f"[embed] {'-'*50}")
    print(f"[embed] Selesai — shape vektor: {arr.shape}")
    return arr.tolist()


# -------------------------------------------------------------
# STEP 2 — Embed Query (dipakai saat search, bukan saat load)
# -------------------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Mengubah 1 teks query menjadi vektor untuk semantic search.
    Dipanggil oleh retriever.py setiap kali user melakukan pencarian.

    Berbeda dari embed_texts() yang memproses ribuan teks sekaligus,
    fungsi ini hanya memproses 1 teks per pemanggilan.
    """
    assert OPENAI_API_KEY, (
        "OPENAI_API_KEY tidak ditemukan! "
        "Pastikan file .env sudah dibuat dan berisi OPENAI_API_KEY=sk-..."
    )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp   = client.embeddings.create(
        model=EMBED_MODEL_OPENAI,
        input=[query],
    )

    # Catat token query
    _log_token(
        jenis         = "search",
        deskripsi     = f"query: {query[:60]}{'...' if len(query) > 60 else ''}",
        prompt_tokens = resp.usage.prompt_tokens,
        total_tokens  = resp.usage.total_tokens,
    )

    # Normalisasi
    vec = np.array(resp.data[0].embedding)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# -------------------------------------------------------------
# STEP 3 — Simpan Laporan Token
# -------------------------------------------------------------

def save_token_report():
    """
    Simpan semua catatan token ke 2 file di TOKEN_REPORT_DIR:
        - token_usage_report.txt  : ringkasan mudah dibaca
        - token_usage_detail.csv  : detail per transaksi untuk analisis
    
    Dipanggil di akhir pipeline oleh scripts/load_qdrant.py
    """
    if not _token_log:
        print("[token] Tidak ada data token yang tercatat.")
        return

    os.makedirs(TOKEN_REPORT_DIR, exist_ok=True)

    embed_logs  = [x for x in _token_log if x["jenis"] == "embedding"]
    search_logs = [x for x in _token_log if x["jenis"] == "search"]

    embed_total  = sum(x["total_tokens"] for x in embed_logs)
    search_total = sum(x["total_tokens"] for x in search_logs)
    grand_total  = embed_total + search_total
    sesi_waktu   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Print ringkasan di terminal ───────────────────────────
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  TOKEN USAGE REPORT")
    print(f"{sep}")
    print(f"  Sesi  : {sesi_waktu}")
    print(f"  Model : {EMBED_MODEL_OPENAI}")
    print(f"{sep}")

    if embed_logs:
        print(f"\n  [EMBEDDING] — {len(embed_logs)} batch")
        for e in embed_logs:
            print(f"    {e['deskripsi']:<40} {e['total_tokens']:>8,} token")
        print(f"    {'SUBTOTAL EMBEDDING':<40} {embed_total:>8,} token")

    if search_logs:
        print(f"\n  [SEARCH] — {len(search_logs)} query")
        for s in search_logs:
            print(f"    {s['deskripsi']:<40} {s['total_tokens']:>8,} token")
        print(f"    {'SUBTOTAL SEARCH':<40} {search_total:>8,} token")

    print(f"\n  {'GRAND TOTAL':<40} {grand_total:>8,} token")
    print(f"{sep}\n")

    # ── Simpan ke .txt ────────────────────────────────────────
    with open(TOKEN_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(f"{'='*62}\n  TOKEN USAGE REPORT\n{'='*62}\n")
        f.write(f"  Sesi  : {sesi_waktu}\n")
        f.write(f"  Model : {EMBED_MODEL_OPENAI}\n{'='*62}\n\n")
        if embed_logs:
            f.write(f"  [EMBEDDING] — {len(embed_logs)} batch\n")
            for e in embed_logs:
                f.write(f"    {e['timestamp']}  {e['deskripsi']:<40} "
                        f"{e['total_tokens']:>8,} token\n")
            f.write(f"    {'SUBTOTAL EMBEDDING':<55} {embed_total:>8,} token\n\n")
        if search_logs:
            f.write(f"  [SEARCH] — {len(search_logs)} query\n")
            for s in search_logs:
                f.write(f"    {s['timestamp']}  {s['deskripsi']:<40} "
                        f"{s['total_tokens']:>8,} token\n")
            f.write(f"    {'SUBTOTAL SEARCH':<55} {search_total:>8,} token\n\n")
        f.write(f"  {'GRAND TOTAL':<55} {grand_total:>8,} token\n{'='*62}\n")

    print(f"[token] Laporan .txt → {TOKEN_TXT_PATH}")

    # ── Simpan ke .csv ────────────────────────────────────────
    with open(TOKEN_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "jenis", "deskripsi", "prompt_tokens", "total_tokens"
        ])
        writer.writeheader()
        writer.writerows(_token_log)

    print(f"[token] Detail .csv  → {TOKEN_CSV_PATH}")
