# =============================================================
# retriever.py
# Tanggung jawab file ini:
#   1. Menerima query teks dari user
#   2. Generate variasi query via LLM (multi-query expansion)
#   3. Mengubah setiap query menjadi vektor via embedder.py
#   4. Mencari dokumen relevan via vector_store.py (per query)
#   5. Menggabungkan hasil dengan Reciprocal Rank Fusion (RRF)
#   6. Memformat hasil menjadi bentuk yang bisa dibaca LangChain
#
# File ini adalah JEMBATAN antara Qdrant dan LangChain.
#
#
# Alur :
#   user query (str)
#     → generate_query_variations()   [LLM hasilkan 3 variasi]
#     → embed_query() x3              [embedder.py, per variasi]
#     → search_vectors() x3           [vector_store.py, per variasi]
#     → reciprocal_rank_fusion()      [gabung & rerank hasil]
#     → Document LangChain            [dikirim ke rag_chain.py]
# =============================================================

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from qdrant_client import QdrantClient
from pydantic import Field
import openai

from imdb_search.config import OPENAI_API_KEY, LLM_MODEL
from imdb_search.embedder import embed_query
from imdb_search.vector_store import search_vectors


# -------------------------------------------------------------
# FORMAT — ubah hasil Qdrant menjadi Document LangChain
# -------------------------------------------------------------

def format_as_document(result: dict) -> Document:
    """
    Mengubah 1 hasil search dari Qdrant menjadi Document LangChain.

    page_content : teks yang akan dibaca oleh LLM
    metadata     : informasi tambahan (tidak dibaca LLM)
    """
    page_content = (
        f"Judul: {result.get('title')} ({result.get('year')})\n"
        f"Genre: {result.get('genre')}\n"
        f"Sutradara: {result.get('director')}\n"
        f"Bintang: {', '.join(result.get('stars', []))}\n"
        f"Rating IMDB: {result.get('imdb_rating')}/10\n"
        f"Sinopsis: {result.get('overview', '-')}"
    )

    metadata = {
        "title":       result.get("title"),
        "year":        result.get("year"),
        "genre":       result.get("genre"),
        "imdb_rating": result.get("imdb_rating"),
        "director":    result.get("director"),
        "score":       result.get("score"),
    }

    return Document(page_content=page_content, metadata=metadata)


# -------------------------------------------------------------
# STEP 1 — Generate Variasi Query via LLM
# -------------------------------------------------------------

def generate_query_variations(query: str, n: int = 3) -> list[str]:
    """
    Menggunakan LLM untuk menghasilkan N variasi query yang berbeda
    dari query asli user.

    Tujuan:
        Query user sering dalam Bahasa Indonesia, sementara teks film
        di Qdrant dalam Bahasa Inggris. Variasi query memastikan
        semantic search bisa menjangkau vocabulary yang berbeda.

    Contoh:
        Input : "film sedih tentang persahabatan sejati"
        Output: [
            "emotional movie about true friendship and loyalty",
            "drama film with deep bond between two characters",
            "sad story about friends supporting each other through hardship"
        ]

    Parameter:
        query : query asli dari user
        n     : jumlah variasi yang dihasilkan (default 3)

    Return:
        list of str — selalu mengembalikan list,
        fallback ke [query] jika LLM gagal
    """
    assert OPENAI_API_KEY, (
        "OPENAI_API_KEY tidak ditemukan! "
        "Pastikan file .env sudah dibuat dan berisi OPENAI_API_KEY=sk-..."
    )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are an expert at generating search query variations for a movie database. "
        "Your task is to generate alternative phrasings of a movie search query. "
        "The movie database contains English text, so always generate variations in English. "
        "Focus on capturing the same intent with different vocabulary and phrasing. "
        f"Generate exactly {n} variations, one per line, no numbering, no explanation."
    )

    user_prompt = (
        f"Original query: {query}\n\n"
        f"Generate {n} English search query variations that capture the same movie search intent. "
        "Use different vocabulary, synonyms, and phrasings. Output only the queries, one per line."
    )

    try:
        resp = client.chat.completions.create(
            model       = LLM_MODEL,
            temperature = 0.7,   # lebih kreatif untuk variasi yang beragam
            max_tokens  = 200,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        raw      = resp.choices[0].message.content.strip()
        # Pisahkan per baris, buang yang kosong
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        variants = variants[:n]   # pastikan tidak lebih dari n

        print(f"[multi-query] Variasi yang dihasilkan:")
        for i, v in enumerate(variants, 1):
            print(f"  {i}. {v}")

        return variants if variants else [query]

    except Exception as e:
        # Jika LLM gagal, fallback ke query asli agar sistem tetap berjalan
        print(f"[multi-query] ⚠ Gagal generate variasi: {e}")
        print(f"[multi-query] Fallback ke query asli.")
        return [query]


# -------------------------------------------------------------
# STEP 2 — Reciprocal Rank Fusion (RRF)
# -------------------------------------------------------------

def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
    top_n: int = 5,
) -> list[dict]:
    """
    Menggabungkan beberapa list hasil search menjadi satu ranking final
    menggunakan algoritma Reciprocal Rank Fusion (RRF).

    Cara kerja RRF:
        Setiap dokumen mendapat score = 1 / (k + rank)
        di setiap list hasil. Score dijumlahkan lintas semua list.
        Dokumen yang konsisten muncul di posisi atas = score tinggi.

    Kenapa RRF dan bukan rata-rata score biasa?
        - Score cosine similarity dari query yang berbeda tidak
          bisa dibandingkan langsung (skala berbeda)
        - RRF hanya melihat POSISI (rank), bukan nilai score
        - Lebih robust terhadap outlier
        - Sederhana tapi terbukti efektif di penelitian IR

    Contoh dengan k=60:
        Film A muncul di rank 1, 2, 3 dari 3 query berbeda:
            score = 1/(60+1) + 1/(60+2) + 1/(60+3) = 0.0492

        Film B muncul di rank 1 dari 1 query saja:
            score = 1/(60+1) = 0.0164

        → Film A menang karena konsisten relevan

    Parameter:
        results_list : list of list[dict], hasil dari beberapa search
        k            : konstanta RRF (default 60, rekomendasi paper asli)
        top_n        : berapa dokumen teratas yang dikembalikan

    Return:
        list of dict — dokumen terurut by RRF score, sudah deduplikasi
    """
    rrf_scores: dict[str, float] = {}   # title → accumulated RRF score
    doc_store:  dict[str, dict]  = {}   # title → data dokumen lengkap

    for results in results_list:
        for rank, doc in enumerate(results):
            title = doc.get("title", "")
            if not title:
                continue

            # Simpan data dokumen (cukup sekali)
            if title not in doc_store:
                doc_store[title] = doc

            # Akumulasi RRF score
            rrf_scores[title] = rrf_scores.get(title, 0.0) + 1.0 / (k + rank + 1)

    # Urutkan berdasarkan RRF score tertinggi
    sorted_titles = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

    # Ambil top_n dan sisipkan rrf_score ke metadata
    fused = []
    for title in sorted_titles[:top_n]:
        doc = doc_store[title].copy()
        doc["rrf_score"] = round(rrf_scores[title], 6)
        fused.append(doc)

    return fused


# -------------------------------------------------------------
# RETRIEVER — class utama yang dipakai LangChain
# -------------------------------------------------------------

class IMDBRetriever(BaseRetriever):
    """
    Custom retriever untuk LangChain yang mengambil dokumen dari Qdrant.
    Kini dilengkapi dengan multi-query expansion dan RRF.

    Cara pakai di rag_chain.py (tidak berubah dari sebelumnya):
        retriever = IMDBRetriever(client=client, top_k=5)
        chain = build_rag_chain(retriever)

    Parameter baru:
        use_multi_query : aktifkan/nonaktifkan multi-query (default True)
        n_variations    : jumlah variasi query yang digenerate (default 3)

    Parameter lama (tidak berubah):
        client, top_k, genre, min_rating, min_year, max_year, min_metascore
    """

    # ── Fields ───────────────────────────
    client:        QdrantClient = Field(...)
    top_k:         int          = Field(default=5)
    genre:         str          = Field(default=None)
    min_rating:    float        = Field(default=None)
    min_year:      int          = Field(default=None)
    max_year:      int          = Field(default=None)
    min_metascore: int          = Field(default=None)

    # ── Fields  (multi-query) ─────────────────────────────
    use_multi_query: bool = Field(default=True)
    n_variations:    int  = Field(default=3)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Method utama yang dipanggil LangChain setiap kali butuh dokumen.

        Alur baru (dengan multi-query):
            1. Generate variasi query via LLM
            2. Untuk setiap variasi: embed → search Qdrant
            3. Gabungkan semua hasil dengan RRF
            4. Format top hasil → list of Document

        Alur fallback (use_multi_query=False):
            1. Embed query asli
            2. Search Qdrant sekali
            3. Format hasil → list of Document
        """
        print(f"\n[retriever] {'='*50}")
        print(f"[retriever] Query asli : '{query}'")

        # ── Path A: Multi-Query Expansion ─────────────────────
        if self.use_multi_query:
            print(f"[retriever] Mode       : Multi-Query Expansion (n={self.n_variations})")

            # Step 1 — generate variasi
            variations = generate_query_variations(query, n=self.n_variations)

            # Step 2 — search per variasi, kumpulkan semua hasil
            # Ambil top_k * 2 per variasi agar RRF punya cukup kandidat
            all_results: list[list[dict]] = []

            for i, var_query in enumerate(variations, 1):
                print(f"\n[retriever] Search {i}/{len(variations)}: '{var_query}'")
                query_vector = embed_query(var_query)
                results      = search_vectors(
                    client        = self.client,
                    query_vector  = query_vector,
                    top_k         = self.top_k * 2,   # ambil lebih banyak untuk RRF
                    genre         = self.genre,
                    min_rating    = self.min_rating,
                    min_year      = self.min_year,
                    max_year      = self.max_year,
                    min_metascore = self.min_metascore,
                )
                print(f"[retriever] → {len(results)} dokumen ditemukan")
                all_results.append(results)

            # Step 3 — gabungkan dengan RRF
            fused = reciprocal_rank_fusion(
                results_list = all_results,
                top_n        = self.top_k,
            )

            print(f"\n[retriever] RRF selesai — top {len(fused)} film:")
            for i, doc in enumerate(fused, 1):
                print(f"  {i}. {doc['title']} ({doc['year']}) "
                      f"— RRF score: {doc['rrf_score']}")

        # ── Path B: Single Query (fallback ) ───────
        else:
            print(f"[retriever] Mode       : Single Query (multi-query dinonaktifkan)")
            query_vector = embed_query(query)
            fused        = search_vectors(
                client        = self.client,
                query_vector  = query_vector,
                top_k         = self.top_k,
                genre         = self.genre,
                min_rating    = self.min_rating,
                min_year      = self.min_year,
                max_year      = self.max_year,
                min_metascore = self.min_metascore,
            )
            print(f"[retriever] {len(fused)} dokumen ditemukan")

        print(f"[retriever] {'='*50}\n")

        # Step 4 — format ke Document LangChain
        return [format_as_document(r) for r in fused]
