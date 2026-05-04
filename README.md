<<<<<<< HEAD
# 🎬 IMDB Movie Debate & Comparison

> AI chatbot untuk debat, perbandingan, dan rekomendasi film dari dataset IMDB Top 1000 — dibangun dengan **LangGraph**, **Qdrant**, dan **MySQL**.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green?logo=langchain)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![MySQL](https://img.shields.io/badge/DB-MySQL-orange?logo=mysql)

---

## Demo

> Ketik query → agent otomatis deteksi mode → panggil tools → jawab

| Mode | Contoh Query |
|------|-------------|
| 🔴 **DEBATE** | *"Debatkan Inception vs Interstellar, mana yang lebih baik?"* |
| 🔵 **COMPARE** | *"Bandingkan data The Dark Knight vs Joker"* |
| 🟢 **RECOMMEND** | *"Rekomendasikan film thriller psikologis rating di atas 8.5"* |

Mode terdeteksi **otomatis** dari kalimat user — tidak perlu pilih manual.

---

## Arsitektur

```
User Query
    │
    ▼
Streamlit UI (app.py)
    │
    ▼
┌─────────────────────────────────────────────┐
│           LangGraph StateGraph              │
│                                             │
│  route_mode ──► generate ◄──► call_tools   │
│      │              │                       │
│  (detect mode)  (LLM answer)  (6 tools)    │
│                                             │
│  SqliteSaver checkpointer (state persist)  │
└─────────────────────────────────────────────┘
    │                    │
    ▼                    ▼
 Qdrant              MySQL
(semantic search)  (structured query
                    + session history)
```

### Komponen Utama

| File | Peran |
|------|-------|
| `agent/graph.py` | LangGraph StateGraph — orchestration utama |
| `agent/state.py` | `DebateState` TypedDict — state antar node |
| `agent/tools.py` | 6 tools (Qdrant + MySQL) |
| `agent/mode_router.py` | Deteksi mode via structured output Pydantic |
| `agent/prompts.py` | System prompt 3 mode |
| `imdb_search/` | RAG pipeline: embed → store → retrieve |
| `database/` | MySQL CRUD: film, sesi, history |

---

## Tech Stack

- **LangGraph** — StateGraph dengan checkpointer untuk multi-turn conversation
- **LangChain** — tools, LLM wrapper, structured output
- **OpenAI** — LLM (`gpt-4o-mini`) + Embedding (`text-embedding-3-small`)
- **Qdrant** — vector store untuk semantic search film
- **MySQL** — penyimpanan film terstruktur + history sesi
- **Streamlit** — UI chat

---

## Setup

### 1. Install dependencies

```bash
cd imdb-debate
pip install -r requirements.txt --upgrade
```

### 2. Konfigurasi environment

```bash
cp .env.example .env
# sesudah copy example lalu edit .env — isi OPENAI_API_KEY, MYSQL_PASSWORD, QDRANT_URL
```

### 3. Inisialisasi database (jalankan sekali)

```bash
python -m scripts.init_db
```

### 4. Load data ke Qdrant + MySQL (jalankan sekali)

```bash
python -m scripts.load_qdrant
```

### 5. Jalankan aplikasi

```bash
streamlit run app.py
```

---

## Struktur Folder

```
imdb-debate/
├── agent/
│   ├── graph.py          ← LangGraph StateGraph (BARU)
│   ├── state.py          ← DebateState TypedDict (BARU)
│   ├── debate_agent.py   ← entry point, run_agent()
│   ├── tools.py          ← 6 tools (Qdrant + MySQL)
│   ├── prompts.py        ← system prompt 3 mode
│   └── mode_router.py    ← deteksi mode (Pydantic structured output)
├── database/
│   ├── connection.py     ← koneksi MySQL
│   ├── models.py         ← CREATE TABLE SQL
│   ├── movie_repo.py     ← CRUD film
│   └── session_repo.py   ← CRUD sesi & history
├── imdb_search/
│   ├── config.py         ← Pydantic BaseSettings
│   ├── embedder.py       ← teks → vektor OpenAI
│   ├── vector_store.py   ← CRUD Qdrant
│   ├── retriever.py      ← multi-query + RRF
│   └── preprocessor.py   ← baca CSV + sync MySQL
├── tests/
│   └── test_agent.py     ← unit test + integration test
├── scripts/
│   ├── init_db.py        ← setup MySQL
│   └── load_qdrant.py    ← load CSV → Qdrant + MySQL
├── data/
│   └── imdb_top_1000.csv
├── app.py                ← Streamlit UI
├── .env.example
└── requirements.txt
```

---

## Tools Agent

| Tool | Sumber Data | Fungsi |
|------|-------------|--------|
| `search_movies_tool` | Qdrant | Semantic search berdasarkan deskripsi/tema |
| `get_movie_details_tool` | MySQL | Detail lengkap satu film |
| `compare_stats_tool` | MySQL | Head-to-head statistik dua film |
| `filter_movies_tool` | MySQL | Filter multi-kriteria (genre, rating, tahun) |
| `save_debate_history_tool` | MySQL | Simpan pesan ke history |
| `get_history_tool` | MySQL | Ambil riwayat sesi |

---

## Menjalankan Test

```bash
# Unit test saja (tidak perlu koneksi eksternal)
pytest tests/ -v -m "not integration"

# Semua test termasuk integration (butuh MySQL + Qdrant + OpenAI)
pytest tests/ -v
```
=======
# Capsone-3
>>>>>>> e18e2c4e924abea3b58f81d687921832cc04c8f3
