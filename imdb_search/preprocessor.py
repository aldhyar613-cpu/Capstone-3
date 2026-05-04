# =============================================================
# imdb_search/preprocessor.py
# Tanggung jawab:
#   1. Membaca & membersihkan CSV
#   2. Membangun teks deskriptif per film
#   3. Sync metadata ke MySQL
#   4. Menghasilkan list of dict siap dikirim ke embedder
# =============================================================

import re
import pandas as pd
from typing import Optional
from pathlib import Path

from imdb_search.config import DATA_PATH


# -------------------------------------------------------------
# HELPER — parsing kolom individual
# -------------------------------------------------------------

def parse_runtime(val) -> Optional[int]:
    if pd.isna(val):
        return None
    match = re.search(r"(\d+)", str(val))
    return int(match.group(1)) if match else None


def parse_gross(val) -> Optional[float]:
    if pd.isna(val):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_year(val) -> Optional[int]:
    match = re.search(r"(\d{4})", str(val))
    return int(match.group(1)) if match else None


# -------------------------------------------------------------
# STEP 1 — Load & Clean
# -------------------------------------------------------------

def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  {len(df)} film dimuat dari {path}")
    df = df.copy()

    df.drop(columns=["Poster_Link"], inplace=True, errors="ignore")

    df["Runtime_Minutes"] = df["Runtime"].apply(parse_runtime)
    df["Gross_USD"]       = df["Gross"].apply(parse_gross)
    df["Year"]            = df["Released_Year"].apply(parse_year)

    df.drop(columns=["Runtime", "Released_Year", "Gross"], inplace=True)

    df["Certificate"] = df["Certificate"].fillna("Not Rated").str.strip()
    df["Meta_score"]  = df["Meta_score"].fillna(-1).astype(int)

    for col in ["Series_Title", "Genre", "Director",
                "Star1", "Star2", "Star3", "Star4", "Overview"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print(f"[clean] OK — {len(df.columns)} kolom tersisa")
    return df


# -------------------------------------------------------------
# STEP 2 — Build Teks Deskriptif
# -------------------------------------------------------------

def build_descriptive_text(row: pd.Series) -> str:
    parts = []

    year_str = f" ({int(row['Year'])})" if pd.notna(row.get("Year")) and row.get("Year") else ""
    parts.append(f"{row['Series_Title']}{year_str} — {row['Genre']}.")
    parts.append(f"Directed by {row['Director']}.")

    stars = [row[s] for s in ["Star1", "Star2", "Star3", "Star4"]
             if pd.notna(row.get(s)) and str(row.get(s, "")).strip() not in ("", "nan")]
    parts.append(f"Stars: {', '.join(stars)}.")

    rt = f"Runtime: {row['Runtime_Minutes']} minutes. " if row.get("Runtime_Minutes") else ""
    parts.append(f"{rt}Certificate: {row['Certificate']}.")

    ms = f" Metascore: {row['Meta_score']}." if row.get("Meta_score", -1) != -1 else ""
    vf = f"{int(row['No_of_Votes']):,}" if pd.notna(row.get("No_of_Votes")) else "N/A"
    parts.append(f"IMDB Rating: {row['IMDB_Rating']}/10 ({vf} votes).{ms}")

    if row.get("Gross_USD"):
        parts.append(f"Box office gross: ${row['Gross_USD']:,.0f}.")

    parts.append(f"Overview: {row['Overview']}")
    return " ".join(parts)


# -------------------------------------------------------------
# STEP 3 — Build Documents + Sync MySQL (BARU)
# -------------------------------------------------------------

def build_documents(df: pd.DataFrame, sync_mysql: bool = True) -> list[dict]:
    """
    Mengubah setiap baris DataFrame menjadi 1 document dict.
    Jika sync_mysql=True, metadata juga disimpan ke MySQL.
    """
    docs = []
    for idx, row in df.iterrows():
        doc = {
            "doc_id":      f"movie_{idx}",
            "text":        build_descriptive_text(row),
            "title":       row["Series_Title"],
            "year":        int(row["Year"]) if pd.notna(row.get("Year")) and row.get("Year") else None,
            "genre":       row["Genre"],
            "genre_list":  [g.strip() for g in row["Genre"].split(",")],
            "director":    row["Director"],
            "stars":       [row[s] for s in ["Star1", "Star2", "Star3", "Star4"]
                            if pd.notna(row.get(s)) and str(row.get(s, "")).strip() not in ("", "nan")],
            "certificate": row["Certificate"],
            "imdb_rating": float(row["IMDB_Rating"]),
            "meta_score":  int(row.get("Meta_score", -1)),
            "no_of_votes": int(row["No_of_Votes"]) if pd.notna(row.get("No_of_Votes")) else 0,
            "runtime_min": row.get("Runtime_Minutes"),
            "gross_usd":   row.get("Gross_USD"),
            "overview":    row["Overview"],
        }
        docs.append(doc)

    # Sync ke MySQL jika diminta
    if sync_mysql:
        try:
            from database.movie_repo import bulk_insert_movies
            bulk_insert_movies(docs)
            print(f"[mysql] {len(docs)} film disync ke MySQL")
        except Exception as e:
            print(f"[mysql] ⚠ Sync gagal (MySQL mungkin belum diinit): {e}")
            print("[mysql] Lanjutkan tanpa MySQL sync...")

    avg = sum(len(d["text"]) for d in docs) / len(docs)
    print(f"[docs]  {len(docs)} dokumen dibuat — rata-rata teks: {avg:.0f} karakter")
    return docs
