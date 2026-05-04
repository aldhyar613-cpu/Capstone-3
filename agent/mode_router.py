# =============================================================
# agent/mode_router.py
# Tanggung jawab:
#   - Deteksi mode (DEBATE/COMPARE/RECOMMEND) dari query user
#   - Ekstrak nama film yang disebutkan user
#   - Dipakai oleh node route_mode di agent/graph.py
#
# Perubahan dari versi sebelumnya:
#   - Pakai Pydantic v2 model + with_structured_output()
#   - Tidak ada json.loads() manual → tidak bisa crash karena format salah
# =============================================================

from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from imdb_search.config import OPENAI_API_KEY, LLM_MODEL
from agent.prompts import MODE_DETECTION_PROMPT


# ------------------------------------------------------------------
# Pydantic schema — LLM wajib return format ini
# ------------------------------------------------------------------

class ModeResult(BaseModel):
    """Schema structured output untuk deteksi mode & film."""
    mode: Literal["DEBATE", "COMPARE", "RECOMMEND"] = Field(
        description="Mode percakapan yang paling sesuai dengan query user"
    )
    film_a: str | None = Field(
        default=None,
        description="Judul film pertama yang disebutkan, atau null jika tidak ada"
    )
    film_b: str | None = Field(
        default=None,
        description="Judul film kedua yang disebutkan, atau null jika tidak ada"
    )


# ------------------------------------------------------------------
# LLM dengan structured output — dibuat sekali
# ------------------------------------------------------------------

_llm = ChatOpenAI(
    model       = LLM_MODEL,
    api_key     = OPENAI_API_KEY,
    temperature = 0,
)

_structured_llm = _llm.with_structured_output(ModeResult)


# ------------------------------------------------------------------
# Public function
# ------------------------------------------------------------------

def detect_mode_and_films(query: str) -> dict:
    """
    Deteksi mode dan nama film dari query user menggunakan
    LLM dengan structured output (Pydantic v2).

    Contoh:
        Input : "Bandingkan rating Inception vs Interstellar"
        Output: {"mode": "COMPARE", "film_a": "Inception", "film_b": "Interstellar"}

        Input : "Rekomendasi film thriller psikologis"
        Output: {"mode": "RECOMMEND", "film_a": None, "film_b": None}

    Fallback ke RECOMMEND jika LLM gagal.
    """
    try:
        result: ModeResult = _structured_llm.invoke(
            MODE_DETECTION_PROMPT.format(query=query)
        )
        print(f"[router] Mode: {result.mode} | Film A: {result.film_a} | Film B: {result.film_b}")
        return {
            "mode":   result.mode,
            "film_a": result.film_a,
            "film_b": result.film_b,
        }

    except Exception as e:
        print(f"[router] ⚠ Deteksi mode gagal: {e} → fallback ke RECOMMEND")
        return {"mode": "RECOMMEND", "film_a": None, "film_b": None}
