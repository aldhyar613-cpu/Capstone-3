# =============================================================
# imdb_search/config.py
# Konfigurasi project menggunakan Pydantic BaseSettings.
#
# =============================================================

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent

# Inject st.secrets ke os.environ jika di Streamlit Cloud
try:
    import streamlit as st
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
        else:
            os.environ.setdefault(key, str(value))
except Exception:
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        extra             = "ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model:      str = Field("gpt-4o-mini", alias="LLM_MODEL")

    # Qdrant
    qdrant_mode:    str = Field("local",  alias="QDRANT_MODE")
    qdrant_url:     str = Field("",       alias="QDRANT_URL")
    qdrant_api_key: str = Field("",       alias="QDRANT_API_KEY")

    # MySQL
    mysql_host:     str = Field("localhost",    alias="MYSQL_HOST")
    mysql_port:     int = Field(3306,           alias="MYSQL_PORT")
    mysql_user:     str = Field("root",         alias="MYSQL_USER")
    mysql_password: str = Field("",             alias="MYSQL_PASSWORD")
    mysql_database: str = Field("imdb_debate",  alias="MYSQL_DATABASE")

    # LangGraph
    langgraph_db_path: str = Field("debate_history.db", alias="LANGGRAPH_DB_PATH")

    # LangFuse (opsional)
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host:       str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")


# Singleton — import dari modul lain pakai objek ini
settings = Settings()

# ------------------------------------------------------------------
# flat untuk backward-compatibility
# ------------------------------------------------------------------
OPENAI_API_KEY = settings.openai_api_key
LLM_MODEL      = settings.llm_model

QDRANT_MODE    = settings.qdrant_mode
QDRANT_URL     = settings.qdrant_url
QDRANT_API_KEY = settings.qdrant_api_key
QDRANT_PATH    = str(ROOT_DIR / "qdrant_imdb")

MYSQL_HOST     = settings.mysql_host
MYSQL_PORT     = settings.mysql_port
MYSQL_USER     = settings.mysql_user
MYSQL_PASSWORD = settings.mysql_password
MYSQL_DATABASE = settings.mysql_database

LANGGRAPH_DB_PATH = settings.langgraph_db_path

# ------------------------------------------------------------------
# Path constants
# ------------------------------------------------------------------
DATA_PATH        = ROOT_DIR / "data" / "imdb_top_1000.csv"
TOKEN_REPORT_DIR = str(ROOT_DIR / "token_reports")
TOKEN_TXT_PATH   = str(ROOT_DIR / "token_reports" / "token_usage_report.txt")
TOKEN_CSV_PATH   = str(ROOT_DIR / "token_reports" / "token_usage_detail.csv")

# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------
COLLECTION         = "imdb_top1000"
EMBED_MODEL_OPENAI = "text-embedding-3-small"
VECTOR_SIZE        = 1536
BATCH_SIZE         = 64
UPSERT_BATCH       = 200

LANGFUSE_PUBLIC_KEY = settings.langfuse_public_key
LANGFUSE_SECRET_KEY = settings.langfuse_secret_key
LANGFUSE_HOST       = settings.langfuse_host