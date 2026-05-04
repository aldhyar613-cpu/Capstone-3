# =============================================================
# rag_chain.py
# Tanggung jawab file ini:
#   1. Membangun prompt template untuk chatbot IMDB
#   2. Menghubungkan retriever → prompt → LLM (LCEL chain)
#   3. Mengelola conversation memory (riwayat chat)
#   4. Menyisipkan LangFuse callback untuk observability
#
# Alur per pesan:
#   user query
#     → retriever (ambil film relevan dari Qdrant)
#     → prompt (gabungkan konteks + history + query)
#     → LLM (generate jawaban)
#     → LangFuse (catat trace otomatis)
# =============================================================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

#from langfuse.callback import CallbackHandler

from imdb_search.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
)


# -------------------------------------------------------------
# PROMPT TEMPLATE
# Dibatasi hanya menjawab dari konteks film yang di-retrieve.
# Mendukung riwayat percakapan via MessagesPlaceholder.
# -------------------------------------------------------------

SYSTEM_PROMPT = """Kamu adalah asisten pencari film IMDB yang ramah dan berpengetahuan luas.
Jawab pertanyaan user HANYA berdasarkan daftar film berikut yang relevan dengan pertanyaannya.

Panduan menjawab:
- Jika ada beberapa film relevan, bandingkan dan berikan rekomendasi terbaik
- Sertakan rating IMDB, tahun, genre, sutradara, dan bintang utama jika relevan
- Jika user menanya hal di luar konteks film yang tersedia, katakan dengan jujur bahwa kamu tidak menemukan film yang sesuai
- Gunakan bahasa yang sama dengan user (Indonesia atau Inggris)
- Jawab dengan ringkas tapi informatif

Konteks film yang tersedia:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# -------------------------------------------------------------
# FORMAT DOKUMEN
# Mengubah list Document LangChain menjadi 1 string konteks
# untuk disisipkan ke prompt.
# -------------------------------------------------------------

def format_docs(docs: list[Document]) -> str:
    """
    Menggabungkan semua dokumen yang di-retrieve menjadi
    satu blok teks yang mudah dibaca LLM.

    Setiap film dipisahkan oleh garis pemisah agar LLM
    tidak mencampur informasi antar film.
    """
    if not docs:
        return "Tidak ada film yang ditemukan untuk query ini."

    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Film {i}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# -------------------------------------------------------------
# LANGFUSE HANDLER
# Dibuat sekali, dipakai ulang di setiap pemanggilan chain.
# Otomatis mengirim trace ke LangFuse dashboard.
# -------------------------------------------------------------

def get_langfuse_handler() -> CallbackHandler | None:
    """
    Membuat LangFuse CallbackHandler jika key tersedia.
    Mengembalikan None jika key belum dikonfigurasi,
    sehingga chain tetap berjalan tanpa observability.
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        print("[langfuse] ⚠ Key tidak ditemukan — tracing dinonaktifkan")
        return None

    handler = CallbackHandler(
        public_key  = LANGFUSE_PUBLIC_KEY,
        secret_key  = LANGFUSE_SECRET_KEY,
        host        = LANGFUSE_HOST,
    )
    print(f"[langfuse] ✓ Tracing aktif → {LANGFUSE_HOST}")
    return handler


# -------------------------------------------------------------
# BUILD CHAIN
# Fungsi utama yang dipanggil oleh main.py dan app.py.
# Mengembalikan chain siap pakai + langfuse handler.
# -------------------------------------------------------------

def build_rag_chain(retriever):
    """
    Membangun LCEL chain lengkap.

    Parameter:
        retriever : IMDBRetriever yang sudah diinisialisasi
                    (dengan client, top_k, dan filter opsional)

    Return:
        chain          : LCEL chain siap dipanggil dengan .invoke()
        langfuse_handler : CallbackHandler untuk dikirim ke config callbacks
                           (None jika key tidak tersedia)

    Cara pakai di main.py / app.py:
        chain, lf_handler = build_rag_chain(retriever)
        callbacks = [lf_handler] if lf_handler else []
        response = chain.invoke(
            {"question": query, "chat_history": history},
            config={"callbacks": callbacks}
        )
    """
    # LLM
    llm = ChatOpenAI(
        model       = LLM_MODEL,
        api_key     = OPENAI_API_KEY,
        temperature = 0.3,   # sedikit kreatif tapi tetap faktual
        streaming   = False,
    )

    # Chain LCEL
    # RunnablePassthrough memastikan 'question' dan 'chat_history'
    # tetap tersedia setelah retriever menambahkan 'context'.
    chain = (
        RunnablePassthrough.assign(
            context = RunnableLambda(
                lambda x: format_docs(retriever.invoke(x["question"]))
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    langfuse_handler = get_langfuse_handler()

    return chain, langfuse_handler


# -------------------------------------------------------------
# HISTORY HELPER
# Utilitas untuk mengkonversi format history yang disimpan
# sebagai list of dict (mudah di-serialize) menjadi
# list of BaseMessage (format LangChain).
# -------------------------------------------------------------

def build_chat_history(history: list[dict]) -> list:
    """
    Mengubah riwayat chat dari format sederhana ke format LangChain.

    Format input (disimpan di session / main.py):
        [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
        ]

    Format output (dibutuhkan MessagesPlaceholder):
        [HumanMessage(...), AIMessage(...), ...]
    """
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
