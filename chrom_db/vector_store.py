import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from chrom_db.config import OPENAI_API_KEY, VECTOR_DIR, CONV_FILE

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings,
)

def load_initial_conversations() -> None:
    if os.path.exists(CONV_FILE):
        with open(CONV_FILE, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("[")]
        if lines:
            vector_store.add_texts(lines)

load_initial_conversations()  # se ejecuta al importar
