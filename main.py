# main.py – FastAPI + Twilio + RAG + persistencia de interacciones
# ───────────────────────────────────────────────────────────────────
"""
Arquitectura en un solo archivo para fines didácticos.  En producción
conviene separar en paquetes (db.py, persistence.py, rag.py, etc.).

Requisitos (pip install ...):
  fastapi uvicorn python-dotenv sqlalchemy psycopg2-binary
  twilio langchain-openai langchain-chroma
"""

import os
import re
from contextlib import contextmanager
from datetime import datetime
from fastapi import FastAPI, Form, Response, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from sqlalchemy import create_engine, text
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# ────── CONFIGURACIÓN ──────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
DB_URL          = os.getenv("DB_URL")           # Ej: 'postgresql://user:pass@host/db'
ACCOUNT_SID     = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER     = os.getenv("TWILIO_FROM_NUMBER")

# ────── APP BASE ──────
app = FastAPI()
router = APIRouter()
twilio = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# La función session() abre una conexión con la base de datos dentro de una transacción segura, y la cierra automáticamente al finalizar el bloque.
engine = create_engine(DB_URL, future=True)

@contextmanager
def session():
    with engine.begin() as conn:
        yield conn


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = Chroma(
    collection_name="productos",  # Nombre lógico de la colección (grupo) de vectores.
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# ────── FUNCIONES DE BD ──────
def get_or_create_user(phone: str):
    with session() as conn:
        tel = conn.execute(
            text("SELECT usuario_id FROM telefono WHERE numero = :n"),
            {"n": phone}
        ).scalar()
        if tel:
            return tel

        uid = conn.execute(text(
            "INSERT INTO usuario (nombre, apellido) VALUES ('', '') RETURNING id"
        )).scalar_one()
        conn.execute(text("""
            INSERT INTO telefono (numero, tipo, usuario_id)
            VALUES (:num, 1, :uid)
        """), {"num": phone, "uid": uid})
        return uid

def save_interaction(uid: int, msg: str, direction: int):
    with session() as conn:
        inter_id = conn.execute(text("""
            INSERT INTO interaccion (fecha, tipo_interaccion, usuario_id)
            VALUES (:f, :t, :u) RETURNING id
        """), {"f": datetime.utcnow().date(), "t": direction, "u": uid}).scalar_one()

        prod = conn.execute(text("""
            SELECT id FROM producto
            WHERE LOWER(nombre) ILIKE '%' || LOWER(:m) || '%'
            LIMIT 1
        """), {"m": msg}).scalar()

        if prod:
            conn.execute(text("""
                INSERT INTO interaccion_producto (contenido, interaccion_id, producto_id)
                VALUES (:c, :iid, :pid)
            """), {"c": msg[:255], "iid": inter_id, "pid": prod})

# ────── LLM + CONTEXTO ──────
def build_context(query: str):
    docs = vector_store.similarity_search(query, k=3)
    return "\n---\n".join(d.page_content for d in docs)

def call_llm(context: str, prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Eres un asistente amable.\nContexto:\n{context}"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()

# ────── WEBHOOK TWILIO ──────
@router.post("/webhook")
async def twilio_webhook(From: str = Form(...), Body: str = Form(...)):
    incoming = Body.strip()
    uid = get_or_create_user(From)
    save_interaction(uid, incoming, direction=1)

    rag_ctx = build_context(incoming)
    answer = call_llm(rag_ctx, incoming)

    save_interaction(uid, answer, direction=2)
    tw_resp = MessagingResponse()
    tw_resp.message(answer)
    return Response(content=str(tw_resp), media_type="application/xml")

app.include_router(router)

# ────── COMANDO ──────
# uvicorn main:app --reload --port 8000