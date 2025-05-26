# main.py
from fastapi import FastAPI, APIRouter, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import os

# ────────────────────────────
# Configuración y credenciales
# ────────────────────────────
load_dotenv(override=True)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ACCOUNT_SID      = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN       = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER      = os.getenv("TWILIO_FROM_NUMBER")

CONV_FILE   = "conversacion.txt"       # log plano
VECTOR_DIR  = "chroma_db"  # base vectorial


app = FastAPI()
twilio = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # cambia en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY,
)

vector_store = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings,
)

def load_initial_conversations() -> None:
    """Ingesta líneas antiguas al vector store (solo texto puro)."""
    if os.path.exists(CONV_FILE):
        with open(CONV_FILE, encoding="utf-8") as f:
            old_lines = [
                ln.strip() for ln in f
                if ln.strip() and not ln.startswith("[")
            ]
        if old_lines:
            vector_store.add_texts(old_lines)

#load_initial_conversations()


class SendMessageRequest(BaseModel):
    to: str
    message: str


@app.post("/izquierda")
def send_message(req: SendMessageRequest):
    msg = twilio.messages.create(
        body=req.message,
        from_=FROM_NUMBER,
        to=req.to,
    )
    _log(f"Twilio → {req.to}: {req.message}")
    return {"status": "sent", "sid": msg.sid}

@app.post("/derecha")
def receive_message(From: str = Form(...), Body: str = Form(...)):
    _log(f"{From} → Bot: {Body}")
    return {"status": "received", "from": From, "message": Body}


router = APIRouter()

@router.post("/webhook")
async def twilio_webhook(
    From: str = Form(...),
    Body: str = Form(...),
):
    incoming = Body.strip()

    # 1 · Aprendizaje continuo
    vector_store.add_texts([incoming])

    # 2 · RAG
    retrieved = vector_store.similarity_search(incoming, k=3)
    ctx = "\n---\n".join(doc.page_content for doc in retrieved) or "Sin contexto previo."

    # 3 · LLM
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente conversacional amable. "
                    "Usa el contexto recuperado para contestar de forma breve.\n\n"
                    f"Contexto:\n{ctx}"
                ),
            },
            {"role": "user", "content": incoming},
        ],
    )
    answer = response.choices[0].message.content.strip()

    # 4 · XML para Twilio
    tw_resp = MessagingResponse()
    tw_resp.message(answer)

    # 5 · Log
    _log(f"{From} → Bot: {incoming}\nBot → {From}: {answer}")

    return Response(content=str(tw_resp), media_type="application/xml")

app.include_router(router)


def _log(text: str) -> None:
    ts = datetime.now().isoformat()
    with open(CONV_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")

# ────────────────────────────
# Arranque
# ────────────────────────────
#  uvicorn main:app --reload --port 8000
