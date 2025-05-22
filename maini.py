from fastapi import FastAPI, APIRouter, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI

from schemas import SendMessageRequest
from config import OPENAI_API_KEY
from twilio_utils import send_sms
from vector_store import vector_store
from logging_utils import log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ──────────── Endpoints básicos ────────────
@router.post("/izquierda")
def send_message(req: SendMessageRequest):
    msg = send_sms(req.to, req.message)
    log(f"Twilio → {req.to}: {req.message}")
    return {"status": "sent", "sid": msg.sid}

@router.post("/derecha")
def receive_message(From: str = Form(...), Body: str = Form(...)):
    log(f"{From} → Bot: {Body}")
    return {"status": "received", "from": From, "message": Body}

# ──────────── Webhook Twilio ────────────
@router.post("/webhook")
async def twilio_webhook(From: str = Form(...), Body: str = Form(...)):
    incoming = Body.strip()

    # 1 · Aprendizaje continuo
    vector_store.add_texts([incoming])

    # 2 · Recuperar contexto
    retrieved = vector_store.similarity_search(incoming, k=3)
    ctx = "\n---\n".join(d.page_content for d in retrieved) or "Sin contexto previo."
     
    print(f"Contexto:\n{ctx}")
    # 3 · ChatCompletion
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": ("Eres un asistente conversacional amable. "
                         "Si no hay suficiente contexto, pide más detalles.\n\n"
                         f"Contexto:\n{ctx}")
            },
            {"role": "user", "content": incoming},
        ],
    )
    answer = response.choices[0].message.content.strip()

    # 4 · Respuesta XML para Twilio
    tw_resp = MessagingResponse()
    tw_resp.message(answer)

    # 5 · Log
    log(f"{From} → Bot: {incoming}\nBot → {From}: {answer}")

    return Response(content=str(tw_resp), media_type="application/xml")

app.include_router(router)
