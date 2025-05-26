from fastapi import FastAPI, APIRouter, Form, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from chrom_db.config import OPENAI_API_KEY
from sqlalchemy.orm import Session
from model import Log
from fastapi import Depends
from database import SessionLocal

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-3.5-turbo"

# Función auxiliar: recuperar contexto
def recuperar_contexto(pregunta: str, k: int = 3) -> str:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    documentos = vector_store.similarity_search(pregunta, k=k)
    return "\n---\n".join(doc.page_content for doc in documentos) or "Sin contexto previo."

#Safe text
def safe_text(text: str) -> str:
    try:
        return text.encode("utf-8", errors="ignore").decode("utf-8")
    except Exception:
        return ""



# Endpoint de Webhook de Twilio
@router.post("/webhook")
async def twilio_webhook(
    From: str = Form(...), 
    Body: str = Form(...), 
    db: Session = Depends(get_db)
) -> Response:
    try:
        print('hola')
        pregunta = Body.strip()
        contexto = recuperar_contexto(pregunta)

        prompt = (
            "Eres un asistente conversacional amable. "
            "Si no hay suficiente contexto, pide más detalles.\n\n"
            f"Contexto:\n{contexto}"
        )

        respuesta = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": pregunta},
            ]
        )

        output = respuesta.choices[0].message.content.strip()
        usage = respuesta.usage  # tokens usados

        # Guardar en la base de datos
        log = Log(
            telefono_formateado=safe_text(From),
            pregunta=safe_text(pregunta),
            contexto=safe_text(contexto),
            respuesta=safe_text(output),
            modelo=MODEL_NAME,
            tokens_input=usage.prompt_tokens,
            tokens_output=usage.completion_tokens,
            error=False
        )
        db.add(log)
        db.commit()

        twilio_response = MessagingResponse()
        twilio_response.message(output)
        return Response(content=str(twilio_response), media_type="application/xml")

    except Exception as e:
        db.rollback()
        error_log = Log(
            telefono_formateado=safe_text(From),
            pregunta=safe_text(Body),
            contexto="",
            respuesta="",
            modelo=MODEL_NAME,
            error=True,
            error_message=safe_text(str(e))
        )
        db.add(error_log)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


# Registrar router
app.include_router(router)

# Comando sugerido para ejecutar:
# uvicorn main:app --reload --port 8000
