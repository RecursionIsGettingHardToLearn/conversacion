import os
import json
import psycopg2
from openai import OpenAI
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

# Carga variables de entorno
load_dotenv(override=True)
DATABASE_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializa cliente OpenAI v1.0+
client = OpenAI(api_key=OPENAI_API_KEY)

# Crea la aplicación FastAPI
app = FastAPI()

# Configuración CORS (ajusta orígenes en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

def fetch_messages(conversacion_id: int) -> list[dict]:
    """
    Recupera los mensajes de la conversación desde PostgreSQL.
    Devuelve una lista de dicts con 'role' y 'content'.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT tipo, contenido
        FROM Mensaje
        WHERE conversacion_id = %s
        ORDER BY fecha
    """, (conversacion_id,))
    rows = cur.fetchall()
    print('suces')
    cur.close()
    conn.close()

    return [
        {"role": "user" if tipo == "user" else "assistant", "content": contenido}
        for tipo, contenido in rows
    ]

# …importaciones y fetch_messages idénticos…
# …fetch_messages, imports, etc…

@app.get("/analizar-intenciones/{conv_id}")
async def analizar(conv_id: int):
    all_msgs = fetch_messages(conv_id)
    user_msgs = [m for m in all_msgs if m["role"]=="user"]
    if not user_msgs:
        raise HTTPException(404, "No hay mensajes de usuario.")

    # 1) System prompt muy breve
    system_msg = {
      "role": "system",
      "content": (
        "Eres un extractor de intenciones y entidades. "
        "Devuelve cadena mas entidades e intensiones."
      )
    }

    # 2) Dos ejemplos explicitos como user/assistant
    example_1 = {"role":"user", "content":"Pon una alarma a las 7 de la mañana."}
    example_1_resp = {"role":"assistant", "content":
    """{
  "intenciones": ["configurar_alarma"],
  "entidades": ["07:00"]
}"""}

    example_2 = {"role":"user", "content":
      "Reserva una mesa para 4 en un restaurante italiano mañana a las 8 pm."}
    example_2_resp = {"role":"assistant", "content":
    """{
  "intenciones": ["reservar_mesa"],
  "entidades": ["4", "restaurante italiano", "mañana", "20:00"]
}"""}

    # 3) Construimos la lista completa de mensajes
    prompt_messages = [
      system_msg,
      example_1, example_1_resp,
      example_2, example_2_resp,
      # ahora los mensajes reales del usuario
      *user_msgs
    ]

    # 4) Imprime para debug
    print("--- PROMPT COMPLETO ---")
    for m in prompt_messages:
        print(f"[{m['role']}] {m['content']}")
    print("----------------------")

    # 5) Llamada con stops y max_tokens
    resp = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=prompt_messages,
      temperature=0,
      max_tokens=200,
      # para que deje de generar justo al cerrar la llave
    )

    # 6) Reconstruye el JSON completo (añade la llave de cierre que `stop` consumió)
    raw = resp.choices[0].message.content
  

    print("--- PAYLOAD RECIBIDO ---\n", raw)
    return raw




@app.get("/fetch-messages/{conv_id}", response_model=list[Message])
async def get_messages(conv_id: int):
    """
    Endpoint para obtener todos los mensajes de una conversación.
    """
    try:
        msgs = fetch_messages(conv_id)
        return msgs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("analisis:app", host="0.0.0.0", port=8000, reload=True)
