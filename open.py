import os
import json
import psycopg2
from openai import OpenAI
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import LETTER

from io import BytesIO

# Carga variables de entorno
load_dotenv(override=True)
DATABASE_URL = os.getenv("DB_URL")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# Inicializa cliente apuntando a OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY
)

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
    cur.close()
    conn.close()

    return [
        {"role": "user" if tipo == "user" else "assistant", "content": contenido}
        for tipo, contenido in rows
    ]

@app.get("/fetch-messages/{conv_id}", response_model=list[Message])
async def get_messages(conv_id: int):
    """
    Endpoint para obtener todos los mensajes de una conversación.
    """
    try:
        return fetch_messages(conv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analizar-intenciones/{conv_id}")
async def analizar(conv_id: int):
    """
    Endpoint que analiza intenciones y entidades de una conversación
    identificada por conv_id y devuelve un JSON con el análisis.
    """
    # Recupera sólo los mensajes de usuario para el análisis
    all_msgs = fetch_messages(conv_id)
    user_msgs = [m for m in all_msgs if m["role"] == "user"]

    if not user_msgs:
        raise HTTPException(status_code=404, detail="No hay mensajes de usuario para analizar.")

    # Few-shot + prompt estricto
    system_msg = {
        "role": "system",
        "content": """
            Eres un extractor de intenciones y entidades.  
            Recibe los siguientes mensajes de usuario y devuelve **solo** un JSON con dos claves:
            - intenciones: lista de identificadores de intenciones (snake_case)
            -entidades: lista de entidades reconocidas.

            Ejemplo:
            Usuario: "¿Cuál es el precio del microondas?"
            Salida:
            {
            "intenciones": ["consulta_precio"],
            "entidades": ["microondas"]
            }

        Ahora analiza estos mensajes:
    """
    }

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",  # o el modelo que elijas
            messages=[system_msg, *user_msgs],
            extra_headers={},    # opcional: referer, X-Title, etc.
            extra_body={}        # opcional
        )

        # Conteo de tokens
        usage = resp.usage
        print(f"→ Tokens prompt: {usage.prompt_tokens}")
        print(f"→ Tokens completion: {usage.completion_tokens}")
        print(f"→ Tokens total: {usage.total_tokens}")

        # Parseo de la respuesta
        payload = resp.choices[0].message.content
        print("→ Payload recibido:", payload)

        #try:
        #    data = json.loads(payload)
        #except json.JSONDecodeError as je:
        #    raise HTTPException(
        #        status_code=500,
        #        detail=f"Error al parsear JSON: {je.msg}. Payload bruto: {payload}"
        #    )

        return payload

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_messages_full(conv_id: int) -> list[dict]:
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, conversacion_id, fecha, tipo, contenido
        FROM Mensaje
        WHERE conversacion_id = %s
        ORDER BY fecha
    """, (conv_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id":              r[0],
            "conversacion_id": r[1],
            "fecha":           r[2].isoformat(),
            "tipo":            r[3],
            "contenido":       r[4],
        }
        for r in rows
    ]



@app.get("/analizar-intereses/{conv_id}")
async def analizar_intereses(conv_id: int):
    try:
        mensajes = fetch_messages_full(conv_id)
        if not mensajes:
            raise HTTPException(status_code=404, detail="No hay mensajes para analizar.")

        # Prepara el prompt
        system_msg = {
            "role": "system",
            "content": (
                "Eres un analista de perfil de usuario. "
                "Recibes la lista completa de registros de la tabla Mensaje "
                "(id, conversacion_id, fecha, tipo, contenido). "
                "Devuélveme un texto con los principales intereses o temas que muestra este usuario, "
                "basado en el contenido de esos mensajes."
            )
        }
        user_msg = {
            "role": "user",
            "content": "Estos son los registros de Mensaje:\n" + json.dumps(
                mensajes, ensure_ascii=False, indent=2
            )
        }

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0
        )
        return resp.choices[0].message.content

    except HTTPException:
        # Si ya es un HTTPException, lo relanzamos para respetar su código
        raise
    except Exception as e:
        # Para todo lo demás, devolvemos 500 con el mensaje de error
        raise HTTPException(status_code=500, detail=str(e))


# Tu función analizar_intereses ya definida arriba...

@app.get("/intereses/{conv_id}/pdf")
async def intereses_como_pdf(conv_id: int):
    """
    Genera un PDF maquetado con título y lista de intereses,
    y lo devuelve para descarga.
    """
    try:
        # 1) Obtén el texto crudo
        texto = await analizar_intereses(conv_id)

        # 2) Prepara el buffer en memoria
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER,
                                title="Análisis de Intereses",
                                author="Tu API")

        styles = getSampleStyleSheet()
        story = []

        # 3) Añade un encabezado
        story.append(Paragraph("Análisis de Intereses del Usuario", styles["Title"]))
        story.append(Spacer(1, 12))

        # 4) Divide en líneas y crea una lista con bullets
        lines = texto.split("\n")
        bullets = []
        for line in lines:
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped.startswith("•") or stripped.startswith("-")):
                item = stripped.lstrip("0123456789. ").lstrip("•- ").strip()
                bullets.append(ListItem(Paragraph(item, styles["Normal"])))
            else:
                story.append(Paragraph(stripped, styles["BodyText"]))
                story.append(Spacer(1, 6))

        if bullets:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Temas principales:", styles["Heading2"]))
            story.append(Spacer(1, 6))
            story.append(ListFlowable(bullets, bulletType="bullet"))

        # 5) Construye el PDF
        doc.build(story)

        buffer.seek(0)
        filename = f"intereses_conversacion_{conv_id}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    except HTTPException:
        # Errores previstos (404, etc.), relanzar
        raise
    except Exception as e:
        # Cualquier otro fallo, 500 con detalle
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {e}")
    

from fastapi.responses import PlainTextResponse

@app.get("/publicidad/{conv_id}", response_class=PlainTextResponse)
async def generar_publicidad(conv_id: int):
    try:
        # 1) Recupera todos los mensajes con todos sus campos
        mensajes = fetch_messages_full(conv_id)  # debe devolver lista de dicts completos
        if not mensajes:
            raise HTTPException(status_code=404, detail="No hay mensajes para generar publicidad.")
        
        # 2) Construye el prompt
        system_msg = {
            "role": "system",
            "content": (
                "Eres un creador de contenido publicitario. "
                "Recibes un array de registros de conversación de un usuario "
                "(id, conversacion_id, fecha, tipo, contenido). "
                "A partir de ese historial, crea un texto publicitario persuasivo "
                "que resalte los productos y promociones que interesan a este usuario, "
                "en un tono cercano y profesional."
            )
        }
        user_msg = {
            "role": "user",
            "content": (
                "Estos son los mensajes del usuario:\n" +
                json.dumps(mensajes, ensure_ascii=False, indent=2)
            )
        }

        # 3) Llamada al modelo
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",       # o el modelo que prefieras
            messages=[system_msg, user_msg],
            temperature=0.7,             # creatividad moderada
            max_tokens=300
        )

        # 4) Devuelve el texto generado
        return resp.choices[0].message.content.strip()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/publicidad/pdf/{conv_id}", response_class=StreamingResponse)
async def generar_publicidad_pdf(conv_id: int):
    try:
        # 1) Genera el texto publicitario (reutiliza tu lógica de /publicidad)
        mensajes = fetch_messages_full(conv_id)
        if not mensajes:
            raise HTTPException(status_code=404, detail="No hay mensajes para generar publicidad.")

        system_msg = {
            "role": "system",
            "content": (
                "Eres un creador de contenido publicitario. "
                "Recibes un array de registros de conversación de un usuario "
                "(id, conversacion_id, fecha, tipo, contenido). "
                "A partir de ese historial, crea un texto publicitario persuasivo "
                "que resalte los productos y promociones que interesan a este usuario, "
                "en un tono cercano y profesional."
            )
        }
        user_msg = {
            "role": "user",
            "content": (
                "Estos son los mensajes del usuario:\n" +
                json.dumps(mensajes, ensure_ascii=False, indent=2)
            )
        }

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0.7,
            max_tokens=300
        )
        text_publicidad = resp.choices[0].message.content.strip()

        # 2) Crea un PDF en memoria con ReportLab
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER,
                                rightMargin=40, leftMargin=40,
                                topMargin=60, bottomMargin=60)
        styles = getSampleStyleSheet()
        story = []

        # Título
        story.append(Paragraph("Publicidad", styles["Title"]))
        story.append(Spacer(1, 12))

        # Cada párrafo por salto de línea
        for linea in text_publicidad.split("\n\n"):
            story.append(Paragraph(linea.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 12))

        doc.build(story)
        buffer.seek(0)

        # 3) Devuelve el PDF como StreamingResponse
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=publicidad_{conv_id}.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("open:app", host="0.0.0.0", port=8000, reload=True)