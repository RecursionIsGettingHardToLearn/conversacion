import json
import os
from fastapi.responses import HTMLResponse
from datetime import datetime
from io import BytesIO
from typing import List
import psycopg2
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import Response as FastAPIResponse
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from pydantic import BaseModel
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sqlalchemy import select
from sqlalchemy import text as sql_text
from sqlalchemy import update
from sqlalchemy.orm import Session
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from extraf import fetch_messages_full
from config import OPENAI_API_KEY
from database import Base, SessionLocal, engine
from model import (
    Almacen,
    ContextoConversation,
    Conversacion,
    Interes,
    Mensaje,
    Producto,
    Promocion,Usuario
)

load_dotenv(override=True)

app = FastAPI()
router = APIRouter()
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga variables de entorno
DATABASE_URL = os.getenv("DB_URL")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
Base.metadata.create_all(bind=engine)


# Modelo de entrada
class QueryRequest(BaseModel):
    prompt: str

# --- Configuración del LLM y la plantilla ---
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

prompt_template = ChatPromptTemplate.from_template(
    """
Eres un asistente experto. Usa el contexto combinado para responder de manera clara.

Contexto local (historial de la conversación):
{contexto_local}

Contexto semántico (fragmento recuperado de Chroma):
{contexto_chroma}

Pregunta actual:
{pregunta}

Respuesta:
"""
)


@app.post("/query", response_class=JSONResponse)
async def query_endpoint(data: QueryRequest, db: Session = Depends(get_db)):
    if data.prompt.lower() == "begin":
        nueva = Conversacion(
            fecha_inicio=datetime.utcnow(), fecha_fin=None, usuario_id=1
        )
        db.add(nueva)
        db.commit()
        db.add(ContextoConversation(conversacion_id=nueva.id))
        db.commit()
        return {"response": " conversacion inicida exitosamnte"}
    if data.prompt.lower() == "end":
        print("se supone que entras aqui")
        convo = (
            db.query(Conversacion)
            .filter(Conversacion.usuario_id == 1, Conversacion.fecha_fin.is_(None))
            .order_by(Conversacion.fecha_inicio.desc())
            .first()
        )
        if convo:
            convo.fecha_fin = datetime.utcnow()
            db.commit()
        return {"response": " conversacion terminada exitosamnte"}
    # 2. Verifica conversación activa
    convo = (
        db.query(Conversacion)
        .filter(Conversacion.usuario_id == 1, Conversacion.fecha_fin.is_(None))
        .order_by(Conversacion.fecha_inicio.desc())
        .first()
    )

    if not convo:
        return {"response": " ESCRIBA BEGIN para iniciar una conversacion"}
    # 1) Verificar si hay una conversación activa:
    convo = (
        db.query(Conversacion)
        .filter(
            Conversacion.usuario_id == 1,           # aquí podrías usar user real
            Conversacion.fecha_fin.is_(None)
        )
        .order_by(Conversacion.fecha_inicio.desc())
        .first()
    )
    if not convo:
        return {"response": "ESCRIBE 'BEGIN' para iniciar una conversación."}

    # 2) Guardar el mensaje del usuario en la tabla Mensaje:
    nuevo_msg = Mensaje(
        conversacion_id=convo.id,
        fecha=datetime.utcnow(),  
        tipo="user",
        contenido=data.prompt.strip()
    )
    db.add(nuevo_msg)
    db.commit()

    try:
        # 3) Recuperar todo el historial de la conversación (contexto local):
        mensajes_bd = (
            db.query(Mensaje)
            .filter(Mensaje.conversacion_id == convo.id)
            .order_by(Mensaje.fecha.asc())
            .all()
        )
        # Convertirlos en un solo string. Por ejemplo, podrías formatear así:
        contexto_local = ""
        for m in mensajes_bd:
            # Prefix con quien envió: [Usuario:] o [Asistente:]
            etiqueta = "Usuario" if m.tipo == "user" else "Asistente"
            contexto_local += f"[{etiqueta}]: {m.contenido}\n"
        # Ahora contexto_local contiene todo el historial desde el inicio de la conversación.

        # 4) Recuperar el contexto semántico de Chroma:
        texto_usuario = data.prompt.strip()
        print('la pregunta\n', texto_usuario)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        vector_store = Chroma(
            persist_directory="segundo_db",
            embedding_function=embeddings
        )
        docs = vector_store.similarity_search(texto_usuario, k=1)
        # Tomamos el contenido de la página (texto Q&A) que Chroma nos devolvió:
        contexto_chroma = "\n".join([doc.page_content for doc in docs])

        # 5) Armar prompt combinando ambos contextos y la pregunta actual:
        prompt_completo = prompt_template.format(
            contexto_local=contexto_local,
            contexto_chroma=contexto_chroma,
            pregunta=texto_usuario
        )

        # 6) Ejecutar el LLM:
        respuesta_llm = llm.predict(prompt_completo)

        # 7) Guardar la respuesta del asistente en la tabla Mensaje:
        msg_assistant = Mensaje(
            conversacion_id=convo.id,
            fecha=datetime.utcnow(),  
            tipo="assistant",
            contenido=respuesta_llm.strip()
        )
        db.add(msg_assistant)
        db.commit()

        return {"response": respuesta_llm}

    except Exception as e:
        return {"response": f"Ocurrió un error al procesar la solicitud: {str(e)}"}
    
INTERESES_PERMITIDOS = [
    "televisor", "smart tv", "tv",
    "laptop", "computadora", "notebook", "hp pavilion", "dell inspiron",
    "iphone", "smartphone", "celular", "galaxy",
    "bocina", "parlante", "altavoz", "sony srs-xb43",
    "refrigeradora", "heladera", "frigorífico", "lg frost free",
    "audífono", "airpods", "auricular", "apple airpods",
    "microondas", "panasonic",
    # etc. SOLO los que sí tienes en la base
]


DICCIONARIO_SINONIMOS = {
    "televisor": ["televisor", "smart tv", "tv"],
    "smart tv": ["smart tv", "televisor", "tv"],
    "tv": ["tv", "televisor", "smart tv"],
    "laptop": ["laptop", "computadora", "notebook", "hp pavilion", "dell inspiron"],
    "computadora": ["computadora", "laptop", "notebook", "hp pavilion", "dell inspiron"],
    "notebook": ["notebook", "laptop", "computadora"],
    "iphone": ["iphone", "smartphone", "celular"],
    "smartphone": ["smartphone", "celular", "iphone", "galaxy"],
    "celular": ["celular", "smartphone", "iphone"],
    "bocina": ["bocina", "parlante", "altavoz", "sony srs-xb43"],
    "parlante": ["parlante", "bocina", "altavoz"],
    "altavoz": ["altavoz", "parlante", "bocina"],
    "refrigeradora": ["refrigeradora", "heladera", "frigorífico", "lg frost free"],
    "audífono": ["audífono", "airpods", "auricular", "apple airpods"],
    "microondas": ["microondas", "panasonic"],
    "heladera": ["refrigeradora", "frigorífico"],
}
def get_sinonimos(interes):
    return DICCIONARIO_SINONIMOS.get(interes.lower(), [interes])


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
@app.get("/publicidad/{conv_id}", response_class=HTMLResponse)
async def generar_publicidad_html(conv_id: int):
    print('get/publicidad')
    try:
        # 1) Obtener mensajes de la conversación
        mensajes = fetch_messages_full(conv_id)
        if not mensajes:
            raise HTTPException(
                status_code=404, detail="No hay mensajes para generar publicidad."
            )

        # 2) Generar contenido con OpenRouter
        system_msg = {
            "role": "system",
            "content": (
                "Eres un creador de contenido publicitario. "
                "Recibes un array de registros de mensajes de un usuario "
                "(id, conversacion_id, fecha, tipo, contenido). "
                "A partir de ese historial, crea un texto publicitario persuasivo "
                "que resalte los productos y promociones que interesan a este usuario, "
                "en un tono cercano y profesional."
            ),
        }
        user_msg = {
            "role": "user",
            "content": "Estos son los mensajes del usuario:\n"
            + json.dumps(mensajes, ensure_ascii=False, indent=2),
        }

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[system_msg, user_msg],
            temperature=0.7,
        )
        text_publicidad = resp.choices[0].message.content.strip()
        #elaboramos  una publicidad con los productos        
        prompt_intereses = {
        "role": "system",
        "content": (
            "Eres un analista de intereses. "
            "Recibes un texto publicitario generado para un usuario. "
            "Devuelve una lista en JSON (solo el array) de los nombres de productos, marcas o categorías de interés detectados. "
            "Si no hay ningún producto, marca o categoría clara, responde con un array vacío"
            "No agregues explicación, solo el array en formato JSON, por ejemplo: [\"Samsung\", \"Celulares\", \"Smart TV\"]."
        ),
        }
        user_msg_intereses = {
            "role": "user",
            "content": text_publicidad
        }

        resp_intereses = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[prompt_intereses, user_msg_intereses],
        temperature=0,
        )
        try:
            intereses = json.loads(resp_intereses.choices[0].message.content.strip())
        except Exception:
            intereses = []

        print('los interese son \n',intereses)
        intereses_singulares=[]
        if intereses:
            prompt_singularizar = {
            "role": "system",
            "content": (
            "Recibes un array de nombres de productos, marcas o categorías en plural en español. "
            "Devuélvelos en singular, respetando el significado. Si ya están en singular, déjalos igual. "
            "No agregues explicación, solo responde con el array en formato JSON."
            "\nEjemplo:\nEntrada: [\"patatas\", \"televisores\", \"huevos de gallina\"]\n"
            "Respuesta: [\"patata\", \"televisor\", \"huevo de gallina\"]"
            ),
            }
            user_msg_singularizar = {
            "role": "user",
            "content": json.dumps(intereses, ensure_ascii=False)
            }
            resp_singularizar = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[prompt_singularizar, user_msg_singularizar],
            temperature=0
            )
            try:
                intereses_singulares = json.loads(resp_singularizar.choices[0].message.content.strip())
            except Exception:
                intereses_singulares = intereses

            print('intereses singulares',intereses_singulares)

            
        conn = psycopg2.connect(DATABASE_URL)
        print("Conexión a la base de datos exitosa")  # Aquí lo imprimes
        cur = conn.cursor()
        productos = []
        for interes in intereses_singulares:
            sinonimos = get_sinonimos(interes)
            print(f'SINNIMOS DE LA PALABRA {interes} son :{sinonimos}')
            for palabra in sinonimos:
                cur.execute("""
                SELECT p.id, p.nombre, p.descripcion, p.precio, i.ruta
                FROM Producto p
                JOIN Imagen i ON p.id = i.producto_id
                WHERE p.nombre ILIKE %s
                LIMIT 1
            """, (f"%{palabra}%",))
            result = cur.fetchone()
            print('resultado de la busqueda',result)
            if result:
                print('se encontro producots')
                productos.append({
                    "nombre": result[1],
                    "descripcion": result[2],
                    "precio": result[3],
                    "imagen": result[4]
                })
                 
        cur.close()
        conn.close()
        print('los productos de interes del cliente son ',productos)
        # 4) Generar HTML de respuesta
        html = f"""
        <html>
        <head><title>Publicidad Personalizada</title></head>
        <body>
        """
        print('los productos son ',productos)
        texto_publicidad_real=""
        #hacemos publicidad solo de los prodcuco que se han encontrado
        if productos:
        # Genera publicidad SOLO con los productos encontrados
            nombres_productos = ", ".join([p['nombre'] for p in productos])
            system_msg_prod = {
            "role": "system",
            "content": (
                "Eres un creador de contenido publicitario. "
                "Recibes una lista de productos disponibles. "
                "Crea un texto persuasivo, breve y atractivo, solo hablando de estos productos: "
                + nombres_productos
            ),
            }
            user_msg_prod = {
                "role": "user",
                "content": "Estos son los productos recomendados: " + nombres_productos
            }
            resp_pub = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[system_msg_prod, user_msg_prod],
                temperature=0.7,
            )
            texto_publicidad_real = resp_pub.choices[0].message.content.strip()
        else:
            texto_publicidad_real = "¡Visita nuestra tienda y descubre lo mejor en electrónica!"

        if productos:
            
            html = f"""
            <html>
            <head>
            <title>Publicidad Personalizada</title>
            <style>
            body {{
            background-color: #f6f6f6;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            }}
            .container {{
            max-width: 700px;
            background: #fff;
            margin: 40px auto;
            border-radius: 12px;
            box-shadow: 0 2px 12px #0002;
            padding: 32px 24px;
             }}
            .titulo {{
            color: #4a90e2;
            font-size: 26px;
            margin-bottom: 10px;
            text-align: center;
            }}
            .subtitulo {{
            color: #444;
            font-size: 18px;
            text-align: center;
            margin-bottom: 24px;
            }}
            .publicidad {{
            background: #eaf6ff;
            padding: 16px 20px;
            border-radius: 7px;
            color: #333;
            font-size: 16px;
            margin-bottom: 30px;
            }}
            table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            }}
            th, td {{
            padding: 12px 8px;
            border-bottom: 1px solid #e2e2e2;
            text-align: left;
            }}
            th {{
            background: #4a90e2;
            color: #fff;
            font-size: 16px;
            }}
            td img {{
            border-radius: 8px;
            border: 1px solid #e2e2e2;
            }}
            .btn-comprar {{
            display: inline-block;
            padding: 10px 24px;
            margin-top: 18px;
            background: #43b76e;
            color: #fff !important;
            font-weight: bold;
            border-radius: 6px;
            text-decoration: none;
            box-shadow: 0 1px 4px #0001;
            font-size: 15px;
            }}
            .footer {{
            text-align: center;
            font-size: 13px;
            color: #999;
            margin-top: 40px;
            }}
            </style>
            </head>
            <body>
            <div class="container">
            <div class="titulo">🛒 ¡Tenemos algo especial para ti!</div>
            <div class="subtitulo">Ofertas exclusivas según tus intereses 🤩</div>
            <div class="publicidad">
            {texto_publicidad_real.replace('\n', '<br>')}
            </div>
            <h3 style="color:#43b76e; text-align:center; margin-top:28px;">Productos recomendados según tus intereses:</h3>
            <table>
            <tr>
                <th>Imagen</th>
                <th>Nombre</th>
                <th>Descripción</th>
                <th>Precio</th>
                <th>Comprar</th>
            </tr>
        """
            for p in productos:
                html += f"""
            <tr>
                <td><img src="{p['imagen']}" alt="Imagen" width="110"></td>
                <td>📺 {p['nombre']}</td>
                <td>{p['descripcion']}</td>
                <td><b style="color:#e67e22;">Bs.{p['precio']}</b></td>
                <td><a class="btn-comprar" href="https://tutienda.com/producto/{p['nombre'].replace(' ', '_')}" target="_blank">Comprar</a></td>
            </tr>
            """
            html += """
            </table>
            <div class="footer">
            ¿Tienes dudas? <a href="mailto:soporte@tutienda.com" style="color:#4a90e2;">Contáctanos</a> | <span style="font-size:18px;">💙</span> ¡Gracias por confiar en nosotros!
            </div>
            </div>
            </body>
            </html>
            """
        else:
            html = ""

         # 5) Obtener usuario_id de la conversación
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT usuario_id FROM Conversacion WHERE id = %s", (conv_id,))
        result = cur.fetchone()
        if not result:
            raise HTTPException(
                status_code=404, detail="No se encontró la conversación."
            )
        usuario_id = result[0]

        # 6) UPSERT en Interes (insertar o actualizar si ya existe)
        cur.execute(
            """
            INSERT INTO Interes (contenido, usuario_id, conversacion_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversacion_id) DO UPDATE SET
                contenido = EXCLUDED.contenido,
                usuario_id = EXCLUDED.usuario_id
            """,
            (html, usuario_id, conv_id),
        )
        conn.commit()
        cur.close()
        conn.close()

        return HTMLResponse(content=html)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))








app.include_router(router)





