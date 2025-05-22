import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv(override=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def cargar_y_procesar_txt(ruta_directorio, ruta_chroma):
    textos = []

    for nombre_archivo in os.listdir(ruta_directorio):
        if nombre_archivo.endswith(".txt"):
            with open(os.path.join(ruta_directorio, nombre_archivo), "r", encoding="utf-8") as archivo:
                textos.append(archivo.read())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.create_documents(textos)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY
    )
    vector_store = Chroma.from_documents(fragmentos, embeddings, persist_directory=ruta_chroma)
    print(f"Base de datos Chroma creada y almacenada automáticamente en: {ruta_chroma}")

if __name__ == "__main__":
    cargar_y_procesar_txt("chatbot\\productos", "chroma_db")