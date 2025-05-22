import os
from dotenv import load_dotenv
import sqlalchemy as sa
from contextlib import contextmanager

# Cargar el archivo .env (asegúrate de que esté en el mismo directorio)
load_dotenv(dotenv_path=".env")

# Leer la URL
db_url = os.getenv("DB_URL")

# Validar si se cargó correctamente
if not db_url:
    raise RuntimeError("❌ La variable DB_URL no está definida o no se pudo cargar desde .env.")

print("✅ DB_URL cargada correctamente:", db_url)

# Crear el engine
engine = sa.create_engine(db_url, future=True)

@contextmanager
def session():
    with engine.begin() as conn:
        yield conn
