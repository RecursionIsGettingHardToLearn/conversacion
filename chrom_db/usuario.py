from db import session
from sqlalchemy import text

def listar_usuarios():
    with session() as conn:
        result = conn.execute(text("SELECT * FROM usuario"))
        for fila in result:
            print(fila)

if __name__ == "__main__":
    listar_usuarios()