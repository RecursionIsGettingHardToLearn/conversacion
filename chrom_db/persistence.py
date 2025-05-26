# persistence.py
from datetime import datetime
from db import session, text

def get_or_create_user(phone: str):
    """Devuelve (usuario_id) y garantiza que el teléfono exista."""
    with session() as conn:
        # 1. ¿existe el teléfono?
        tel = conn.execute(
            text("SELECT usuario_id FROM telefono WHERE numero = :n"),
            {"n": phone}
        ).scalar()

        if tel:
            return tel    # ya está vinculado a un usuario

        # 2. Crear usuario anónimo y su teléfono
        uid = conn.execute(text(
            "INSERT INTO usuario (nombre, apellido) VALUES ('', '') RETURNING id"
        )).scalar_one()

        conn.execute(text("""
            INSERT INTO telefono (numero, tipo, usuario_id)
            VALUES (:num, 1, :uid)
        """), {"num": phone, "uid": uid})

        return uid


def save_interaction(uid: int, msg: str, direction: int):
    """Inserta una fila en interaccion y enlaza con producto si hace match."""
    with session() as conn:
        inter_id = conn.execute(text("""
            INSERT INTO interaccion (fecha, tipo_interaccion, usuario_id)
            VALUES (:f, :t, :u) RETURNING id
        """), {"f": datetime.utcnow().date(), "t": direction, "u": uid}).scalar_one()

        # detección sencilla de producto ↴
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
