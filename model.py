from sqlalchemy import Column, Integer, Text, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    telefono_formateado = Column(Text)
    pregunta = Column(Text, nullable=False)
    contexto = Column(Text)
    respuesta = Column(Text)
    modelo = Column(Text, default="gpt-3.5-turbo")
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    error = Column(Boolean, default=False)
    error_message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


