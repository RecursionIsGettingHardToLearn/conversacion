import os
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN")
FROM_NUMBER    = os.getenv("TWILIO_FROM_NUMBER")

CONV_FILE  = "conversacion.txt"
VECTOR_DIR = "chroma_db"
