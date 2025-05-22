from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()
# Usa variables de entorno o reemplaza directamente
account_sid = os.getenv('TWILIO_ACCOUNT_SID')  # '
auth_token = os.getenv('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

message = client.messages.create(
    body='este es el segundo mensaje',
    from_='whatsapp:+14155238886',
    to='whatsapp:+59163618447'
)

print("Mensaje enviado correctamente. SID:", message.sid)
