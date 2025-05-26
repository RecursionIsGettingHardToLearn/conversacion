from twilio.rest import Client as TwilioClient
from chrom_db.config import ACCOUNT_SID, AUTH_TOKEN, FROM_NUMBER

twilio_client = TwilioClient(ACCOUNT_SID, AUTH_TOKEN)

def send_sms(to: str, body: str):
    """Envía un SMS con Twilio y devuelve el objeto Message creado."""
    return twilio_client.messages.create(body=body, from_=FROM_NUMBER, to=to)
