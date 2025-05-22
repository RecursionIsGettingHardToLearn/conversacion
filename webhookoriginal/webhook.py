from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/")
def root():
    return "Welcome"

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get('Body', '').strip().lower()
    sender = request.values.get('From', '')
    print(f"Mensaje recibido: {incoming_msg} de {sender}")

    resp = MessagingResponse()

    if 'hello' in incoming_msg:
        resp.message("Hi there! I'm your WhatsApp bot. How can I help you today?")
    elif 'help' in incoming_msg:
        resp.message("I can assist with:\n- Product information\n- Business hours\n- Contact details\n\nJust ask me what you need!")
    elif 'hours' in incoming_msg:
        resp.message("We're open Monday-Friday, 9am-5pm.")
    else:
        resp.message("I didn't understand that. Try asking for 'help' to see what I can do.")

   
    return str(resp)

@app.route("/status", methods=["POST"])
def status_callback():
    data = request.form
    print("Mensaje status:", data)
    return Response("OK", status=200)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
