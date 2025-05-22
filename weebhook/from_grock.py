import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the WhatsApp bot"

@app.route("/webhook", methods=["POST"])
def webhook():
    incoming_msg = request.values.get('Body', '').strip().lower()
    sender = request.values.get('From', '')

    # Create Twilio response object1111
    resp = MessagingResponse()

    # Dictionary-based response logic
    responses = {
        'hello': "Hi there! I'm your WhatsApp bot. How can I help you today?",
        'help': "I can assist with:\n- Product information\n- Business hours\n- Contact details\n\nJust ask me what you need!",
        'hours': "We're open Monday-Friday, 9am-5pm."
    }

    response_text = responses.get(incoming_msg, "I didn't understand that. Try asking for 'help' to see what I can do.")
    resp.message(response_text)

    return str(resp)

# ✅ Correct usage of __main__
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, port=port)
