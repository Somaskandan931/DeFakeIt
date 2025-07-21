import os
import logging
from twilio.rest import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # Your Twilio phone number

if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
    logger.error("Twilio credentials are not fully set in environment variables!")

# Initialize Twilio client (only if credentials are available)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]) else None

def send_sms(to_phone: str, message: str) -> bool:
    """
    Send an SMS alert via Twilio.

    Args:
        to_phone (str): Recipient phone number (E.164 format, e.g. '+1234567890')
        message (str): Text message content

    Returns:
        bool: True if sent successfully, False otherwise
    """
    if client is None:
        logger.error("Twilio client not initialized. Check credentials.")
        return False

    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        logger.info(f"✅ SMS sent to {to_phone}. SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send SMS to {to_phone}: {e}")
        return False
