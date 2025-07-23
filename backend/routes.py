import shap
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

from backend.shap_explainer import explain_text
from backend.database import feedback, subscribers
from backend.sms_alerts import send_sms
from model.predict import predict_news
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# ====== 🔹 Request Schemas ======

class NewsText(BaseModel):
    text: str

class FeedbackData(BaseModel):
    news_text: str
    predicted_label: str
    correct_label: str

class SubscriptionData(BaseModel):
    phone: str

class AlertData(BaseModel):
    news_title: str

# ====== 🔹 Helper Function to Send SMS Alerts ======

def alert_subscribers(news_title: str):
    alert_message = f"🚨 Alert: Potential Fake News Detected!\nTitle: {news_title}"
    subscriber_phones = [sub['phone'] for sub in subscribers.find()]
    success_count = 0
    fail_count = 0

    for phone in subscriber_phones:
        success = send_sms(phone, alert_message)
        if success:
            success_count += 1
        else:
            fail_count += 1
            logger.error(f"Failed to send SMS to {phone}")

    logger.info(f"SMS Alert Summary: {success_count} sent, {fail_count} failed.")
    return {"sent": success_count, "failed": fail_count}

# ====== 🔹 Predict News Endpoint ======

@router.post("/predict")
async def predict_fake_news(item: NewsText):
    try:
        result = predict_news(item.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ====== 🔹 SHAP Explanation Endpoint (HTML) ======

@router.post("/explain", response_class=HTMLResponse)
def explain_news(data: NewsText):
    try:
        shap_values = explain_text(data.text)

        # Generate interactive HTML from SHAP
        html = f"""
        <html>
        <head><title>SHAP Explanation</title></head>
        <body>
        <h3>SHAP Explanation for: <i>{data.text[:100]}...</i></h3>
        {shap.plots.text(shap_values[0], display=False)}
        </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {str(e)}")

# ====== 🔹 Feedback Endpoint ======

@router.post("/feedback")
def submit_feedback(data: FeedbackData):
    try:
        feedback.insert_one(data.dict())
        return {"message": "✅ Feedback saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# ====== 🔹 Subscribe for Alerts ======

@router.post("/subscribe")
def subscribe_user(data: SubscriptionData):
    try:
        if subscribers.find_one({"phone": data.phone}):
            return {"message": "Already subscribed."}
        subscribers.insert_one({"phone": data.phone})
        return {"message": "✅ Subscription successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription failed: {str(e)}")

# ====== 🔹 SMS Alert Trigger Endpoint ======

@router.post("/send-alert")
def send_alert(data: AlertData):
    try:
        result = alert_subscribers(data.news_title)
        return {"message": f"Alerts sent: {result['sent']}, failed: {result['failed']}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send alerts: {str(e)}")

# ====== 🔹 Health Check Endpoint ======

@router.get("/health")
async def health_check():
    return {"status": "ok"}
