from pymongo import MongoClient
import os

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URL)
db = client["defakeit"]

# Collections
articles = db["verified_articles"]
feedback = db["user_feedback"]
subscribers = db["subscribers"]

def save_feedback(text, prediction, user_comment):
    feedback.insert_one({
        "text": text,
        "prediction": prediction,
        "comment": user_comment
    })
