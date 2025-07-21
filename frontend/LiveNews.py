import streamlit as st
import pymongo
import os
from datetime import datetime

# ========== 🔹 MongoDB Connection ==========
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = pymongo.MongoClient(MONGO_URI)
db = client["fake_news_detection"]
collection = db["live_news"]

# ========== 🔹 UI Title ==========
st.set_page_config(page_title="Live News", layout="wide")
st.title("📰 Live News Fetched from NewsAPI")
st.markdown("This table shows the latest trending news related to **fake news, misinformation, hoaxes,** etc.")

# ========== 🔹 Fetch & Display ==========
news_list = list(collection.find().sort("published_at", -1))

if not news_list:
    st.warning("⚠️ No news articles found in the database.")
else:
    for article in news_list:
        st.markdown("---")
        st.subheader(article.get("title", "No Title"))
        st.markdown(f"**Source:** {article.get('source', 'Unknown')} | 🕒 {article.get('published_at', 'N/A')}")
        st.write(article.get("description", "No Description"))
        st.markdown(f"[🔗 Read More]({article.get('url')})")

# ========== 🔹 Footer ==========
st.markdown("---")
st.caption(f"Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
