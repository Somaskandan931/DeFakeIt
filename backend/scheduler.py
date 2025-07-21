import time
import schedule
import logging
from backend.fetch_news import fetch_trending_fake_news

# ========== 🔹 Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def job():
    logging.info("🔄 Running scheduled fetch job...")
    fetch_trending_fake_news()

# Every 6 hours (change as needed)
schedule.every(6).hours.do(job)

if __name__ == "__main__":
    logging.info("📅 News fetch scheduler started.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute
from model.retrain import retrain
schedule.every().sunday.at("02:00").do(retrain)
