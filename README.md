# DeFakeIt: Real-Time Explainable Fake News Detection System

DeFakeIt is a real-time fake news detection platform that combines the power of deep learning and explainable AI to identify misinformation in news articles. It allows users to verify news authenticity, visualize model reasoning, provide feedback, and receive live SMS alerts on trending fake news.

---

## Features

* Detects whether a news article is **real** or **fake** using a **BERT-LSTM** hybrid model.
* Offers **SHAP-based explainability** to visualize word-level influence on predictions.
* Interactive **Streamlit frontend** for manual verification, feedback, and subscription.
* **FastAPI backend** to serve predictions, SHAP visualizations, feedback, and alerts.
* Periodic **live news scraping** via NewsAPI and automated retraining.
* Sends **SMS alerts** for fake news using Twilio.
* Stores feedback, subscriptions, and news articles in **MongoDB**.

---

## Tech Stack

### Machine Learning & NLP

* BERT (Hugging Face Transformers)
* LSTM (PyTorch)
* BERT-LSTM Hybrid Classifier
* SHAP for interpretability

### Backend

* FastAPI (REST API)
* Uvicorn (ASGI server)

### Frontend

* Streamlit (interactive user interface)

### Data

* ISOT Fake News Dataset
* NewsAPI for real-time news

### Utilities & Storage

* MongoDB (feedback, news, subscriptions)
* Twilio API (SMS alerts)
* Docker (optional deployment)
* Git & GitHub for version control

---

## Folder Structure

```
DeFakeIt/
│
├── backend/                 
│   ├── main.py              # FastAPI entry point
│   ├── routes.py            # API endpoints
│   ├── model.py             # Model loading and inference
│   ├── shap_explainer.py    # SHAP explainability logic
│   ├── preprocessing.py     # Text cleaning and tokenization
│   ├── sms_alerts.py        # Twilio SMS logic
│   ├── database.py          # MongoDB connection
│   └── scheduler.py         # News scraping and retraining
│
├── frontend/                
│   ├── Home.py              # News verification interface
│   ├── SHAP.py              # SHAP visualizer
│   ├── Feedback.py          # Feedback form
│   ├── Subscribe.py         # SMS subscription UI
│   └── Alert.py             # Admin-only SMS sender
│
├── model/                   
│   ├── train.py             # Model training pipeline
│   ├── evaluate.py          # Evaluation reports
│   ├── predict.py           # Inference logic
│   ├── bert_lstm_model.py   # Model architecture
│   ├── tokenizer.pkl        # Saved tokenizer
│   └── model.pt             # Trained weights
│
├── data/                    
│   ├── isot_fake_news.csv   # ISOT combined dataset
│   ├── true.csv             # Real news
│   ├── fake.csv             # Fake news
│   └── new_articles.json    # Scraped articles
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── tests/                   
│   ├── test_backend.py
│   ├── test_model.py
│   └── test_utils.py
│
├── requirements.txt
├── .env
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DeFakeIt.git
cd DeFakeIt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root with the following:

```
NEWS_API_KEY=your_newsapi_key
MONGO_URI=mongodb://localhost:27017/
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
```

### 4. Run the Backend

```bash
cd backend
uvicorn main:app --reload
```

### 5. Run the Frontend

```bash
cd frontend
streamlit run Home.py
```

---

## API Endpoints

| Method | Endpoint      | Description                         |
| ------ | ------------- | ----------------------------------- |
| POST   | `/predict`    | Predict if news is fake or real     |
| POST   | `/explain`    | Get SHAP explanation for input text |
| POST   | `/feedback`   | Submit user feedback                |
| POST   | `/subscribe`  | Register for SMS alerts             |
| POST   | `/send-alert` | Send SMS to all subscribers         |
| GET    | `/health`     | Health check                        |

---

## Example Use Cases

* Verify breaking news for authenticity.
* Explore model reasoning using SHAP plots.
* Subscribe to SMS alerts for trending fake news.
* Submit corrections to improve model quality.

---

## License

This project is licensed under the [MIT License](LICENSE).
