import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("💬 Submit Feedback")

news_text = st.text_area("News Text:", height=150)
predicted_label = st.selectbox("Predicted Label", ["fake", "real"])
correct_label = st.selectbox("Correct Label", ["fake", "real"])

if st.button("Submit Feedback"):
    if news_text.strip():
        payload = {
            "news_text": news_text,
            "predicted_label": predicted_label,
            "correct_label": correct_label
        }
        with st.spinner("Submitting feedback..."):
            response = requests.post(f"{API_URL}/feedback", json=payload)
            if response.status_code == 200:
                st.success("Thank you! Your feedback has been saved.")
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter the news text.")
