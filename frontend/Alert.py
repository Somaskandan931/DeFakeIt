import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🚨 Send SMS Alert to Subscribers (Admin)")

news_title = st.text_input("News Title to Alert Subscribers About:")

if st.button("Send Alert"):
    if news_title.strip():
        with st.spinner("Sending alerts..."):
            response = requests.post(f"{API_URL}/send-alert", json={"news_title": news_title})
            if response.status_code == 200:
                data = response.json()
                st.success(f"Alerts sent: {data.get('message', '')}")
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter a news title.")
