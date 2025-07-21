import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("🔍 SHAP Explanation for News Prediction")

news_text = st.text_area("Enter news text for explanation:", height=200)

if st.button("Get Explanation"):
    if news_text.strip():
        with st.spinner("Generating explanation..."):
            response = requests.post(f"{API_URL}/explain", json={"text": news_text})
            if response.status_code == 200:
                data = response.json()
                # SHAP HTML image embedded
                st.markdown(data["shap_html"], unsafe_allow_html=True)
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter some news text.")
