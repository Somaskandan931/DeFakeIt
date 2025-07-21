import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Adjust to your backend URL

st.title("📰 DeFakeIt: Real-Time Fake News Detection")

news_text = st.text_area("Paste news text to verify:", height=200)

if st.button("Check News"):
    if news_text.strip():
        with st.spinner("Analyzing news..."):
            response = requests.post(f"{API_URL}/predict", json={"text": news_text})
            if response.status_code == 200:
                data = response.json()
                st.success(f"Prediction: **{data['label'].upper()}**")
                st.write(f"Fake Probability: {data['fake_probability']:.2%}")
                if data['label'] == "fake":
                    st.warning("⚠️ This news is likely fake!")
                else:
                    st.info("✅ This news appears genuine.")
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter some news text to verify.")
