import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("📲 Subscribe for Daily SMS Alerts")

phone = st.text_input("Enter your phone number (E.164 format, e.g. +1234567890):")

if st.button("Subscribe"):
    if phone.strip():
        with st.spinner("Subscribing..."):
            response = requests.post(f"{API_URL}/subscribe", json={"phone": phone})
            if response.status_code == 200:
                data = response.json()
                if "already" in data.get("message", "").lower():
                    st.info(data["message"])
                else:
                    st.success(data["message"])
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter a valid phone number.")
