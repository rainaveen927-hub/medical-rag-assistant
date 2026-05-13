import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Medical AI Assistant")

st.title("🩺 Medical RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a medical question")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    response = requests.post(
        API_URL,
        json={"query": user_input}
    )

    data = response.json()

    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"]
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])