import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Policy RAG Assistant", layout="centered")

st.title("📄 Policy RAG Assistant")
st.caption("Ask questions grounded in the policy document")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask a policy-related question...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Call backend API
    with st.spinner("Thinking..."):
        response = requests.post(
            API_URL,
            json={"question": query},
            timeout=60
        )

    data = response.json()
    answer = data["answer"]
    sources = data["sources"]

    # Show assistant response
    assistant_content = f"{answer}\n\n---\n**Sources:**\n{sources}"
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_content}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_content)
