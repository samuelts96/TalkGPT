import streamlit as st
import os
from loader import load_documents
from embed_store import create_or_load_vectorstore
from rag_chain import build_qa_chain, qa_chain

# Initialize QA chain
qa_chain_func = qa_chain()

st.set_page_config(page_title="Echo Bot", layout="wide")
st.title("☁️ Echo Bot — Weather Q&A Assistant")

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -----------------------------
# Quick Suggestions (Clickable)
# -----------------------------
st.markdown("### 💡 Quick Suggestions")

suggestions = [
    "What causes thunderstorms?",
    "How are hurricanes formed?",
    "What is the difference between weather and climate?"
]

cols = st.columns(3)
for i, (col, suggestion) in enumerate(zip(cols, suggestions)):
    if col.button(suggestion, key=f"suggest_{i}"):
        st.session_state.input_text = suggestion

# -----------------------------
# Chat Display
# -----------------------------
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if message["role"] == "assistant" and message.get("context"):
        for idx, doc in enumerate(message["context"]):
            with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                st.text_area(
                    label=' ',
                    value=doc.page_content,
                    height=150,
                    key=f"context_{i}_{idx}"
                )

# -----------------------------
# Chat Input & Submit
# -----------------------------
chat_history = []
for i in range(0, len(st.session_state.messages) - 1, 2):
    user_msg = st.session_state.messages[i]
    if i + 1 < len(st.session_state.messages):
        assistant_msg = st.session_state.messages[i + 1]
        chat_history.append({
            "user": user_msg["content"],
            "assistant": assistant_msg["content"],
        })

with st.form("chat_form", clear_on_submit=True):
    prompt = st.text_input("Ask something about the weather...", value=st.session_state.input_text, key="text_input")
    submitted = st.form_submit_button("Send")

if submitted and prompt:
    st.session_state.input_text = ""  # Clear suggestion input

    st.session_state.messages.append({"role": "user", "content": prompt, 'context': None})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = qa_chain_func({"question": prompt, "chat_history": chat_history})

    with st.chat_message("assistant"):
        st.markdown(response['answer'])

        if response.get("source_documents"):
            st.markdown("**Context Source:**")
            for idx, doc in enumerate(response["source_documents"]):
                with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                    st.text_area(
                        label=' ',
                        value=doc.page_content,
                        height=150,
                        key=f"latest_context_{idx}"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response['answer'],
        "context": response.get("source_documents", [])
    })
