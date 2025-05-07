import os
import streamlit as st
from rag_chain import build_qa_chain, qa_chain

def save_chatgpt_model(st, model):
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model

def load_api_key(st):
    if "OPENAI_API_KEY" not in st.secrets:
        raise ValueError("ERROR: API KEY NOT FOUND. ENSURE AN API KEY IS SAVED LOCALLY.")
    else:
        return st.secrets['OPENAI_API_KEY']

directory = "meteorology_files"

chatgpt_model = "gpt-4.1-nano-2025-04-14"
save_chatgpt_model(st, chatgpt_model)

api_key = load_api_key(st)

temperature = 0.3
qa_chain_func = qa_chain(directory, api_key, chatgpt_model, temperature)

suggested_questions = [
    "What causes thunderstorms?",
    "How does a cold front affect weather?",
    "What is the difference between weather and climate?",
    "How are hurricanes formed?",
    "Why do we see lightning before hearing thunder?"
]

cols = st.columns(len(suggested_questions))
st.title("Mr. Weather")
st.markdown('To answer all your Meteorology related queries')

if "messages" not in st.session_state:
    st.session_state.messages = []

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and message.get("context"):
            st.markdown("**Context Source:**")
            for idx, doc in enumerate(message["context"]):
                with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                    st.text_area(
                        label=' ',
                        value=doc.page_content,
                        height=150,
                        key=f"context_{i}_{idx}"
                    )

chat_history = []
for i in range(0, len(st.session_state.messages) - 1, 2):
    user_msg = st.session_state.messages[i]
    if i + 1 < len(st.session_state.messages):
        assistant_msg = st.session_state.messages[i + 1]
        chat_history.append((
            "user", user_msg["content"],
            "assistant", assistant_msg["content"],
        ))

if prompt := st.chat_input("Ask me anything about meteorology!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, 'context': None})
    response = qa_chain_func.invoke({"question": prompt, "chat_history": chat_history})
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
        st.markdown("**Context Source:**")
        if response.get("source_documents"):
            for idx, doc in enumerate(response["source_documents"]):
                with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                    st.text_area(
                        label= ' ',
                        value=doc.page_content,
                        height=150,
                        key=f"latest_context_{idx}"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response['answer'],
        "context": response.get("source_documents", [])
    })