import streamlit as st
from loader import load_documents
from embed_store import create_or_load_vectorstore
from rag_chain import build_qa_chain, qa_chain
import os

qa_chain_func = qa_chain()

st.title("Echo Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

            # Flattened: one expander per source doc, directly under response
            if message.get("context"):
                
                for idx, doc in enumerate(message["context"]):
                    st.write(doc.metadata)
                    st.markdown("**Context Source:**")
                    with st.expander(f"📄 Document {idx + 1} — {doc.metadata.get('source', 'Unknown')}"):
                        print(doc)
                        st.text_area(
                            label=' ',
                            value=doc.metadata,
                            height=150,
                            key=f"context_{i}_{idx}"
                        )

chat_history = []
for i in range(0, len(st.session_state.messages) - 1, 2):
    user_msg = st.session_state.messages[i]
    if i + 1 < len(st.session_state.messages):
        assistant_msg = st.session_state.messages[i + 1]
        chat_history.append({
            "user": user_msg["content"],
            "assistant": assistant_msg["content"],
            # "context": assistant_msg.get("context", None)
        })

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, 'context': None})

    response = qa_chain_func({"question": prompt, "chat_history": chat_history})

    with st.chat_message("assistant"):
        st.markdown(response['answer'])

        # Show each source document as a separate expander
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