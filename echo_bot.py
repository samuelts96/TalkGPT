import streamlit as st
from loader import load_documents
from embed_store import create_or_load_vectorstore
from rag_chain import build_qa_chain, qa_chain
import os 

qa_chain_func = qa_chain()
# documents = load_documents()
# vectorstore = create_or_load_vectorstore(documents)
# qa_chain = build_qa_chain(vectorstore, openai_api_key=os.environ["OPENAI_API_KEY"])

st.title("Echo Bot")

# documents = load_documents()
# vectorstore = create_or_load_vectorstore(documents)
# retriever = vectorstore.as_retriever()
# qa_chain = build_qa_chain(vectorstore, openai_api_key=os.environ["OPENAI_API_KEY"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Extract chat history (just user & assistant content)
chat_history = [
    (m["content"], st.session_state.messages[i + 1]["content"])
    for i, m in enumerate(st.session_state.messages)
    if m["role"] == "user" and i + 1 < len(st.session_state.messages)
]

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # response = f"Echo: {prompt}"
    response = qa_chain_func({"question": prompt, "chat_history": chat_history})    # Display assistant response in chat message container
    # print(response.source_documents)
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
