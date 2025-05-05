import streamlit as st
from loader import load_documents
from embed_store import create_or_load_vectorstore
from rag_chain import build_qa_chain, qa_chain
import os 
from dotenv import load_dotenv
from config import load_API_KEY
from langchain_openai import OpenAIEmbeddings
from prompts import RESTRICTED_CONTEXT_PROMPT, TEACHING_PROMPT,FRIENDLY_PROMPT,CONCISE_PROMPT
#from langchain.prompts import PromptTemplate

# Load API Key from .env
load_dotenv()  # This ensures the .env file is loaded and environment variables are set

api_key = load_API_KEY()  # Load API key using the helper function from config.py

# Check if the API key is loaded properly (for debugging purposes)
if not api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file.")
else:
    print(f"API Key loaded successfully: {api_key[:5]}...")  # Print the first few characters of the API key to confirm


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
    print("***** SOURCE DOCUMENTS *****")
    print(response['source_documents'])
    print("***** ANSWER *****")
    print(response['answer'])
     # Debug: show retrieved documents
    for i, doc in enumerate(response["source_documents"]):
        print(f"\n--- Retrieved Document {i+1} ---\n")
        print(doc.page_content[:500])
    # print(response.source_documents)
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})



# Define function to create/load the vectorstore
def create_or_load_vectorstore(documents):
    embedding_model = "text-embedding-ada-002"
    
    # Use the API key to create embeddings
    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)

    # Add code to handle documents and create/load vectorstore as needed
    vectorstore = None  # Placeholder - actual implementation will depend on your setup
    return vectorstore


prompt_style = st.sidebar.selectbox("Select prompt style", ["Restricted", "Teaching", "Friendly", "Concise"])

if prompt_style == "Teaching":
    selected_prompt = TEACHING_PROMPT
elif prompt_style == "Friendly":
    selected_prompt = FRIENDLY_PROMPT
elif prompt_style == "Concise":
    selected_prompt = CONCISE_PROMPT
else:
    selected_prompt = RESTRICTED_CONTEXT_PROMPT