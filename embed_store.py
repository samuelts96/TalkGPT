# embed_store.py

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import pickle

def create_or_load_vectorstore(documents, persist_path="vectorstore/faiss_index", embedding_model="text-embedding-3-small"):
    os.makedirs("vectorstore", exist_ok=True)

    index_file = persist_path + ".faiss"
    store_file = persist_path + ".pkl"

    # Try to load existing index
    if os.path.exists(index_file) and os.path.exists(store_file):
        with open(store_file, "rb") as f:
            store = pickle.load(f)
        return store

    # Otherwise, embed and save
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model, openai_api_key=os.environ["OPENAI_API_KEY"])
    store = FAISS.from_documents(split_docs, embeddings)

    store.save_local(persist_path)
    with open(store_file, "wb") as f:
        pickle.dump(store, f)

    return store
