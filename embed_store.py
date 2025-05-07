import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def create_or_load_vectorstore(documents, persist_path="vectorstore/faiss_index", embedding_model="text-embedding-3-small"):
    os.makedirs("vectorstore", exist_ok=True)

    index_file = persist_path + "/index.faiss"

    if os.path.exists(index_file):
        print(f"CREATE OR LOAD VECTORSTORE - STORE/FILE {index_file} ALREADY EXISTS")
        return FAISS.load_local(persist_path, OpenAIEmbeddings(model=embedding_model),allow_dangerous_deserialization=True)

    if not documents:
        raise ValueError("No documents provided for embedding.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = FAISS.from_documents(split_docs, embeddings)

    store.save_local(persist_path)
    print(f"CREATE OR LOAD VECTORSTORE - STORE/FILE {index_file} CREATED")

    return store
