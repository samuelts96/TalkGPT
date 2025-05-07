import os
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from prompts import RESTRICTED_CONTEXT_PROMPT
from loader import retrieve_file_load_knowledge_base
from embed_store import create_or_load_vectorstore

def qa_chain(directory, api_key=None, model="gpt-4.1-nano-2025-04-14", temperature=0.1):
    documents = retrieve_file_load_knowledge_base(directory)
    vectorstore = create_or_load_vectorstore(documents)
    qa_chain = build_qa_chain(vectorstore, api_key=api_key, model=model, temperature=temperature)
    return qa_chain

def build_qa_chain(vectorstore, api_key=None, model="gpt-4.1-nano-2025-04-14", temperature=0.1):
    print("TEMPERATURE IS : " + str(temperature))
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        temperature=temperature
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RESTRICTED_CONTEXT_PROMPT}
    )
    return chain