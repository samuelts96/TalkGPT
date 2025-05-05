from pathlib import Path
from langchain.schema import Document
from bs4 import BeautifulSoup
from langchain.document_loaders import PyMuPDFLoader, UnstructuredFileLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader, TextLoader
import os

def load_html_utf8(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        text = soup.get_text()
        return [Document(page_content=text, metadata={"source": os.path.basename(filepath)})]

def load_documents(doc_dir="./meteorology_files"):
    supported_extensions = [".pdf", ".txt", ".doc", ".docx", ".html"]
    docs = []

    for filepath in Path(doc_dir).glob("*"):
        if filepath.suffix.lower() not in supported_extensions:
            continue

        try:
            if filepath.suffix == ".pdf":
                loader = PyMuPDFLoader(str(filepath))
                documents = loader.load()
            elif filepath.suffix in [".doc", ".docx"]:
                loader = UnstructuredWordDocumentLoader(str(filepath))
                documents = loader.load()
            elif filepath.suffix == ".txt":
                loader = TextLoader(str(filepath), encoding="utf-8")
                documents = loader.load()
            elif filepath.suffix == ".html":
                documents = load_html_utf8(filepath)
            else:
                continue  # skip unsupported types

            # Add filename as metadata to each doc
            for doc in documents:
                doc.metadata["source"] = filepath.name

            docs.extend(documents)

        except Exception as e:
            print(f"⚠️ Failed to load {filepath.name}: {e}")

    return docs
