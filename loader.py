from pathlib import Path
from langchain.schema import Document
from bs4 import BeautifulSoup
from langchain.document_loaders import PyMuPDFLoader

def load_html_utf8(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
        text = soup.get_text()
        return [Document(page_content=text)]

def load_documents(doc_dir = "./docs"):
    html_file = Path(doc_dir) / "meteorology_ebook.html"
    pdf_file = Path(doc_dir) / "ngc.pdf"

    docs = []
    if html_file.exists():
        docs.extend(load_html_utf8(html_file))
    if pdf_file.exists():
        loader = PyMuPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    return docs
