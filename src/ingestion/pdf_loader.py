from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(folder):

    pdfs = Path(folder).glob("*.pdf")

    docs = []

    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        documents = loader.load()
        docs.extend(documents)

    return docs