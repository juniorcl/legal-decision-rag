from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(folder):
    """Load and combine the PDF documents in a directory.

    Parameters
    ----------
    folder : str or pathlib.Path
        Directory containing the PDF files to process.

    Returns
    -------
    list of Document
        Pages loaded from all PDF files found.
    """

    pdfs = Path(folder).glob("*.pdf")

    docs = []

    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        documents = loader.load()
        docs.extend(documents)

    return docs