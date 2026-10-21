from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings.embedding_model import load_embeddings
from langchain_experimental.text_splitter import SemanticChunker


def recursive_chunking(docs):
    """Split documents into recursively generated chunks of limited size.

    Parameters
    ----------
    docs : list of Document
        Documents to split into chunks.

    Returns
    -------
    list of Document
        Chunks of up to 200 characters with a 100-character overlap.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(docs)

def semantic_chunking(docs):
    """Split documents into chunks based on semantic similarity.

    Parameters
    ----------
    docs : list of Document
        Documents to split semantically.

    Returns
    -------
    list of Document
        Chunks produced by the semantic splitter.
    """

    embeddings = load_embeddings()
    
    splitter = SemanticChunker(embeddings)

    return splitter.split_documents(docs)
