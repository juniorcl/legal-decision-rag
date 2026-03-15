from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings.embedding_model import load_embeddings
from langchain_experimental.text_splitter import SemanticChunker


def recursive_chunking(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    return splitter.split_documents(docs)

def semantic_chunking(docs):

    embeddings = load_embeddings()
    
    splitter = SemanticChunker(embeddings)

    return splitter.split_documents(docs)
