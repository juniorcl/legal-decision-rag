from src.ingestion.chunking import semantic_chunking
from src.ingestion.pdf_loader import load_pdfs
from src.vectorstore.faiss_store import create_vector_store
from src.embeddings.embedding_model import load_embeddings


docs = load_pdfs("data/raw")

chunks = semantic_chunking(docs)

embeddings = load_embeddings()

create_vector_store(chunks, embeddings)