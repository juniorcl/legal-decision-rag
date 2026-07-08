from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_model import load_embeddings
from src.ingestion.chunking import semantic_chunking
from src.ingestion.pdf_loader import load_pdfs
from src.vector_store.faiss_store import create_vector_store


docs = load_pdfs("data/raw")

chunks = semantic_chunking(docs)

embeddings = load_embeddings()

create_vector_store(chunks, embeddings)