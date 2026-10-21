from src.retrievers.bm25_retriever import get_bm25_retriever
from src.retrievers.vector_retriever import get_vector_retriever


def get_hybrid_retriever(query, db, k=10):
    """Retrieve documents by combining BM25 and vector search.

    Parameters
    ----------
    query : str
        Query used by both retrieval strategies.
    db : VectorStore
        Vector store that provides documents and semantic search results.
    k : int, default=10
        Maximum number of unique documents returned.

    Returns
    -------
    list of Document
        Documents deduplicated by page content and limited to `k` results.
    """
    k_half = round(k)

    bm25_retriever = get_bm25_retriever(db, k_half)
    vector_retriever = get_vector_retriever(db, k_half)

    docs_bm25 = bm25_retriever.invoke(query)
    docs_vector = vector_retriever.invoke(query)

    docs = docs_bm25 + docs_vector
    unique_docs = {doc.page_content: doc for doc in docs}

    return list(unique_docs.values())[:k]
