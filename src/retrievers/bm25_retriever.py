from langchain_community.retrievers import BM25Retriever


def get_bm25_retriever(db, k=10):
    """Create a BM25 retriever from the vector store's documents.

    Parameters
    ----------
    db : VectorStore
        Store whose documents will be indexed by BM25.
    k : int, default=10
        Number of documents returned per query.

    Returns
    -------
    BM25Retriever
        Lexical retriever configured to return `k` documents.
    """
    chunks = list(db.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return bm25_retriever