def get_vector_retriever(db, k=10):
    """Create a document retriever based on vector search.

    Parameters
    ----------
    db : VectorStore
        Vector store queried by the retriever.
    k : int, default=10
        Maximum number of documents returned per query.

    Returns
    -------
    VectorStoreRetriever
        Retriever configured to return the `k` nearest documents.
    """

    return db.as_retriever(
        search_kwargs={"k": k}
    )
