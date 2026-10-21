from sentence_transformers import CrossEncoder


RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def apply_cross_rerank(query, docs, top_k=5):
    """Rerank documents by relevance scores estimated by a CrossEncoder.

    Parameters
    ----------
    query : str
        Query used to evaluate document relevance.
    docs : list of Document
        Candidate documents to rerank.
    top_k : int, default=5
        Maximum number of documents to keep after reranking.

    Returns
    -------
    list of Document
        Documents sorted from highest to lowest score.
    """

    pairs = [(query, doc.page_content) for doc in docs]
    
    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]