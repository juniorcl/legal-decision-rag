from sentence_transformers import CrossEncoder


RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def apply_cross_rerank(query, docs, top_k=5):

    pairs = [(query, doc.page_content) for doc in docs]
    
    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for _, doc in ranked[:top_k]]