from src.retrivers.bm5_retriver import get_bm5_retriver
from src.retrivers.vector_retriver import get_vector_retriever


def get_hybrid_retriver(query, db, k=10):

    k_half = round(k)

    bm25_retriever = get_bm5_retriver(db, k_half)
    vector_retriever = get_vector_retriever(db, k_half)

    docs_bm25 = bm25_retriever.invoke(query)
    docs_vector = vector_retriever.invoke(query)

    docs = docs_bm25 + docs_vector

    unique_docs = {doc.page_content: doc for doc in docs}

    return list(unique_docs.values())[:k]

