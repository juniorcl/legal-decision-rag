from langchain_community.retrievers import BM25Retriever


def get_bm5_retriver(db, k=10):

    chunks = list(db.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    return bm25_retriever