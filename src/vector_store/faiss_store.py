from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vector_store")

    return db


def load_vector_store(embeddings):

    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )