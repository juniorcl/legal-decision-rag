from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embeddings):
    """Create and persist a FAISS index for the provided documents.

    Parameters
    ----------
    chunks : list of Document
        Documents to index.
    embeddings : Embeddings
        Model used to convert documents into vectors.

    Returns
    -------
    FAISS
        Vector store created and saved in the `vector_store` directory.
    """

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local("vector_store")

    return db


def load_vector_store(embeddings):
    """Load the persisted FAISS index from the project directory.

    Parameters
    ----------
    embeddings : Embeddings
        Model compatible with the vectors stored in the index.

    Returns
    -------
    FAISS
        Vector store loaded from the `vector_store` directory.
    """

    return FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )