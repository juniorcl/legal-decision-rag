from langchain_huggingface import HuggingFaceEmbeddings


def load_embeddings():
    """Load the embedding model used by the project.

    Returns
    -------
    HuggingFaceEmbeddings
        `all-MiniLM-L6-v2` model configured to generate embeddings.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
