def get_vector_retriever(db, k=10):

    return db.as_retriever(
        search_kwargs={"k": k}
    )
