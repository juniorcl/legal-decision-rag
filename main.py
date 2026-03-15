from src.llm.model import load_llm
from src.prompts.legal_prompt import build_prompt
from src.rerankers.cross_reranker import apply_cross_rerank
from src.vector_store.faiss_store import load_vector_store
from src.retrivers.hybrid_retriver import get_hybrid_retriver
from src.embeddings.embedding_model import load_embeddings


def main():

    print("Loading Embeddings...")
    embeddings = load_embeddings()

    print("Loading Vector Store...")
    db = load_vector_store(embeddings)

    print("Loading Prompt...")
    prompt = build_prompt()

    print("Loading LLM..")
    rag = load_llm()


    while True:

        query = input("\nPergunta: ")

        if query.lower() in ["exit", "quit"]:
            break

        docs = get_hybrid_retriver(query, db)

        reranked_docs = apply_cross_rerank(query, docs)

        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        print("Analisando..")

        response = rag.invoke(prompt.format(context=context, question=query))

        print("\nResposta:\n", response.content)


if __name__ == "__main__":
    main()