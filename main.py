from src.embeddings.embedding_model import load_embeddings
from src.graph.graph import build_legal_rag_graph
from src.llm.model import load_llm
from src.prompts.legal_prompt import build_prompt
from src.rerankers.cross_reranker import apply_cross_rerank
from src.retrivers.hybrid_retriver import get_hybrid_retriver
from src.vector_store.faiss_store import load_vector_store


def main():
    print("Loading Embeddings...")
    embeddings = load_embeddings()

    print("Loading Vector Store...")
    db = load_vector_store(embeddings)

    print("Loading Prompt...")
    prompt = build_prompt()

    print("Loading LLM..")
    rag = load_llm()

    graph = build_legal_rag_graph(
        llm=rag,
        prompt=prompt,
        retriever_func=get_hybrid_retriver,
        reranker_func=apply_cross_rerank,
        db=db,
    )

    query = input("\nPergunta: ")

    print("Analisando..")

    result = graph.invoke(
        {
            "question": query,
            "query_type": None,
            "sub_questions": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "context": "",
            "answer": None,
            "critique": None,
            "is_satisfactory": False,
            "iterations": 0,
            "max_iterations": 2,
            "conversation_history": [],
        }
    )

    print("\n -------------- Resposta -------------- \n", result.get("answer"))
    print("\n -------------- Critique -------------- \n", result.get("critique"))


if __name__ == "__main__":
    main()