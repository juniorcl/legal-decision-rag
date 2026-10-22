from src.embeddings.embedding_model import load_embeddings
from src.graph.graph import build_legal_rag_graph
from src.llm.model import load_llm
from src.observability.langfuse_tracing import (
    build_run_config,
    init_langfuse,
    record_result,
    trace_interaction,
)
from src.prompts.legal_prompt import build_prompt
from src.rerankers.cross_reranker import apply_cross_rerank
from src.retrievers.hybrid_retriever import get_hybrid_retriever
from src.vector_store.faiss_store import load_vector_store


def main():
    """Run the interactive legal question-answering workflow.

    Load the RAG components, compile the processing graph, prompt the user
    for a question, and display the generated answer and evaluation.
    """
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
        retriever_func=get_hybrid_retriever,
        reranker_func=apply_cross_rerank,
        db=db,
    )

    langfuse_client, langfuse_handler = init_langfuse()

    try:
        query = input("\nPergunta: ")

        print("Analisando..")

        with trace_interaction(langfuse_client, query) as trace:
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
                },
                config=build_run_config(
                    langfuse_handler,
                    metadata={"question": query},
                ),
            )

            record_result(langfuse_client, trace, result)

        print("\n -------------- Resposta -------------- \n", result.get("answer"))
        print("\n -------------- Critique -------------- \n", result.get("critique"))
    finally:
        if langfuse_client is not None:
            langfuse_client.flush()


if __name__ == "__main__":
    main()