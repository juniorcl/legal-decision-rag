from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, StateGraph, START

from src.graph.nodes import (
    analyze_query,
    critique_answer,
    decompose_query,
    generate_answer,
    rerank_documents,
    retrieve_documents,
    rewrite_query,
)
from src.graph.state import LegalRAGState


def build_legal_rag_graph(llm: Any, prompt: str, retriever_func: Callable[..., list[Any]], reranker_func: Callable[..., list[Any]], db: Any = None):
    """Build and compile the retrieval and legal answer generation workflow.

    Parameters
    ----------
    llm : Any
        Language model used by the analysis and generation nodes.
    prompt : str
        Template used to generate an answer from the retrieved context.
    retriever_func : callable
        Function that retrieves documents for a query.
    reranker_func : callable
        Function that reranks the retrieved documents.
    db : Any, default=None
        Store passed to the retrieval function.

    Returns
    -------
    CompiledStateGraph
        Graph ready to receive the initial state and run the RAG workflow.
    """
    workflow = StateGraph(LegalRAGState)

    workflow.add_node("analyze_query", lambda state: analyze_query(state, llm))
    workflow.add_node("decompose_query", lambda state: decompose_query(state, llm))
    workflow.add_node("retrieve", lambda state: retrieve_documents(state, retriever_func, db))
    workflow.add_node("rerank", lambda state: rerank_documents(state, reranker_func))
    workflow.add_node("generate", lambda state: generate_answer(state, llm, prompt))
    workflow.add_node("critique", lambda state: critique_answer(state, llm))
    workflow.add_node("rewrite_query", lambda state: rewrite_query(state, llm))

    workflow.add_edge(START, "analyze_query")

    def route_after_analysis(state: LegalRAGState) -> str:
        """Select the next node based on the question classification.

        Parameters
        ----------
        state : LegalRAGState
            State containing the identified question type.

        Returns
        -------
        str
            Name of the decomposition, rewrite, or retrieval node.
        """
        if state.get("query_type") == "complex":
            return "decompose_query"
        
        if state.get("query_type") == "ambiguous":
            return "rewrite_query"
        
        return "retrieve"

    workflow.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "decompose_query": "decompose_query",
            "rewrite_query": "rewrite_query",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("decompose_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "critique")

    def route_after_critique(state: LegalRAGState) -> str:
        """Decide whether the workflow ends or retries after the critique.

        Parameters
        ----------
        state : LegalRAGState
            State containing the evaluation, question type, and iteration count.

        Returns
        -------
        str
            Next node identifier, or `END` when the workflow should terminate.
        """
        if state.get("query_type") == "ambiguous":
            if state.get("iterations", 0) >= state.get("max_iterations", 2):
                return END
            return "rewrite_query"
        
        if state.get("is_satisfactory"):
            return END
        
        if state.get("iterations", 0) >= state.get("max_iterations", 2):
            return END
        
        return "rewrite_query"

    workflow.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            END: END,
            "rewrite_query": "rewrite_query",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")

    return workflow.compile()
