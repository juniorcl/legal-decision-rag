from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from src.prompts.legal_prompt import build_prompt


def analyze_query(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Classify the question as simple, complex, or ambiguous.

    Parameters
    ----------
    state : dict
        Workflow state containing the question in `question`.
    llm : Any
        Language model used to classify the question.

    Returns
    -------
    dict
        State update containing the classification in `query_type`.
    """
    question = state["question"].strip().lower()

    prompt = build_prompt()

    response = llm.invoke(
        prompt.format(
            context="",
            question=(
                "Classifique a pergunta abaixo como 'simple', 'complex' ou 'ambiguous'. "
                "Responda apenas com um desses termos.\n\nPergunta: "
                f"{question}"
            ),
        )
    )

    query_type = str(response.content).strip().lower()
    
    return {"query_type": query_type}


def decompose_query(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Split a complex question into retrieval subquestions.

    Parameters
    ----------
    state : dict
        Workflow state containing the question in `question`.
    llm : Any
        Language model used to generate the subquestions.

    Returns
    -------
    dict
        State update containing the subquestions in `sub_questions`.
    """
    prompt = build_prompt()
    
    response = llm.invoke(
        prompt.format(
            context="",
            question=(
                "Quebre a pergunta em 2 a 4 sub-perguntas curtas e objetivas. "
                "Responda em formato de lista numerada.\n\nPergunta: "
                f"{state['question']}"
            ),
        )
    )

    lines = [line.strip() for line in str(response.content).splitlines() if line.strip()]
    sub_questions = []
    
    for line in lines:
        if line[0].isdigit():
            sub_questions.append(line.split('.', 1)[1].strip() if '.' in line else line)
    
    if not sub_questions:
        sub_questions = [state["question"]]
    
    return {"sub_questions": sub_questions}


def retrieve_documents(state: dict[str, Any], retriever_func: Any, db: Any = None) -> dict[str, Any]:
    """Retrieve documents using the question or its subquestions.

    Parameters
    ----------
    state : dict
        Workflow state with `question` and optional `sub_questions`.
    retriever_func : callable
        Function that accepts the query, store, and result limit.
    db : Any, default=None
        Store passed to the retrieval function.

    Returns
    -------
    dict
        State update containing the documents in `retrieved_docs`.
    """
    query = state["question"]
    
    if state.get("sub_questions"):
        query = " ".join(state["sub_questions"])
    
    docs = retriever_func(query, db, k=10)
    
    return {"retrieved_docs": docs}


def rerank_documents(state: dict[str, Any], reranker_func: Any) -> dict[str, Any]:
    """Rerank retrieved documents according to their relevance.

    Parameters
    ----------
    state : dict
        Workflow state with the question and retrieved documents.
    reranker_func : callable
        Function that ranks documents for the given question.

    Returns
    -------
    dict
        State update containing the documents in `reranked_docs`.
    """
    docs = state.get("retrieved_docs", [])
    reranked_docs = reranker_func(state["question"], docs, top_k=5)
    return {"reranked_docs": reranked_docs}


def generate_answer(state: dict[str, Any], llm: Any, prompt_template: str) -> dict[str, Any]:
    """Generate an answer using the question and reranked context.

    Parameters
    ----------
    state : dict
        Workflow state with the question and reranked documents.
    llm : Any
        Language model used to generate the answer.
    prompt_template : str
        Template with the `{context}` and `{question}` fields.

    Returns
    -------
    dict
        State update containing the context and generated answer.
    """
    context = "\n\n".join(doc.page_content for doc in state.get("reranked_docs", []))
    response = llm.invoke(prompt_template.format(context=context, question=state["question"]))
    return {"context": context, "answer": str(response.content)}


def critique_answer(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Evaluate the answer and determine whether it meets the quality threshold.

    Parameters
    ----------
    state : dict
        Workflow state with the question, answer, and context.
    llm : Any
        Language model used to score the answer.

    Returns
    -------
    dict
        State update containing the critique and satisfaction indicator.
    """
    critique_text = "Score: 7/10"
    
    prompt = build_prompt()

    response = llm.invoke(
        prompt.format(
            context=state.get("context", ""),
            question=(
                "Avalie a resposta abaixo em uma escala de 0 a 10 para completude, "
                "fundamentação e precisão. Responda no formato 'Score: X/10'.\n\n"
                f"Resposta: {state.get('answer', '')}"
            ),
        )
    )

    critique_text = str(response.content)

    print(f"[critique] {critique_text}")
    
    score_text = critique_text.lower()
    
    if "score:" in score_text:
        try:
            score = int(score_text.split(":", 1)[1].split("/", 1)[0].strip())
        except ValueError:
            score = 7
    
    else:
        score = 7
    
    return {
        "critique": critique_text,
        "is_satisfactory": score >= 7,
    }


def rewrite_query(state: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Rewrite the question to improve context retrieval.

    Parameters
    ----------
    state : dict
        Workflow state with the current question and iteration count.
    llm : Any
        Language model used to rewrite the question.

    Returns
    -------
    dict
        State update containing the new question and incremented iteration count.
    """
    prompt = build_prompt()
    
    response = llm.invoke(
        prompt.format(
            context=state.get("context", ""),
            question=(
                "Reformule a pergunta abaixo para melhorar a recuperação de contexto. "
                "Responda apenas com a pergunta reformulada.\n\nPergunta: "
                f"{state['question']}"
            ),
        )
    )
    
    next_iterations = state.get("iterations", 0) + 1
    
    return {
        "question": str(response.content).strip(),
        "iterations": next_iterations,
    }
