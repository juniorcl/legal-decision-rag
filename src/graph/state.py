from typing import Any, TypedDict

from langchain_core.documents import Document


class LegalRAGState(TypedDict):
    question: str
    query_type: str | None
    sub_questions: list[str]
    retrieved_docs: list[Document]
    reranked_docs: list[Document]
    context: str
    answer: str | None
    critique: str | None
    is_satisfactory: bool
    iterations: int
    max_iterations: int
    conversation_history: list[dict[str, Any]]