import unittest
from types import SimpleNamespace

from src.graph.graph import build_legal_rag_graph


class StubLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(str(prompt))
        text = str(prompt).lower()
        if "classifique" in text:
            return SimpleNamespace(content="ambiguous")
        if "sub-perguntas" in text:
            return SimpleNamespace(content="1. Qual é a base legal?\n2. Quais são os precedentes?")
        if "reformule" in text:
            return SimpleNamespace(content="Pergunta reformulada")
        if "avalie" in text or "critique" in text:
            return SimpleNamespace(content="Score: 7/10")
        return SimpleNamespace(content="Resposta")


class LangGraphTests(unittest.TestCase):
    def test_simple_query_completes_graph(self):
        graph = build_legal_rag_graph(
            llm=StubLLM(),
            prompt="Contexto: {context}\nPergunta: {question}",
            retriever_func=lambda query, db=None, k=10: [SimpleNamespace(page_content="contexto")],
            reranker_func=lambda query, docs, top_k=5: docs,
        )

        result = graph.invoke(
            {
                "question": "Qual é a regra geral?",
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

        self.assertEqual(result["query_type"], "simple")
        self.assertEqual(result["answer"], "Resposta")
        self.assertIn("Score:", result["critique"])

    def test_ambiguous_query_rewrites_until_iteration_limit(self):
        graph = build_legal_rag_graph(
            llm=StubLLM(),
            prompt="Contexto: {context}\nPergunta: {question}",
            retriever_func=lambda query, db=None, k=10: [SimpleNamespace(page_content="contexto")],
            reranker_func=lambda query, docs, top_k=5: docs,
        )

        result = graph.invoke(
            {
                "question": "Esta regra é aplicável?",
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

        self.assertEqual(result["query_type"], "ambiguous")
        self.assertEqual(result["iterations"], 2)
        self.assertIn("Pergunta reformulada", result["question"])


if __name__ == "__main__":
    unittest.main()
