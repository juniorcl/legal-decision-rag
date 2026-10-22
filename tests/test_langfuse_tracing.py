import os
import unittest
from unittest import mock

from src.observability import langfuse_tracing as tracing


class LangfuseTracingTests(unittest.TestCase):
    def test_init_without_credentials_disables_tracing(self):
        """Verify that missing credentials turn tracing into a no-op."""
        with mock.patch.object(tracing, "load_dotenv", return_value=False):
            with mock.patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": ""}):
                os.environ.pop("LANGFUSE_SECRET_KEY", None)
                client, handler = tracing.init_langfuse()

        self.assertIsNone(client)
        self.assertIsNone(handler)

    def test_trace_interaction_is_noop_without_client(self):
        """Verify that the tracing context works without a configured client."""
        with tracing.trace_interaction(None, "Qual é a regra?") as span:
            self.assertIsNone(span)

        tracing.record_result(None, span, {"answer": "Resposta"})

    def test_build_run_config_adds_callbacks_when_handler_exists(self):
        """Verify that the callback handler is forwarded to the graph config."""
        handler = object()

        config = tracing.build_run_config(handler, metadata={"question": "teste"})

        self.assertEqual(config["run_name"], "legal-rag-graph")
        self.assertEqual(config["callbacks"], [handler])
        self.assertEqual(
            config["metadata"],
            {"component": "legal-rag", "question": "teste"},
        )

    def test_extract_score_parses_numeric_critique(self):
        """Verify that the critique score is extracted for Langfuse scores."""
        self.assertEqual(tracing._extract_score("Score: 7/10"), 7.0)
        self.assertEqual(tracing._extract_score("score: 8,5/10"), 8.5)
        self.assertIsNone(tracing._extract_score("sem nota"))
        self.assertIsNone(tracing._extract_score(None))


if __name__ == "__main__":
    unittest.main()
