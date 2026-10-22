"""Langfuse instrumentation for the legal RAG application.

The helpers in this module keep the tracing code in one place and make
Langfuse optional: when credentials are not configured, every function
behaves as a no-op and the RAG workflow keeps running normally.
"""

from __future__ import annotations

import getpass
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

TRACE_NAME = "legal-rag-interaction"
DEFAULT_SESSION_ID = "legal-decision-rag-cli"
DEFAULT_TAGS = ["legal-decision-rag", "rag"]


def init_langfuse() -> tuple[Any | None, Any | None]:
    """Load the environment and create the Langfuse client and callback handler.

    Returns
    -------
    tuple
        `(client, handler)`. Both values are `None` when the Langfuse
        credentials are not present, which disables tracing without breaking
        the application.
    """
    load_dotenv()

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    tracing_enabled = os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() not in (
        "false",
        "0",
    )

    if not public_key or not secret_key:
        print("Langfuse: credenciais não encontradas, rastreamento desativado.")
        return None, None

    if not tracing_enabled:
        print("Langfuse: rastreamento desativado por LANGFUSE_TRACING_ENABLED.")
        return None, None

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    client = get_client()
    handler = CallbackHandler()

    print(
        "Langfuse: rastreamento ativo em "
        f"{os.getenv('LANGFUSE_BASE_URL', 'cloud')}"
    )

    return client, handler


def build_run_config(
    handler: Any | None,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the LangChain/LangGraph config that enables Langfuse callbacks.

    Parameters
    ----------
    handler : Any | None
        Callback handler returned by `init_langfuse`.
    metadata : dict, optional
        Extra metadata attached to the graph run.

    Returns
    -------
    dict
        Configuration passed to `graph.invoke`.
    """
    config: dict[str, Any] = {
        "run_name": "legal-rag-graph",
        "metadata": {"component": "legal-rag", **(metadata or {})},
    }

    if handler is not None:
        config["callbacks"] = [handler]

    return config


def _resolve_user_id() -> str:
    """Return the user identifier used to group traces in Langfuse."""
    return os.getenv("LANGFUSE_USER_ID") or getpass.getuser() or "local-user"


def _resolve_session_id() -> str:
    """Return the session identifier, grouping CLI runs by default."""
    return os.getenv("LANGFUSE_SESSION_ID") or DEFAULT_SESSION_ID


@contextmanager
def trace_interaction(
    client: Any | None,
    question: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[Any | None]:
    """Create the root Langfuse span for one user question.

    Parameters
    ----------
    client : Any | None
        Langfuse client. When `None`, the context manager is a no-op.
    question : str
        Question submitted by the user.
    user_id : str, optional
        Overrides `LANGFUSE_USER_ID`.
    session_id : str, optional
        Overrides `LANGFUSE_SESSION_ID`.

    Yields
    ------
    Any | None
        Root span that can later be updated with the workflow result.
    """
    if client is None:
        yield None
        return

    from langfuse import propagate_attributes

    with propagate_attributes(
        user_id=user_id or _resolve_user_id(),
        session_id=session_id or _resolve_session_id(),
        trace_name=TRACE_NAME,
        tags=DEFAULT_TAGS,
        metadata={"app": "legal-decision-rag"},
    ):
        with client.start_as_current_observation(
            as_type="span",
            name=TRACE_NAME,
            input={"question": question},
        ) as span:
            try:
                yield span
            except Exception as exc:
                span.update(level="ERROR", status_message=str(exc))
                raise


def _extract_score(critique: str | None) -> float | None:
    """Parse the numeric score from the critique text (e.g. `Score: 7/10`)."""
    if not critique:
        return None

    match = re.search(r"score:\s*(\d+(?:[.,]\d+)?)", critique.lower())

    if match is None:
        return None

    return float(match.group(1).replace(",", "."))


def _is_span_recording(span: Any) -> bool:
    """Return whether the underlying OpenTelemetry span accepts updates."""
    otel_span = getattr(span, "_otel_span", None)
    is_recording = getattr(otel_span, "is_recording", None)

    if not callable(is_recording):
        return True

    return bool(is_recording())


def record_result(client: Any | None, span: Any | None, result: dict[str, Any]) -> None:
    """Attach the workflow result and critique score to the active trace.

    Parameters
    ----------
    client : Any | None
        Langfuse client. When `None`, the function is a no-op.
    span : Any | None
        Root span yielded by `trace_interaction`.
    result : dict
        Final state returned by the LangGraph workflow.
    """
    if client is None or span is None or not _is_span_recording(span):
        return

    critique = result.get("critique")
    score = _extract_score(critique)

    span.update(
        output={
            "answer": result.get("answer"),
            "critique": critique,
        },
        metadata={
            "query_type": result.get("query_type"),
            "iterations": result.get("iterations", 0),
            "is_satisfactory": result.get("is_satisfactory", False),
            "final_question": result.get("question"),
        },
    )

    if score is not None:
        client.score_current_trace(
            name="critique_score",
            value=score,
            data_type="NUMERIC",
            comment=critique,
        )
