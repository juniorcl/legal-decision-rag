from langchain_ollama import ChatOllama


def load_llm():
    """Load the local language model configured for the project.

    Returns
    -------
    ChatOllama
        `qwen2.5:7b-instruct` model ready to receive requests.
    """

    return ChatOllama(
        model="qwen2.5:7b-instruct"
    )