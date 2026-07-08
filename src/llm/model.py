from langchain_ollama import ChatOllama


def load_llm():

    return ChatOllama(
        model="qwen2.5:7b-instruct"
    )