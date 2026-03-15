from langchain_community.chat_models import ChatOllama


def load_llm():

    return ChatOllama(
        model="qwen2.5:7b-instruct"
    )