from langchain_core.prompts import ChatPromptTemplate


def build_prompt():

    return ChatPromptTemplate.from_template(
        """
        Você é um assistente que responde perguntas usando apenas o contexto.

        Contexto:
        {context}

        Pergunta:
        {question}

        Resposta:
        """
    )
