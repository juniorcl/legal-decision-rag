from langchain_core.prompts import ChatPromptTemplate


def build_prompt():
    """Create the prompt template for context-grounded answers.

    Returns
    -------
    ChatPromptTemplate
        Template with context and question fields.
    """

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
