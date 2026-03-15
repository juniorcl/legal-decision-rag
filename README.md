# Legal Decision RAG

Sistema de **Retrieval-Augmented Generation (RAG)** para consulta e análise de decisões administrativas relacionadas a recursos de multas.

O sistema permite realizar perguntas em linguagem natural e obter respostas baseadas em documentos jurídicos indexados, utilizando recuperação semântica e geração por LLM.

## Arquitetura

O projeto utiliza uma arquitetura de **RAG híbrido**, combinando busca vetorial e busca lexical.

Pipeline:

```
PDFs
↓
Loader
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Retriever
↓
Reranker
↓
LLM
↓
Resposta
```

## Tecnologias utilizadas

* Python
* LangChain
* FAISS
* Sentence Transformers
* BM25 Retriever
* CrossEncoder Reranker
* Ollama (LLM local)
* UV (gerenciador de dependências)

**Principais bibliotecas:**

* langchain
* langchain-community
* faiss-cpu
* sentence-transformers
* pymupdf

## Estrutura do projeto

```
legal-decision-rag/
│
├── data/
│   └── raw/
│       └── pdfs
│
├── vector_store/
│
├── notebooks/
│
├── src/
│
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── chunking.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── vector_store/
│   │   └── faiss_store.py
│   │
│   ├── retrievers/
│   │   └── hybrid_retriever.py
│   │
│   ├── reranker/
│   │   └── cross_encoder.py
│   │
│   ├── prompts/
│   │   └── legal_prompt.py
│   │
│   └── rag/
│       └── rag_chain.py
│
├── scripts/
│   ├── build_index.py
│
└── main.py
```

## Instalação

Este projeto utiliza **uv** para gerenciamento de dependências.

Instalar dependências:

```bash
uv sync
```

Alternativamente:

```bash
pip install -r requirements.txt
```

## Preparação dos dados

Coloque os PDFs na pasta:

```
data/raw/
```

Exemplo:

```
data/raw/recurso_001.pdf
data/raw/recurso_002.pdf
```

# Construção do índice vetorial

Para criar o índice FAISS:

```bash
uv run python scripts/build_index.py
```

Esse processo executa:

```
PDF → chunking → embeddings → FAISS
```

Os arquivos gerados serão salvos em:

```
vector_store/
```

## Executar o sistema

Execute o programa principal:

```bash
uv run python main.py
```

Exemplo de uso:

```
Pergunta:
Quando um recurso pode ser deferido?

Resposta:
O recurso pode ser deferido quando houver comprovação de erro na autuação ou irregularidade no processo administrativo.
```

## Busca híbrida

O sistema suporta **Hybrid Retrieval**, combinando:

* **BM25** → busca lexical
* **FAISS** → busca semântica

Essa abordagem melhora significativamente a recuperação de contexto em documentos legais.

## Reranking

Após a recuperação inicial, os documentos são reordenados utilizando **CrossEncoder Reranker**, aumentando a relevância do contexto enviado à LLM.

Pipeline:

```
Retriever → Top 10 documentos
↓
Reranker
↓
Top 5 documentos
↓
LLM
```

## Exemplo de pergunta

```
Quando um recurso administrativo pode ser indeferido?
```

A resposta será gerada com base nos documentos indexados.

## Melhorias futuras

* API com FastAPI
* Interface web
* Avaliação automática de RAG
* Query rewriting
* Multi-query retrieval
* Context compression
* Suporte a bancos vetoriais (Qdrant, Weaviate)

## Objetivo

Este projeto tem como objetivo explorar técnicas de **RAG aplicadas ao domínio jurídico**, permitindo consultas inteligentes em grandes volumes de decisões administrativas.

# Licença

MIT License
