# Legal Decision RAG

A **Hybrid Retrieval-Augmented Generation (RAG)** system for querying and analyzing administrative decisions on fine appeals (recursos de multas). Ask questions in natural language (Portuguese) and get answers grounded in indexed legal documents using local LLMs.

## Architecture

```
PDFs
  ↓
Loader (PyMuPDF)
  ↓
Chunking (Recursive + Semantic)
  ↓
Embeddings (all-MiniLM-L6-v2)
  ↓
FAISS Vector Store
  ↓
Hybrid Retriever (BM25 lexical + FAISS semantic)
  ↓
CrossEncoder Reranker
  ↓
LLM (qwen2.5:7b-instruct via Ollama)
  ↓
Answer
```

Hybrid retrieval combines lexical search (BM25) with semantic search (FAISS), then re-ranks the top results using a CrossEncoder for higher relevance before feeding context to the LLM.

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.13 | Language |
| LangChain / LangChain-Community | RAG orchestration (loaders, retrievers, chains) |
| FAISS | Vector store |
| Sentence-Transformers | Embeddings (`all-MiniLM-L6-v2`) + CrossEncoder reranker (`ms-marco-MiniLM-L-6-v2`) |
| BM25 (rank-bm25) | Lexical retrieval |
| Ollama | Local LLM host (`qwen2.5:7b-instruct`) |
| PyMuPDF | PDF text extraction |
| UV | Dependency management |

## Project Structure

```
legal-decision-rag/
├── data/
│   ├── raw/                    # Source PDFs (18 sample files)
│   └── interim/
│       └── semantic_chunks.pkl # Pre-processed chunks
├── notebooks/
│   ├── create_vector_store.ipynb
│   └── running_rag.ipynb
├── src/
│   ├── embeddings/
│   │   └── embedding_model.py
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── chunking.py
│   ├── llm/
│   │   └── model.py
│   ├── prompts/
│   │   └── legal_prompt.py
│   ├── rerankers/
│   │   └── cross_reranker.py
│   ├── retrivers/
│   │   ├── bm5_retriver.py
│   │   ├── hybrid_retriver.py
│   │   └── vector_retriver.py
│   ├── scripts/
│   │   └── build_index.py
│   └── vector_store/
│       └── faiss_store.py
├── vector_store/               # Serialized FAISS index
│   ├── index.faiss
│   └── index.pkl
├── main.py                     # CLI entry point (REPL loop)
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.ai/) running locally with the model:

```bash
ollama pull qwen2.5:7b-instruct
```

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -r requirements.txt
```

### Place your PDFs

Put your administrative decision PDFs in `data/raw/`:

```
data/raw/recurso_001.pdf
data/raw/recurso_002.pdf
```

### Build the vector index

```bash
uv run python src/scripts/build_index.py
```

This runs: PDF → chunking (recursive + semantic) → embeddings → FAISS. The index is saved to `vector_store/`.

## Usage

```bash
uv run python main.py
```

The REPL loop accepts questions in Portuguese. Example:

```
Pergunta: Quando um recurso pode ser deferido?

Resposta: O recurso pode ser deferido quando houver comprovação de erro na autuação ou irregularidade no processo administrativo.
```

Type `exit` or `quit` to stop.

## Dataset

The `data/raw/` folder includes 18 sample PDFs covering different decision types:

- **Deferimento** (granted) — `recurso_deferimento_02`, `03`, `04`
- **Indeferimento** (denied) — `recurso_indeferimento_05`, `06`, `07`
- **Parcial** (partial) — `recurso_parcial_08`, `09`, `10`
- **Noisy** (with noise) — 9 PDFs with varying quality for robustness testing

## Hybrid Retrieval

| Strategy | Method | Strengths |
|---|---|---|
| Lexical | BM25 | Exact keyword matching, handles legal jargon |
| Semantic | FAISS + all-MiniLM-L6-v2 | Understands meaning, handles synonyms |
| Fusion | Weighted combination + dedup | Best of both worlds |
| Reranking | CrossEncoder (ms-marco-MiniLM-L-6-v2) | Re-ranks top-10 → top-5 by relevance |

## Future Improvements

- FastAPI REST API
- Web interface
- Automated RAG evaluation (RAGAS, etc.)
- Query rewriting
- Multi-query retrieval
- Context compression
- Support for vector databases (Qdrant, Weaviate)

## License

MIT © [Clébio Júnior](LICENSE)
