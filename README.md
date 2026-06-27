# Production Grade RAG Application

A production-ready AI application designed for high-accuracy document retrieval and generation (RAG). This application is built with hybrid search, re-ranking, and strict citation enforcement to ensure enterprise-level reliability.

## 🚀 Key Features

*   **Hybrid Retrieval**: Combines BM25 keyword search with Vector semantic search for maximum recall.
*   **Cross-Encoder Reranking**: Utilizes state-of-the-art re-rankers to prioritize the most relevant context.
*   **Citation Enforcement**: Every response is grounded in provided documents with traceablity to sources.
*   **Production Architecture**: Built with FastAPI, Dependency Injection, and a decoupled component model.
*   **Evaluation Pipeline**: CI-gated quality checks to prevent regressions in retrieval accuracy.

## 🛠️ Tech Stack

*   **Framework**: FastAPI (Python 3.11)
*   **RAG Engine**: LlamaIndex
*   **Vector DB**: Qdrant (Local/Remote)
*   **LLM Providers**: Ollama, OpenAI, Azure, Gemini, SageMaker
*   **UI**: Gradio

## 🚦 Getting Started

### Installation

```bash
# Install dependencies with production extras
poetry install --extras "ui llms-ollama embeddings-ollama vector-stores-qdrant"
```

### Running the Application

To run the application with the default production settings:

```bash
# Set profiles and start
$env:PGPT_PROFILES="ollama"
poetry run python -m production_rag
```

## ⚙️ Configuration

Configuration is managed via YAML profiles in the root directory:
- `settings.yaml`: Global defaults.
- `settings-ollama.yaml`: Configuration for Ollama local execution.
- `settings-local.yaml`: Configuration for local LlamaCPP/HuggingFace execution.

## 🧪 Evaluation

Run the evaluation pipeline to verify retrieval quality:

```bash
python production_rag/rag/evaluation/evaluator.py --threshold 0.7
```

## 🛡️ License

This project is licensed under the Apache-2.0 License.

