# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF Q&A Chatbot using Hybrid RAG (BM25 + Semantic search) over PDF documents. Users upload PDFs via a Streamlit UI, which are chunked and indexed; questions are answered by combining keyword (BM25, weight 0.4) and semantic (ChromaDB, weight 0.6) retrieval via LangChain's EnsembleRetriever, with Google Gemini 2.5 Flash generating final answers.

## Commands

```bash
# Install dependencies (uv recommended, Python 3.12+)
uv sync
# Or: pip install -r requirements.txt

# Run the application (starts both Streamlit UI and FastAPI backend)
streamlit run app.py

# The FastAPI server starts automatically on port 8001 via a daemon thread in app.py
# Streamlit UI is available at http://localhost:8501
# FastAPI docs (Swagger) at http://localhost:8001/
```

## Architecture

Two-file application with `app.py` as the entry point:

- **`app.py`** (Streamlit frontend) — Launches the FastAPI backend in a background daemon thread on first run (`st.session_state.api_started`), then communicates with it over HTTP on `localhost:8001`. Manages chat history in `st.session_state.messages`.

- **`api.py`** (FastAPI backend) — Contains all RAG logic. On import, it initializes HuggingFace embeddings (`all-MiniLM-L6-v2`), ChromaDB vectorstore (persisted in `data/`), and the Gemini LLM. It loads existing chunks from ChromaDB into an in-memory `documents_store` list at startup. Two endpoints:
  - `POST /postPDF` — Accepts PDF upload, extracts text via PyPDFLoader, splits into 1000-char chunks (200 overlap), adds to both ChromaDB and in-memory `documents_store`
  - `POST /askQuestion` — Builds a hybrid retriever each call (BM25 from `documents_store` + Chroma semantic), runs RetrievalQA chain, logs detailed ranking scores from both retrievers

Key detail: The BM25 retriever is rebuilt on every question from `documents_store` (the in-memory document list), so both retrieval paths stay in sync when new PDFs are uploaded.

## Environment

Requires a `.env` file with `GOOGLE_API_KEY` (see `.envExample`). Loaded via `python-dotenv` in `api.py`.

## Data

ChromaDB persistence lives in `data/` (gitignored). Deleting this directory resets all indexed documents.
