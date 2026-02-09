# PDF & SQL Q&A Chatbot

Chatbot com interface Streamlit que responde perguntas com base em documentos PDF e arquivos SQL utilizando RAG Híbrido (BM25 + Semântico). A aplicação combina busca por palavras-chave (BM25) com busca semântica (ChromaDB) para recuperar os trechos mais relevantes e gerar respostas com o Google Gemini.

## Funcionalidades

- Upload de arquivos PDF e SQL via interface web
- OCR automático para PDFs escaneados/imagens via Gemini Vision (fallback quando texto não é detectado)
- Processamento em background com progresso em tempo real na sidebar
- Registro de arquivos já processados (evita reprocessamento por hash SHA-256)
- RAG Híbrido: combina BM25 (keyword, peso 0.4) + ChromaDB (semântico, peso 0.6) via EnsembleRetriever
- Embeddings locais com HuggingFace (`all-MiniLM-L6-v2`)
- Geração de respostas com Google Gemini 2.5 Flash
- Persistência dos embeddings em ChromaDB (diretório `data/`)
- Interface de chat com histórico de conversas
- API REST com FastAPI (iniciada automaticamente pelo Streamlit)

## Arquitetura

```
┌──────────────┐       HTTP        ┌──────────────────────────────┐
│  Streamlit   │  ──────────────►  │  FastAPI (porta 8001)        │
│  (app.py)    │  polling status   │  (api.py)                    │
└──────────────┘                   └──────┬───────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                        ┌─────▼─────┐          ┌──────▼──────┐
                        │   BM25    │          │  ChromaDB   │
                        │ (keyword) │          │ (semântico) │
                        └─────┬─────┘          └──────┬──────┘
                              │                       │
                              └───────────┬───────────┘
                                          │
                                 EnsembleRetriever
                                          │
                                   ┌──────▼──────┐
                                   │ Gemini 2.5  │
                                   │   Flash     │
                                   └─────────────┘

Pipeline de upload:
  PDF ──► PyPDFLoader ──► Detecção de páginas sem texto ──► OCR via Gemini Vision (se necessário)
  SQL ──► Leitura direta do texto
  ──► Chunking (1000 chars, 200 overlap) ──► ChromaDB + BM25 (em lotes de 5000)
```

## Tecnologias

- **[Streamlit](https://docs.streamlit.io/)** — Interface web interativa (chat)
- **[FastAPI](https://fastapi.tiangolo.com/)** — API REST para upload e perguntas
- **[LangChain](https://langchain.com/)** — Orquestração do pipeline RAG
- **[Google Gemini](https://ai.google.dev/)** — LLM para geração de respostas + OCR via Vision
- **[HuggingFace Embeddings](https://huggingface.co/)** — Embeddings locais (`all-MiniLM-L6-v2`)
- **[ChromaDB](https://docs.trychroma.com/)** — Banco de dados vetorial com persistência
- **[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)** — Retriever por palavras-chave (rank_bm25)
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — Renderização de páginas PDF para OCR
- **[PyPDF](https://pypdf.readthedocs.io/)** — Extração de texto de PDFs

## Requisitos

- Python 3.12 ou superior
- Chave de API do Google Generative AI

## Instalação e Execução

**1. Clone o repositório:**
```bash
git clone <url-do-repositorio>
cd qa-answer-streamlit
```

**2. Crie um arquivo `.env` na raiz do projeto:**
```bash
GOOGLE_API_KEY="SUA_CHAVE_API"
```

**3. Instale as dependências (com uv ou pip):**
```bash
# Com uv (recomendado)
uv sync

# Ou com pip
pip install -r requirements.txt
```

**4. Inicie a aplicação:**
```bash
streamlit run app.py
```

O Streamlit inicia automaticamente a API FastAPI na porta 8001. A interface web estará disponível em `http://localhost:8501`.

## Endpoints da API

### Upload de arquivo
`POST /uploadFile`

Faz upload de um PDF ou SQL, processa o conteúdo e indexa no ChromaDB. O processamento roda em background — a resposta é imediata com um `file_hash` para acompanhar o progresso.

| Parâmetro | Tipo       | Descrição                    |
|-----------|------------|------------------------------|
| `file`    | UploadFile | Arquivo PDF ou SQL           |

**Resposta (200) — processamento iniciado:**
```json
{
  "message": "Processamento iniciado",
  "status": "processing",
  "file_hash": "728dabeb64123d04"
}
```

**Resposta (200) — arquivo já processado:**
```json
{
  "message": "Arquivo ja processado anteriormente (42 chunks)",
  "chunks": 42,
  "status": "done",
  "file_hash": "728dabeb64123d04"
}
```

### Status do processamento
`GET /statusPDF/{file_hash}`

Consulta o progresso do processamento de um arquivo.

**Resposta (200) — em andamento:**
```json
{
  "status": "processing",
  "filename": "documento.pdf",
  "progress": "OCR: 5/20 paginas...",
  "total_pages": 20,
  "ocr_pages": 15
}
```

**Resposta (200) — concluído:**
```json
{
  "status": "done",
  "filename": "documento.pdf",
  "chunks": 42,
  "elapsed": 8.25
}
```

### Fazer uma pergunta
`POST /askQuestion`

Busca trechos relevantes via RAG híbrido e gera resposta com Gemini.

| Parâmetro  | Tipo   | Descrição                          |
|------------|--------|------------------------------------|
| `question` | string | Pergunta sobre os documentos       |

**Resposta (200):**
```json
{
  "answer": "A resposta baseada no conteúdo dos documentos..."
}
```

## Estrutura do Projeto

```
├── app.py               # Interface Streamlit (frontend + polling)
├── api.py               # API FastAPI (backend RAG + OCR + background processing)
├── requirements.txt     # Dependências (pip)
├── pyproject.toml       # Configuração do projeto (uv)
├── uv.lock              # Lock de dependências (uv)
├── .env                 # Variáveis de ambiente (GOOGLE_API_KEY)
├── .envExample          # Exemplo de .env
├── data/                # ChromaDB + registro de arquivos processados (pdf_registry.json)
└── .streamlit/          # Configurações do Streamlit
```
