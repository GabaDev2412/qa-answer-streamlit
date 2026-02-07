# PDF Q&A Chatbot

Chatbot com interface Streamlit que responde perguntas com base em documentos PDF utilizando RAG Híbrido (BM25 + Semântico). A aplicação combina busca por palavras-chave (BM25) com busca semântica (ChromaDB) para recuperar os trechos mais relevantes dos PDFs e gerar respostas com o Google Gemini.

## Funcionalidades

- Upload de arquivos PDF via interface web
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
│  (app.py)    │                   │  (api.py)                    │
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
```

## Tecnologias

- **[Streamlit](https://docs.streamlit.io/)** — Interface web interativa (chat)
- **[FastAPI](https://fastapi.tiangolo.com/)** — API REST para upload e perguntas
- **[LangChain](https://langchain.com/)** — Orquestração do pipeline RAG
- **[Google Gemini](https://ai.google.dev/)** — LLM para geração de respostas
- **[HuggingFace Embeddings](https://huggingface.co/)** — Embeddings locais (`all-MiniLM-L6-v2`)
- **[ChromaDB](https://docs.trychroma.com/)** — Banco de dados vetorial com persistência
- **[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)** — Retriever por palavras-chave (rank_bm25)
- **[PyPDF2](https://pypdf2.readthedocs.io/)** — Extração de texto de PDFs

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

### Upload de PDF
`POST /postPDF`

Faz upload de um PDF, extrai o texto, divide em chunks e indexa no ChromaDB.

| Parâmetro | Tipo       | Descrição            |
|-----------|------------|----------------------|
| `file`    | UploadFile | Arquivo PDF          |

**Resposta (200):**
```json
{
  "message": "Arquivo carregado com sucesso!",
  "chunks": 42
}
```

### Fazer uma pergunta
`POST /askQuestion`

Busca trechos relevantes via RAG híbrido e gera resposta com Gemini.

| Parâmetro  | Tipo   | Descrição                    |
|------------|--------|------------------------------|
| `question` | string | Pergunta sobre o PDF         |

**Resposta (200):**
```json
{
  "answer": "A resposta baseada no conteúdo do PDF..."
}
```

## Estrutura do Projeto

```
├── app.py               # Interface Streamlit (frontend)
├── api.py               # API FastAPI (backend RAG)
├── requirements.txt     # Dependências (pip)
├── pyproject.toml       # Configuração do projeto (uv)
├── uv.lock              # Lock de dependências (uv)
├── .env                 # Variáveis de ambiente (GOOGLE_API_KEY)
├── .envExample          # Exemplo de .env
├── data/                # Diretório persistente do ChromaDB
└── .streamlit/          # Configurações do Streamlit
```
