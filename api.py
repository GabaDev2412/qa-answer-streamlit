from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from fastapi import FastAPI, HTTPException, File, UploadFile
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
import tempfile
import time
import os

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag")

# --- Globals (initialized once) ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="data")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# In-memory store to feed BM25
documents_store: list[Document] = []


def load_documents_from_chroma():
    """Load existing documents from ChromaDB into memory for BM25."""
    result = vectorstore.get(include=["documents", "metadatas"])
    for doc_text, metadata in zip(result["documents"], result["metadatas"]):
        documents_store.append(
            Document(page_content=doc_text, metadata=metadata or {})
        )
    logger.info(f"Startup: loaded {len(documents_store)} chunks from ChromaDB into memory")


def build_hybrid_retriever():
    """Build an ensemble retriever combining BM25 (keyword) + Chroma (semantic)."""
    if not documents_store:
        raise ValueError("No documents loaded")

    bm25_retriever = BM25Retriever.from_documents(documents_store, k=4)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return bm25_retriever, chroma_retriever, EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6],
    )


def log_docs(label: str, docs: list[Document]):
    """Log retrieved documents with preview."""
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", doc.metadata.get("page", "?"))
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        logger.info(f"  {label} [{i}] page={page} source={source} | {preview}...")


# Load existing docs on startup
load_documents_from_chroma()

app = FastAPI(
    title="PDF Q&A API",
    description="API para chatbot com RAG hibrido (BM25 + semantico) sobre documentos PDF",
    version="0.2",
    docs_url="/",
)


class QuestionRequest(BaseModel):
    question: str


@app.post("/postPDF")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="O arquivo precisa ser um PDF")

    try:
        logger.info(f"Upload: receiving file '{file.filename}'")
        t0 = time.time()

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(contents)
            temp_pdf_path = temp_pdf.name

        loader = PyPDFLoader(temp_pdf_path)
        raw_documents = loader.load()
        logger.info(f"Upload: extracted {len(raw_documents)} pages from PDF")

        chunks = text_splitter.split_documents(raw_documents)
        logger.info(f"Upload: split into {len(chunks)} chunks (size=1000, overlap=200)")

        vectorstore.add_documents(chunks)
        documents_store.extend(chunks)
        logger.info(f"Upload: added to ChromaDB + memory store (total={len(documents_store)} chunks)")

        os.remove(temp_pdf_path)

        elapsed = time.time() - t0
        logger.info(f"Upload: completed in {elapsed:.2f}s")

        return {
            "message": "Arquivo carregado com sucesso!",
            "chunks": len(chunks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/askQuestion")
async def ask_question(request: QuestionRequest):
    if not documents_store:
        raise HTTPException(
            status_code=400,
            detail="Nenhum documento carregado. Faca upload de um PDF primeiro.",
        )

    try:
        question = request.question
        logger.info(f"Question: '{question}'")
        logger.info(f"RAG: building hybrid retriever (docs in memory: {len(documents_store)})")

        bm25_retriever, chroma_retriever, ensemble_retriever = build_hybrid_retriever()

        # --- Log individual retrievers ---
        t0 = time.time()
        bm25_docs = bm25_retriever.invoke(question)
        t_bm25 = time.time() - t0
        logger.info(f"BM25 (keyword, weight=0.4): {len(bm25_docs)} docs in {t_bm25:.3f}s")
        log_docs("BM25", bm25_docs)

        t0 = time.time()
        chroma_docs = chroma_retriever.invoke(question)
        t_chroma = time.time() - t0
        logger.info(f"Chroma (semantic, weight=0.6): {len(chroma_docs)} docs in {t_chroma:.3f}s")
        log_docs("CHROMA", chroma_docs)

        # --- Run the chain with ensemble ---
        logger.info("RAG: running QA chain with EnsembleRetriever...")
        t0 = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": question})
        t_chain = time.time() - t0

        # Log ensemble results
        source_docs = result.get("source_documents", [])
        logger.info(f"Ensemble: {len(source_docs)} docs merged and sent to LLM")
        log_docs("ENSEMBLE", source_docs)

        answer = result["result"]
        logger.info(f"LLM response ({t_chain:.2f}s): {answer[:150].replace(chr(10), ' ')}...")
        logger.info(f"Total RAG pipeline: BM25={t_bm25:.3f}s + Chroma={t_chroma:.3f}s + Chain={t_chain:.2f}s")

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
