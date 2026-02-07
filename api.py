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
import numpy as np

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag")

# --- Globals ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="data")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents_store: list[Document] = []


def load_documents_from_chroma():
    result = vectorstore.get(include=["documents", "metadatas"])
    for doc_text, metadata in zip(result["documents"], result["metadatas"]):
        documents_store.append(
            Document(page_content=doc_text, metadata=metadata or {})
        )
    logger.info(f"Startup: {len(documents_store)} chunks carregados do ChromaDB")


def build_hybrid_retriever():
    if not documents_store:
        raise ValueError("No documents loaded")

    bm25_retriever = BM25Retriever.from_documents(documents_store, k=4)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return bm25_retriever, chroma_retriever, EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6],
    )


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
        t0 = time.time()

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(contents)
            temp_pdf_path = temp_pdf.name

        loader = PyPDFLoader(temp_pdf_path)
        raw_documents = loader.load()

        chunks = text_splitter.split_documents(raw_documents)

        vectorstore.add_documents(chunks)
        documents_store.extend(chunks)

        os.remove(temp_pdf_path)

        elapsed = time.time() - t0
        logger.info(
            f"Upload '{file.filename}': {len(raw_documents)} paginas, "
            f"{len(chunks)} chunks, {elapsed:.2f}s"
        )

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
        logger.info(f"Pergunta: '{question}'")

        bm25_retriever, chroma_retriever, ensemble_retriever = build_hybrid_retriever()

        # --- BM25 (keyword) ---
        t0 = time.time()
        processed_query = bm25_retriever.preprocess_func(question)
        bm25_all_scores = bm25_retriever.vectorizer.get_scores(processed_query)
        top_indices = np.argsort(bm25_all_scores)[::-1][:bm25_retriever.k]
        bm25_docs = [bm25_retriever.docs[i] for i in top_indices]
        bm25_scores = [float(bm25_all_scores[i]) for i in top_indices]
        t_bm25 = time.time() - t0

        # --- Chroma (semantic) ---
        t0 = time.time()
        chroma_results = vectorstore.similarity_search_with_score(question, k=4)
        chroma_docs = [doc for doc, _ in chroma_results]
        chroma_scores = [float(score) for _, score in chroma_results]
        t_chroma = time.time() - t0

        # --- Log do ranking ---
        logger.info("=" * 60)
        logger.info(f"BM25 (peso 0.4) - {len(bm25_docs)} docs em {t_bm25:.3f}s")
        for i, (doc, score) in enumerate(zip(bm25_docs, bm25_scores), 1):
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            logger.info(f"  #{i}  score={score:.4f}  pg.{page}  | {preview}")
        logger.info("-" * 60)
        logger.info(f"Chroma (peso 0.6) - {len(chroma_docs)} docs em {t_chroma:.3f}s  [menor distancia = mais relevante]")
        for i, (doc, score) in enumerate(zip(chroma_docs, chroma_scores), 1):
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            logger.info(f"  #{i}  dist={score:.4f}  pg.{page}  | {preview}")
        logger.info("=" * 60)

        # --- QA chain com Ensemble ---
        t0 = time.time()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": question})
        t_chain = time.time() - t0

        # --- Log do resultado final ---
        source_docs = result.get("source_documents", [])
        logger.info(f"Ensemble: {len(source_docs)} docs enviados ao LLM")
        for i, doc in enumerate(source_docs, 1):
            page = doc.metadata.get("page", "?")
            bm25_rank = next((r + 1 for r, d in enumerate(bm25_docs) if d.page_content == doc.page_content), None)
            chroma_rank = next((r + 1 for r, d in enumerate(chroma_docs) if d.page_content == doc.page_content), None)
            bm25_tag = f"BM25 #{bm25_rank}" if bm25_rank else "BM25 ---"
            chroma_tag = f"Chroma #{chroma_rank}" if chroma_rank else "Chroma ---"
            preview = doc.page_content[:80].replace("\n", " ")
            logger.info(f"  [{i}] {bm25_tag} | {chroma_tag}  pg.{page}  | {preview}")

        answer = result["result"]
        logger.info(
            f"Tempo total: BM25={t_bm25:.3f}s + Chroma={t_chroma:.3f}s + LLM={t_chain:.2f}s "
            f"= {t_bm25 + t_chroma + t_chain:.2f}s"
        )

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
