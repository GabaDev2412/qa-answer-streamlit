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
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import threading
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

genai_client = genai.Client()

# --- Registro de PDFs processados ---
PDF_REGISTRY_PATH = os.path.join("data", "pdf_registry.json")
# Status em memoria para PDFs sendo processados agora
pdf_processing_status: dict[str, dict] = {}  # hash -> {status, filename, progress, ...}


def _load_pdf_registry() -> dict[str, dict]:
    if os.path.exists(PDF_REGISTRY_PATH):
        with open(PDF_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_pdf_registry(registry: dict[str, dict]):
    os.makedirs(os.path.dirname(PDF_REGISTRY_PATH), exist_ok=True)
    with open(PDF_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


OCR_BATCH_SIZE = 10
OCR_MAX_RETRIES = 3
OCR_MIN_CHARS = 20


def _ocr_single_page(image_bytes: bytes, page_num: int, total: int) -> str:
    """OCR a single page image with Gemini vision, with retry and quality validation."""
    for attempt in range(1, OCR_MAX_RETRIES + 1):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    "Extraia todo o texto desta imagem de documento. "
                    "Retorne apenas o texto extraido, sem comentarios adicionais.",
                ],
            )
            text = response.text.strip() if response.text else ""

            # --- Validacao de qualidade ---
            if len(text) < OCR_MIN_CHARS:
                logger.warning(
                    f"OCR pagina {page_num + 1}: qualidade baixa na tentativa {attempt}/{OCR_MAX_RETRIES} "
                    f"({len(text)} chars < {OCR_MIN_CHARS} minimo)"
                )
                if attempt < OCR_MAX_RETRIES:
                    time.sleep(attempt)  # backoff crescente
                    continue
                logger.warning(f"OCR pagina {page_num + 1}: todas as tentativas retornaram pouco texto, aceitando resultado")

            logger.info(f"OCR pagina {page_num + 1}/{total} concluido ({len(text)} chars extraidos)")
            return text

        except Exception as e:
            logger.error(f"OCR pagina {page_num + 1}: erro na tentativa {attempt}/{OCR_MAX_RETRIES} - {e}")
            if attempt < OCR_MAX_RETRIES:
                time.sleep(attempt)  # backoff crescente
                continue
            logger.error(f"OCR pagina {page_num + 1}: falhou apos {OCR_MAX_RETRIES} tentativas")
            return ""

    return ""


def ocr_pages_with_gemini(pdf_path: str, page_nums: list[int], file_hash: str = "") -> list[str]:
    """OCR pages in parallel batches of OCR_BATCH_SIZE. Runs in a worker thread."""
    pdf_doc = fitz.open(pdf_path)

    # Pre-render all pages to PNG
    page_images: list[tuple[int, bytes]] = []
    for page_num in page_nums:
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 144 DPI
        page_images.append((page_num, pix.tobytes("png")))
    pdf_doc.close()

    results: dict[int, str] = {}  # page_num -> text
    failed_pages: list[int] = []
    done_count = 0

    for batch_start in range(0, len(page_images), OCR_BATCH_SIZE):
        batch = page_images[batch_start:batch_start + OCR_BATCH_SIZE]
        batch_label = f"{batch_start + 1}-{batch_start + len(batch)}/{len(page_images)}"
        logger.info(f"OCR lote {batch_label}: processando {len(batch)} paginas em paralelo...")

        future_to_page = {}
        with ThreadPoolExecutor(max_workers=OCR_BATCH_SIZE) as batch_executor:
            for page_num, img_bytes in batch:
                future = batch_executor.submit(
                    _ocr_single_page, img_bytes, page_num, len(page_nums),
                )
                future_to_page[future] = page_num

            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                text = future.result()
                results[page_num] = text
                if not text:
                    failed_pages.append(page_num + 1)

        done_count += len(batch)
        if file_hash and file_hash in pdf_processing_status:
            pdf_processing_status[file_hash]["progress"] = f"OCR: {done_count}/{len(page_images)} paginas..."

    if failed_pages:
        logger.warning(f"OCR: {len(failed_pages)} paginas falharam ou retornaram vazio: {failed_pages}")

    return [results[pn] for pn in page_nums]


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


ALLOWED_EXTENSIONS = {".pdf", ".sql"}


def _process_file_background(file_hash: str, filename: str, temp_path: str):
    """Processa o arquivo em thread separada, sem bloquear o event loop do FastAPI."""
    try:
        t0 = time.time()
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            pdf_processing_status[file_hash]["progress"] = "Extraindo texto do PDF..."

            loader = PyPDFLoader(temp_path)
            raw_documents = loader.load()
            total_pages = len(raw_documents)
            pdf_processing_status[file_hash]["total_pages"] = total_pages

            ocr_page_nums = [
                i for i, doc in enumerate(raw_documents)
                if len(doc.page_content.strip()) < 50
            ]
            pdf_processing_status[file_hash]["ocr_pages"] = len(ocr_page_nums)

            if ocr_page_nums:
                pdf_processing_status[file_hash]["progress"] = f"OCR: 0/{len(ocr_page_nums)} paginas..."
                logger.info(f"{len(ocr_page_nums)} de {total_pages} paginas sem texto detectado, iniciando OCR via Gemini...")
                ocr_texts = ocr_pages_with_gemini(temp_path, ocr_page_nums, file_hash)
                for page_num, ocr_text in zip(ocr_page_nums, ocr_texts):
                    raw_documents[page_num].page_content = ocr_text
                logger.info(f"OCR finalizado: {len(ocr_page_nums)} de {total_pages} paginas processadas via Gemini")

        elif ext == ".sql":
            pdf_processing_status[file_hash]["progress"] = "Lendo arquivo SQL..."

            with open(temp_path, "r", encoding="utf-8", errors="replace") as f:
                sql_content = f.read()

            raw_documents = [
                Document(
                    page_content=sql_content,
                    metadata={"source": filename, "type": "sql"},
                )
            ]
            total_pages = 1
            ocr_page_nums = []
            pdf_processing_status[file_hash]["total_pages"] = 1
            pdf_processing_status[file_hash]["ocr_pages"] = 0
            logger.info(f"SQL '{filename}': {len(sql_content)} chars lidos")

        os.remove(temp_path)

        pdf_processing_status[file_hash]["progress"] = "Indexando chunks..."

        chunks = text_splitter.split_documents(raw_documents)

        CHROMA_BATCH_SIZE = 5000
        for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch = chunks[i:i + CHROMA_BATCH_SIZE]
            vectorstore.add_documents(batch)
            pdf_processing_status[file_hash]["progress"] = f"Indexando chunks {i + len(batch)}/{len(chunks)}..."
            logger.info(f"Indexado lote {i + len(batch)}/{len(chunks)} chunks")

        documents_store.extend(chunks)

        elapsed = time.time() - t0
        logger.info(
            f"Upload '{filename}': {total_pages} paginas, "
            f"{len(chunks)} chunks, {elapsed:.2f}s"
        )

        # Registrar no historico persistente
        registry = _load_pdf_registry()
        registry[file_hash] = {
            "filename": filename,
            "pages": total_pages,
            "ocr_pages": len(ocr_page_nums),
            "chunks": len(chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(elapsed, 2),
        }
        _save_pdf_registry(registry)

        pdf_processing_status[file_hash] = {
            "status": "done",
            "filename": filename,
            "total_pages": total_pages,
            "ocr_pages": len(ocr_page_nums),
            "chunks": len(chunks),
            "elapsed": round(elapsed, 2),
            "progress": "Concluido",
        }
    except Exception as e:
        logger.error(f"Erro processando '{filename}': {e}", exc_info=True)
        pdf_processing_status[file_hash] = {
            "status": "error",
            "filename": filename,
            "progress": f"Erro: {e}",
        }
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/uploadFile")
async def upload_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de arquivo nao suportado. Aceitos: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    contents = await file.read()
    file_hash = hashlib.sha256(contents).hexdigest()[:16]

    # Ja foi processado antes?
    registry = _load_pdf_registry()
    if file_hash in registry:
        info = registry[file_hash]
        logger.info(f"Arquivo '{file.filename}' ja processado anteriormente (hash={file_hash})")
        return {
            "message": f"Arquivo ja processado anteriormente ({info['chunks']} chunks)",
            "chunks": info["chunks"],
            "status": "done",
            "file_hash": file_hash,
        }

    # Esta sendo processado agora?
    if file_hash in pdf_processing_status and pdf_processing_status[file_hash]["status"] == "processing":
        return {
            "message": "Arquivo ja esta sendo processado",
            "status": "processing",
            "file_hash": file_hash,
        }

    # Salvar e iniciar processamento em background
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    pdf_processing_status[file_hash] = {
        "status": "processing",
        "filename": file.filename,
        "progress": "Iniciando...",
        "total_pages": 0,
        "ocr_pages": 0,
        "chunks": 0,
    }

    threading.Thread(
        target=_process_file_background,
        args=(file_hash, file.filename, temp_path),
        daemon=True,
    ).start()

    return {
        "message": "Processamento iniciado",
        "status": "processing",
        "file_hash": file_hash,
    }


@app.get("/statusPDF/{file_hash}")
async def status_pdf(file_hash: str):
    if file_hash in pdf_processing_status:
        return pdf_processing_status[file_hash]
    registry = _load_pdf_registry()
    if file_hash in registry:
        info = registry[file_hash]
        return {"status": "done", **info}
    raise HTTPException(status_code=404, detail="PDF nao encontrado")


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
