import streamlit as st
import requests
import threading
import time
import uvicorn

API_URL = "http://localhost:8001"


def start_api():
    from api import app
    uvicorn.run(app, host="0.0.0.0", port=8001)


if "api_started" not in st.session_state:
    st.session_state.api_started = True
    threading.Thread(target=start_api, daemon=True).start()

st.set_page_config(layout="wide", page_title="PDF Q&A Chatbot")

# --- Sidebar ---
st.sidebar.title("PDF Q&A Chatbot")
st.sidebar.caption("Gemini 2.5 Flash | Hybrid RAG (BM25 + Semantic)")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("Escolha um arquivo", type=["pdf", "sql"])

if uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded"):
    try:
        mime = "application/pdf" if uploaded_file.name.endswith(".pdf") else "text/plain"
        files = {"file": (uploaded_file.name, uploaded_file, mime)}
        response = requests.post(f"{API_URL}/uploadFile", files=files, timeout=30)

        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            file_hash = data.get("file_hash")

            if status == "done":
                st.session_state.last_uploaded = uploaded_file.name
                st.sidebar.success(f"Arquivo ja processado! ({data.get('chunks', '?')} chunks)")

            elif status == "processing" and file_hash:
                st.session_state.last_uploaded = uploaded_file.name
                st.session_state["processing_hash"] = file_hash
                st.rerun()
        else:
            st.sidebar.error(f"Erro: {response.json().get('detail')}")
    except requests.ConnectionError:
        st.sidebar.error("API indisponivel. Aguarde a inicializacao.")
    except requests.Timeout:
        st.sidebar.error("Timeout ao enviar o arquivo. Tente novamente.")

# Polling de progresso do PDF em processamento
if "processing_hash" in st.session_state:
    file_hash = st.session_state["processing_hash"]
    try:
        resp = requests.get(f"{API_URL}/statusPDF/{file_hash}", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            status = info.get("status")

            if status == "done":
                del st.session_state["processing_hash"]
                chunks = info.get("chunks", "?")
                ocr_pages = info.get("ocr_pages", 0)
                msg = f"Arquivo processado! ({chunks} chunks)"
                if ocr_pages:
                    msg += f" | OCR em {ocr_pages} paginas"
                st.sidebar.success(msg)
            elif status == "error":
                del st.session_state["processing_hash"]
                st.sidebar.error(f"Erro ao processar arquivo: {info.get('progress')}")
            else:
                progress = info.get("progress", "Processando...")
                st.sidebar.info(f"⏳ {progress}")
                time.sleep(3)
                st.rerun()
    except (requests.ConnectionError, requests.Timeout):
        st.sidebar.warning("Aguardando API...")
        time.sleep(3)
        st.rerun()

st.sidebar.divider()

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF and ask me a question."}
    ]
    st.rerun()

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF and ask me a question."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Faca uma pergunta sobre o PDF:"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando resposta..."):
            try:
                response = requests.post(
                    f"{API_URL}/askQuestion",
                    json={"question": user_input},
                    timeout=120,
                )
                if response.status_code == 200:
                    answer = response.json().get("answer")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.markdown(answer)
                else:
                    detail = response.json().get("detail", "Erro desconhecido")
                    st.error(detail)
            except requests.ConnectionError:
                st.error("API indisponivel. Aguarde a inicializacao e tente novamente.")
            except requests.Timeout:
                st.error("Timeout ao buscar resposta. Tente novamente.")
