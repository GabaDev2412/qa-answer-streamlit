import streamlit as st
import requests
import threading
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

uploaded_file = st.sidebar.file_uploader("Escolha um arquivo PDF", type="pdf")

if uploaded_file is not None and uploaded_file.name != st.session_state.get("last_uploaded"):
    with st.sidebar, st.spinner("Processando PDF..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            response = requests.post(
                f"{API_URL}/postPDF", files=files, timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                chunks = data.get("chunks", "?")
                st.session_state.last_uploaded = uploaded_file.name
                st.sidebar.success(f"Arquivo carregado! ({chunks} chunks)")
            else:
                st.sidebar.error(f"Erro: {response.json().get('detail')}")
        except requests.ConnectionError:
            st.sidebar.error("API indisponivel. Aguarde a inicializacao.")
        except requests.Timeout:
            st.sidebar.error("Timeout ao carregar o PDF. Tente novamente.")

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
