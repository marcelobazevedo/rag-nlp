import streamlit as st

from rag.graph.rag_graph import run_streaming_rag
from typing import List, Dict

# Configuração da Página e Título
st.set_page_config(
    page_title="NLP-RAG",
)
st.title("ChatBOT RAG")
st.write(
    "Faça uma pergunta em linguagem natural sobre a Lei 14.133, processos licitatorios, pareceres e notas juridicas."
)

with st.sidebar:
    st.header("Configuração de Recuperação")
    retrieval_mode = st.selectbox(
        "Modo",
        options=["dense", "sparse", "hybrid"],
        index=2,
        help="dense: vetorial, sparse: BM25, hybrid: fusão RRF",
    )
    top_k = st.selectbox("Top-k", options=[3, 5, 10], index=1)

# Gerenciamento do Histórico de Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura da Pergunta e Execução do Fluxo
if prompt := st.chat_input("Ex: O que a Lei 14.133 trata sobre fase preparatoria?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicia a UI de resposta do assistente
    with st.chat_message("assistant"):
        details_expander = st.expander("🔎 **Detalhes da Busca (Chunks Recuperados)**")
        query_placeholder = details_expander.empty()
        filter_placeholder = details_expander.empty()
        chunks_placeholder = details_expander.empty()
        answer_placeholder = st.empty()

        full_answer = ""
        recovered_chunks: List[Dict] = []

        for event in run_streaming_rag(prompt, retrieval_mode=retrieval_mode, top_k=top_k):
            if event["type"] == "details":
                data = event["data"]
                query_placeholder.markdown(f"**Busca Semântica:** `{data['query']}`")
                filter_placeholder.markdown(f"**Query Expansion:** `{data['filter']}`")
                chunks_placeholder.markdown(
                    f"**Modo:** `{data['mode']}` | **Top-k:** `{data['top_k']}`"
                )
            elif event["type"] == "token":
                token = event["data"]
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")
            elif event["type"] == "sources":
                answer_placeholder.markdown(full_answer)
                sources = event["data"]
                recovered_chunks = sources
                if sources:
                    with st.expander("📚 **Chunks/Fontes Recuperadas**", expanded=True):
                        for source in sources:
                            score = source.get("score")
                            score_text = f"{float(score):.4f}" if score is not None else "-"
                            st.markdown(
                                f"- **Título:** `{source.get('titulo')}`\n"
                                f"- **Arquivo:** `{source.get('pdf_name')}`\n"
                                f"- **Fonte:** `{source.get('fonte')}`\n"
                                f"- **Numero do Documento:** `{source.get('numero_documento')}`\n"
                                f"- **Tipo do Chunk:** `{source.get('chunk_type')}`\n"
                                f"- **Score:** `{score_text}`\n"
                                f"- **Citação:** `[{source.get('doc_id')}#{source.get('chunk_id')}]`"
                            )

        # Adiciona a resposta completa ao histórico de chat
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
