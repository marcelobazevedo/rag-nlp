from typing import List, Dict
from urllib.parse import quote

import streamlit as st

from rag.graph.rag_graph import run_streaming_rag


st.set_page_config(
    page_title="NLP-RAG",
)
st.title("ChatBOT RAG")
st.write(
    "Faça uma pergunta em linguagem natural sobre a Lei 14.133, processos licitatorios, pareceres e notas juridicas."
)


def _source_label(source: Dict) -> str:
    titulo = (source.get("titulo") or "").strip()
    numero = (source.get("numero_documento") or "").strip()
    tipo = (source.get("tipo") or "").strip()

    if tipo == "lei" and numero:
        return f"Lei n.º {numero}"
    if titulo and numero:
        return f"{titulo} n.º {numero}"
    if titulo:
        return titulo
    if numero:
        return numero
    return source.get("pdf_name") or "Documento"


def _render_sources(sources: List[Dict]) -> None:
    unique_sources = []
    seen_pdf_names = set()
    for source in sources:
        pdf_name = source.get("pdf_name")
        if not pdf_name or pdf_name in seen_pdf_names:
            continue
        seen_pdf_names.add(pdf_name)
        unique_sources.append(source)

    if not unique_sources:
        return

    st.markdown("**Fontes consultadas:**")
    lines = []
    for source in unique_sources:
        pdf_name = source["pdf_name"]
        pdf_href = f"/app/static/{quote(pdf_name)}"
        label = _source_label(source)
        lines.append(f"- {label} ([{pdf_name}]({pdf_href})).")

    st.markdown("\n".join(lines))

with st.sidebar:
    st.header("Configuração de Recuperação")
    retrieval_mode = st.selectbox(
        "Modo",
        options=["dense", "sparse", "hybrid"],
        index=2,
        help="dense: vetorial, sparse: BM25, hybrid: fusão RRF",
    )
    top_k = st.selectbox("Top-k", options=[3, 5, 10], index=1)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: O que a Lei 14.133 trata sobre fase preparatoria?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        details_expander = st.expander("🔎 **Detalhes da Busca (Chunks Recuperados)**")
        query_placeholder = details_expander.empty()
        filter_placeholder = details_expander.empty()
        chunks_placeholder = details_expander.empty()
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        full_answer = ""
        sources: List[Dict] = []

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
                payload = event["data"]
                sources = payload.get("sources", [])
                with sources_placeholder.container():
                    _render_sources(sources)

        st.session_state.messages.append({"role": "assistant", "content": full_answer})
