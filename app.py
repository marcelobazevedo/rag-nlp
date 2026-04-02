import json
from pathlib import Path
from typing import List, Dict
import unicodedata
from urllib.parse import quote

import streamlit as st

from rag.graph.rag_graph import run_streaming_rag
from rag.retrieval.retrieval_node import retrieve_docs


st.set_page_config(
    page_title="NLP-RAG",
)
st.title("ChatBOT RAG")
st.write(
    "Faça uma pergunta em linguagem natural sobre a Lei 14.133, processos licitatorios, pareceres e notas juridicas."
)

_GOLDEN_SET_PATH = Path("data/eval_dataset.json")


def _normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = " ".join(normalized.split())
    return normalized.strip().casefold()


def _load_golden_set() -> Dict[str, Dict]:
    if not _GOLDEN_SET_PATH.exists():
        return {}

    payload = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    return {
        _normalize_question(item.get("question") or ""): item
        for item in items
        if (item.get("question") or "").strip()
    }


def _to_id_set(raw_relevant: List) -> set[str]:
    result = set()
    for item in raw_relevant:
        if isinstance(item, str):
            result.add(item)
            continue
        chunk_id = item.get("chunk_id")
        if chunk_id:
            result.add(chunk_id)
    return result


def _from_results(results: List[Dict]) -> set[str]:
    ids = set()
    for row in results:
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            md = row.get("metadata", {})
            chunk_id = md.get("chunk_id")
        if chunk_id:
            ids.add(chunk_id)
    return ids


def _recall_at_k(relevant: set[str], retrieved: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def _build_dense_recall_metrics(question: str) -> Dict | None:
    golden_set = _load_golden_set()
    item = golden_set.get(_normalize_question(question))
    if not item:
        return None

    expected_behavior = (item.get("expected_behavior") or "").strip().lower()
    if expected_behavior == "refuse":
        return {
            "question_found": True,
            "expected_behavior": "refuse",
            "relevant": [],
            "metrics": {},
        }

    relevant = _to_id_set(item.get("relevant", []))
    if not relevant:
        return {
            "question_found": True,
            "expected_behavior": "",
            "relevant": [],
            "metrics": {},
        }

    modes = ["dense", "sparse", "hybrid"]
    ks = [3, 5, 10]
    metrics = {}
    for mode in modes:
        metrics[mode] = {}
        for k in ks:
            results = retrieve_docs(
                question=question,
                mode=mode,
                top_k=k,
                use_named_doc=False,
                use_query_expansion=False,
            )["docs"]
            retrieved = _from_results(results)
            found = sorted(relevant.intersection(retrieved))
            missing = sorted(relevant.difference(retrieved))
            metrics[mode][k] = {
                "recall": _recall_at_k(relevant, retrieved),
                "retrieved": sorted(retrieved),
                "found": found,
                "missing": missing,
            }

    return {
        "question_found": True,
        "expected_behavior": "",
        "relevant": sorted(relevant),
        "metrics": metrics,
    }


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


def _render_eval_metrics(question: str) -> None:
    metrics = _build_dense_recall_metrics(question)
    if not metrics:
        st.caption("Avaliação: pergunta não encontrada no golden set.")
        return

    if metrics["expected_behavior"] == "refuse":
        st.caption("Avaliação: pergunta marcada no golden set como caso de recusa.")
        return

    if not metrics["metrics"]:
        st.caption("Avaliação: pergunta encontrada no golden set, sem chunks relevantes anotados.")
        return

    st.markdown("**Avaliação do Golden Set**")
    st.markdown(f"**Chunks esperados:** `{', '.join(metrics['relevant'])}`")
    summary_lines = [
        "| Modo | Recall@3 | Recall@5 | Recall@10 |",
        "| --- | --- | --- | --- |",
    ]
    for mode in ("dense", "sparse", "hybrid"):
        summary_lines.append(
            "| "
            f"{mode} | "
            f"{metrics['metrics'][mode][3]['recall']:.4f} | "
            f"{metrics['metrics'][mode][5]['recall']:.4f} | "
            f"{metrics['metrics'][mode][10]['recall']:.4f} |"
        )
    st.markdown("\n".join(summary_lines))

    for mode in ("dense", "sparse", "hybrid"):
        mode_top10 = metrics["metrics"][mode][10]
        st.markdown(f"**Modo `{mode}` até o top-10**")
        st.markdown(
            f"Encontrados: `{', '.join(mode_top10['found']) or 'nenhum'}`"
        )
        st.markdown(
            f"Faltantes: `{', '.join(mode_top10['missing']) or 'nenhum'}`"
        )

    with st.expander("Detalhes técnicos do ranking"):
        for mode in ("dense", "sparse", "hybrid"):
            st.markdown(f"**Modo `{mode}`**")
            for k in (3, 5, 10):
                mode_metrics = metrics["metrics"][mode][k]
                st.markdown(
                    f"**Top-{k} recuperado:** `{', '.join(mode_metrics['retrieved']) or 'nenhum'}`"
                )
                st.markdown(
                    f"**Encontrados no top-{k}:** `{', '.join(mode_metrics['found']) or 'nenhum'}`"
                )
                st.markdown(
                    f"**Faltantes no top-{k}:** `{', '.join(mode_metrics['missing']) or 'nenhum'}`"
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
        eval_placeholder = st.empty()

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
                with eval_placeholder.container():
                    _render_eval_metrics(prompt)

        st.session_state.messages.append({"role": "assistant", "content": full_answer})
