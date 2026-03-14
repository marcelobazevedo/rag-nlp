import json
import os
import re

import psycopg2
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from typing import Dict

from rag.graph.model_provider import generate_text
from rag.retrieval.retriever import HybridRetriever, reciprocal_rank_fusion

load_dotenv()

# Padrões para detectar referência a documento específico na query
_DOC_PATTERNS = [
    (re.compile(r'\bparecer\s*n?[oº°]?\s*(\d+)\b', re.IGNORECASE), 'parecer'),
    (re.compile(r'\bnota\s+jur[íi]dica\s*n?[oº°]?\s*(\d+)\b', re.IGNORECASE), 'nota'),
    (re.compile(r'\bnota\s*n?[oº°]?\s*(\d+)\b', re.IGNORECASE), 'nota'),
]


def _detect_named_doc(question: str):
    """Retorna (tipo, numero) se a query menciona um documento específico, ou None."""
    for pattern, tipo in _DOC_PATTERNS:
        m = pattern.search(question)
        if m:
            return (tipo, m.group(1))
    return None


def _expand_query(question: str) -> str:
    """Reformula a pergunta com terminologia jurídica técnica para melhorar recall no BM25/dense."""
    prompt = (
        "Você é um assistente jurídico especialista na Lei 14.133/2021 (licitações e contratos).\n"
        "Dada a pergunta do usuário abaixo, reescreva-a usando terminologia jurídica técnica precisa "
        "e adicione sinônimos ou termos legais correlatados que ajudem a localizar os artigos relevantes "
        "na lei. Responda SOMENTE com a pergunta reformulada, sem explicações.\n\n"
        f"Pergunta: {question}\n\nPergunta reformulada:"
    )
    try:
        expanded = generate_text(prompt=prompt, timeout=15, max_tokens=100)
        return expanded if expanded else question
    except Exception:
        return question


def retrieve(
    state,
    config: RunnableConfig,
    collection_name: str = "dados",
    k: int = 5,
) -> Dict:
    """Nó de recuperação com suporte aos modos dense, sparse e hybrid."""
    print("Executando o nó de recuperação...")
    mode = state.get("retrieval_mode", "hybrid")
    top_k = int(state.get("top_k", k))
    question = state["question"]

    retriever = HybridRetriever()

    # ── Atalho: query menciona documento por nome (ex: "parecer 8") ──────────
    named = _detect_named_doc(question)
    if named:
        tipo, numero = named
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"), port=int(os.getenv("POSTGRES_PORT", 5432)),
            user=os.getenv("POSTGRES_USER"), password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB"),
        )
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, text, metadata FROM dados "
            "WHERE doc_id ~ %s ORDER BY chunk_id LIMIT %s",
            (f"^{tipo}_{numero}(_|$)", top_k),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        if rows:
            print(f"[retrieval] Atalho por nome: tipo={tipo} numero={numero} → {len(rows)} chunks")
            named_docs = []
            for chunk_id, doc_id, text, metadata in rows:
                md = metadata or {}
                if isinstance(md, str):
                    md = json.loads(md)
                named_docs.append({
                    "id": chunk_id, "chunk_id": chunk_id, "doc_id": doc_id,
                    "text": text, "metadata": md, "score": 1.0,
                })
            retriever.close()
            return {
                "docs": named_docs,
                "generated_query": question,
                "generated_filter": f"{tipo}_{numero}",
                "retrieval_mode": "named_doc",
                "top_k": top_k,
                "retrieval_modes": {"dense": named_docs, "sparse": named_docs, "hybrid": named_docs},
            }
    # ─────────────────────────────────────────────────────────────────────────

    # Expande a query com terminologia jurídica técnica para melhorar recall
    expanded = _expand_query(question)
    if expanded and expanded != question:
        print(f"[retrieval] Query expandida: {expanded[:100]!r}")

    def _fetch_all(q: str):
        return {
            "dense": retriever.dense_search(q, k=top_k),
            "sparse": retriever.bm25_search(q, k=top_k),
            "hybrid": retriever.hybrid_search(q, k=top_k),
        }

    orig = _fetch_all(question)
    if expanded and expanded != question:
        exp = _fetch_all(expanded)
        dense_docs = reciprocal_rank_fusion([orig["dense"], exp["dense"]], k=top_k)
        sparse_docs = reciprocal_rank_fusion([orig["sparse"], exp["sparse"]], k=top_k)
        hybrid_docs = reciprocal_rank_fusion([orig["hybrid"], exp["hybrid"]], k=top_k)
    else:
        dense_docs = orig["dense"]
        sparse_docs = orig["sparse"]
        hybrid_docs = orig["hybrid"]

    docs_by_mode = {
        "dense": dense_docs,
        "sparse": sparse_docs,
        "hybrid": hybrid_docs,
    }
    docs = docs_by_mode.get(mode, hybrid_docs)

    for idx, doc in enumerate(docs, start=1):
        md = doc.get("metadata", {})
        print(
            f"[retrieval][{mode}] rank={idx} "
            f"doc_id={md.get('doc_id', doc.get('doc_id'))} "
            f"chunk_id={md.get('chunk_id', doc.get('chunk_id'))} "
            f"score={doc.get('score', 0):.4f}"
        )

    retriever.close()
    print(f"Busca finalizada. Encontrados {len(docs)} chunks no modo '{mode}'.")

    filter_info = f"query expandida: {expanded[:80]}..." if (expanded and expanded != question) else "nenhum"

    return {
        "docs": docs,
        "generated_query": expanded if (expanded and expanded != question) else question,
        "generated_filter": filter_info,
        "retrieval_mode": mode,
        "top_k": top_k,
        "retrieval_modes": {
            "dense": dense_docs,
            "sparse": sparse_docs,
            "hybrid": hybrid_docs,
        },
    }
