import re
import unicodedata
from typing import Any, Dict, List

from rag.graph.model_provider import generate_text

_STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "em", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por",
    "qual", "quais", "que", "sobre", "segundo", "sua", "suas", "seu", "seus",
    "um", "uma", "uns", "umas", "sao", "são", "ser", "trata", "estabelece",
    "previstos", "previstas", "previsto", "prevista", "lei", "artigo",
}

_DOC_PATTERN = re.compile(
    r"\b(?:lei\s*\d[\d\./]*|parecer\s*\d[\d\./]*|nota(?:\s+juridica)?\s*\d[\d\./]*)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize(text)
    return re.findall(r"[a-z0-9]+", normalized)


def _keywords(question: str) -> List[str]:
    tokens = _tokenize(question)
    return [token for token in tokens if len(token) > 2 and token not in _STOPWORDS]


def _doc_match_signal(question: str, docs: List[Dict[str, Any]]) -> bool:
    mentioned_docs = _DOC_PATTERN.findall(_normalize(question))
    if not mentioned_docs:
        return False

    searchable = []
    for doc in docs:
        md = doc.get("metadata", {})
        searchable.append(
            " ".join(
                [
                    str(md.get("doc_id", "")),
                    str(md.get("numero_documento", "")),
                    str(md.get("titulo", "")),
                    str(md.get("pdf_name", "")),
                ]
            ).lower()
        )

    return any(any(ref in blob for blob in searchable) for ref in mentioned_docs)


def _judge_with_llm(question: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    context_parts = []
    for index, doc in enumerate(docs[:3], start=1):
        md = doc.get("metadata", {})
        snippet = (doc.get("text", "") or "")[:700]
        context_parts.append(
            f"Documento {index}\n"
            f"- doc_id: {md.get('doc_id', '')}\n"
            f"- titulo: {md.get('titulo', '')}\n"
            f"- numero_documento: {md.get('numero_documento', '')}\n"
            f"- texto: {snippet}\n"
        )

    prompt = (
        "Voce eh um verificador estrito de evidencia em RAG.\n"
        "Responda SOMENTE com JSON no formato "
        '{"sufficient": true|false, "reason": "curta"}.\n'
        "Marque sufficient=true apenas se o contexto recuperado for realmente suficiente para responder "
        "a pergunta sem usar conhecimento externo. Se o contexto for irrelevante, tangencial ou insuficiente, "
        "marque sufficient=false.\n\n"
        f"Pergunta: {question}\n\n"
        f"Contexto:\n{''.join(context_parts)}"
    )
    try:
        raw = generate_text(prompt=prompt, timeout=20, max_tokens=80)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return {"sufficient": None, "reason": "llm_parse_failed"}
        payload = raw[start : end + 1]
        import json

        parsed = json.loads(payload)
        return {
            "sufficient": bool(parsed.get("sufficient")),
            "reason": str(parsed.get("reason", "")).strip(),
        }
    except Exception:
        return {"sufficient": None, "reason": "llm_unavailable"}


def assess_evidence(question: str, docs: List[Dict[str, Any]], retrieval_mode: str) -> Dict[str, Any]:
    if not docs:
        return {
            "evidence_sufficient": False,
            "refusal_reason": "sem_chunks",
            "evidence_summary": {
                "docs_count": 0,
                "question_keywords": [],
                "matched_keywords": [],
                "keyword_coverage": 0.0,
                "doc_match_signal": False,
                "retrieval_mode": retrieval_mode,
            },
        }

    question_keywords = _keywords(question)
    searchable_parts = []
    scores = []
    for doc in docs:
        md = doc.get("metadata", {})
        searchable_parts.append(
            " ".join(
                [
                    doc.get("text", ""),
                    str(md.get("titulo", "")),
                    str(md.get("numero_documento", "")),
                    str(md.get("pdf_name", "")),
                    str(md.get("assunto", "")),
                    str(md.get("tipo", "")),
                ]
            )
        )
        score = doc.get("score")
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass

    searchable_blob = _normalize(" ".join(searchable_parts))
    matched_keywords = sorted({kw for kw in question_keywords if kw in searchable_blob})
    keyword_coverage = (
        len(matched_keywords) / len(question_keywords) if question_keywords else 1.0
    )
    doc_match_signal = _doc_match_signal(question, docs)
    max_score = max(scores) if scores else None

    evidence_sufficient = False
    refusal_reason = "evidencia_insuficiente"

    if retrieval_mode == "named_doc" and docs:
        evidence_sufficient = True
        refusal_reason = ""
    elif doc_match_signal:
        evidence_sufficient = True
        refusal_reason = ""
    elif len(matched_keywords) >= 3:
        evidence_sufficient = True
        refusal_reason = ""
    elif len(docs) >= 2 and len(matched_keywords) >= 2 and keyword_coverage >= 0.25:
        evidence_sufficient = True
        refusal_reason = ""
    elif len(docs) >= 1 and len(matched_keywords) >= 2 and keyword_coverage >= 0.4:
        evidence_sufficient = True
        refusal_reason = ""
    elif retrieval_mode == "sparse" and max_score is not None and max_score <= 0:
        evidence_sufficient = False
        refusal_reason = "scores_baixos"

    llm_judge = {"sufficient": None, "reason": ""}
    if evidence_sufficient:
        llm_judge = _judge_with_llm(question, docs)
        if llm_judge["sufficient"] is False:
            evidence_sufficient = False
            refusal_reason = "evidencia_insuficiente"

    return {
        "evidence_sufficient": evidence_sufficient,
        "refusal_reason": refusal_reason,
        "evidence_summary": {
            "docs_count": len(docs),
            "question_keywords": question_keywords,
            "matched_keywords": matched_keywords,
            "keyword_coverage": round(keyword_coverage, 4),
            "doc_match_signal": doc_match_signal,
            "max_score": max_score,
            "retrieval_mode": retrieval_mode,
            "llm_judge_sufficient": llm_judge["sufficient"],
            "llm_judge_reason": llm_judge["reason"],
        },
    }
