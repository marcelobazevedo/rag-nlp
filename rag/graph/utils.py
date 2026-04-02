from dotenv import load_dotenv

load_dotenv()


def _format_docs(docs: list) -> str:
    parts = []
    for d in docs:
        md = d.get("metadata", {}) if isinstance(d, dict) else (d.metadata or {})
        text = d.get("text", "") if isinstance(d, dict) else getattr(d, "page_content", "")
        doc_id = md.get("doc_id", d.get("doc_id") if isinstance(d, dict) else "?")
        chunk_id = md.get("chunk_id", d.get("chunk_id") if isinstance(d, dict) else "?")
        score = d.get("score") if isinstance(d, dict) else None
        head = (
            f"[citacao={doc_id}#{chunk_id}]"
            f"\n[{md.get('pdf_name', '?')} | numero: {md.get('numero_documento', 'nao informado')} | tipo: {md.get('tipo', 'nao informado')} | secao: {md.get('chunk_type', 'chunk')}]"
            f"\nassunto: {md.get('assunto', 'nao informado')}"
            f"\ndata: {md.get('data', 'nao informado')}"
        )
        if score is not None:
            head += f"\nscore: {float(score):.4f}"
        parts.append(f"{head}\n\n{text}")
    return "\n\n---\n\n".join(parts)
