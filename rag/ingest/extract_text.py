import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from rag.graph.model_provider import embed_text
from rag.ingest.pgvector_store import PgVectorStore

load_dotenv()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def _read_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    page_texts = []
    for page in reader.pages:
        page_texts.append(page.extract_text() or "")
    return "\n".join(page_texts).strip()


def _normalize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")


_NATURAL_BREAK = re.compile(
    r'\b(?:Art\.?\s+\d+[\.,º°]|CAPÍTULO\s+[IVXLCDM]+|SEÇÃO\s+[IVXLCDM]+)',
    re.IGNORECASE,
)


def _split_with_overlap(text: str, size: int, overlap: int) -> List[str]:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]

    # Posições de fronteiras naturais (Art., CAPÍTULO, SEÇÃO)
    break_positions = [m.start() for m in _NATURAL_BREAK.finditer(clean)]

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = start + size
        if end >= len(clean):
            chunks.append(clean[start:].strip())
            break

        # Procura a última fronteira natural no intervalo [start + size//2, end]
        # Garante que o chunk tenha pelo menos metade do tamanho alvo antes de quebrar
        min_end = start + size // 2
        best_break = None
        for bp in reversed(break_positions):
            if min_end <= bp <= end:
                best_break = bp
                break

        if best_break is not None:
            # Quebra limpa: chunk termina antes do novo artigo/capítulo
            chunks.append(clean[start:best_break].strip())
            start = best_break  # próximo chunk começa no artigo — sem overlap redundante
        else:
            # Sem fronteira natural: usa overlap padrão para não perder contexto
            chunks.append(clean[start:end].strip())
            start = end - overlap

    return [c for c in chunks if c]


def _infer_tipo_documento(file_stem: str) -> str:
    stem = file_stem.lower()
    if "lei" in stem or "14133" in stem or "14.133" in stem:
        return "lei"
    if "parecer" in stem:
        return "parecer"
    if "nota" in stem:
        return "nota_juridica"
    return "outro"


def _infer_numero_documento(file_stem: str, text: str, tipo: str) -> str:
    stem = file_stem.lower()

    if tipo == "lei":
        lei_match = re.search(r"14\.?133(?:/2021)?", stem)
        if lei_match:
            return "14.133/2021"
        lei_text = re.search(r"lei\s*n?[oº°]?\s*(14\.?133(?:/2021)?)", text, flags=re.IGNORECASE)
        if lei_text:
            value = lei_text.group(1).replace(".", ".")
            return "14.133/2021" if "2021" in value or value.startswith("14") else value

    number_match = re.search(r"(\d{1,4})[_.\-/](\d{2,4})", file_stem)
    if number_match:
        return f"{number_match.group(1)}/{number_match.group(2)}"

    simple_match = re.search(r"(\d{1,4})", file_stem)
    if simple_match:
        return simple_match.group(1)

    return ""


def _infer_data_documento(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    if match:
        return match.group(1)
    return None


def _infer_orgao_emissor(text: str) -> Optional[str]:
    patterns = [
        r"Tribunal\s+de\s+Contas[^\n,.;]*",
        r"Procuradoria[^\n,.;]*",
        r"Assessoria\s+Juridica[^\n,.;]*",
        r"Controladoria[^\n,.;]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _infer_assunto(text: str) -> Optional[str]:
    # Usa o primeiro trecho significativo como resumo de assunto.
    snippet = re.sub(r"\s+", " ", text).strip()[:220]
    return snippet or None


def _extract_sections(text: str) -> List[Dict[str, str]]:
    upper_text = text.upper()
    markers = [
        ("objeto", ["OBJETO", "EMENTA", "INTRODUCAO", "INTRODUÇÃO"]),
        ("fundamentacao", ["FUNDAMENTACAO", "FUNDAMENTAÇÃO", "BASE LEGAL", "ANALISE", "ANÁLISE"]),
        ("conclusao", ["CONCLUSAO", "CONCLUSÃO", "PARECER", "ENCAMINHAMENTO"]),
    ]

    positions: List[tuple] = []
    for section_name, candidates in markers:
        found_idx = -1
        found_text = ""
        for candidate in candidates:
            idx = upper_text.find(candidate)
            if idx != -1 and (found_idx == -1 or idx < found_idx):
                found_idx = idx
                found_text = candidate
        if found_idx != -1:
            positions.append((found_idx, section_name, found_text))

    if not positions:
        return [{"tipo": "objeto", "texto": text.strip()}]

    positions.sort(key=lambda x: x[0])
    sections: List[Dict[str, str]] = []
    for i, (start_idx, section_name, marker) in enumerate(positions):
        end_idx = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[start_idx:end_idx].strip()
        # Remove o marcador do inicio quando possivel.
        section_text = re.sub(rf"^{re.escape(marker)}\s*[:\-]?\s*", "", section_text, flags=re.IGNORECASE)
        if section_text:
            sections.append({"tipo": section_name, "texto": section_text})

    return sections if sections else [{"tipo": "objeto", "texto": text.strip()}]


def process_pdf_file(file_path: str) -> List[Dict[str, Any]]:
    """Extrai texto, infere metadados minimos e cria chunks deterministicamente."""
    pdf_name = os.path.basename(file_path)
    file_stem = Path(file_path).stem
    default_doc_id = _normalize_id(file_stem) or "doc_desconhecido"

    text_content = _read_pdf_text(str(file_path))
    if not text_content:
        print(f"[ingest] Sem texto extraivel: {pdf_name}")
        return []

    tipo_documento = _infer_tipo_documento(file_stem)
    numero_documento = _infer_numero_documento(file_stem, text_content, tipo_documento)
    doc_id = _normalize_id(f"{tipo_documento}_{numero_documento}") if numero_documento else default_doc_id

    titulo = file_stem
    fonte = f"Arquivo local: {pdf_name}"
    data_documento = _infer_data_documento(text_content)
    assunto = _infer_assunto(text_content)
    orgao_emissor = _infer_orgao_emissor(text_content)

    sections = _extract_sections(text_content)
    processed: List[Dict[str, Any]] = []
    chunk_counter = 0

    for section in sections:
        section_type = section["tipo"]
        section_text = section["texto"]
        for split_text in _split_with_overlap(section_text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunk_id = f"chunk_{chunk_counter:03d}"
            metadata = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "titulo": titulo,
                "fonte": fonte,
                "data": data_documento,
                "tipo": tipo_documento,
                "numero_documento": numero_documento,
                "assunto": assunto,
                "orgao_emissor": orgao_emissor,
                "pdf_name": pdf_name,
                "chunk_type": section_type,
                "chunk_index": chunk_counter,
            }
            # Prefixo com título garante recuperação por nome do documento (BM25 e dense).
            indexed_text = f"{titulo} [{tipo_documento}]: {split_text}"
            processed.append({"text": indexed_text, "metadata": metadata})
            chunk_counter += 1

    return processed


def main(collection: str = "documentos_licitacoes", pasta_pdfs: str = "documentos"):
    vector_store = PgVectorStore()
    vector_store.create_table()
    pdf_files = list(Path(pasta_pdfs).glob("*.pdf"))
    if not pdf_files:
        print("Nenhum PDF encontrado na pasta.")
        return

    total_chunks = 0
    processed_files = 0
    skipped_files = 0
    for pdf_file in pdf_files:
        chunks = process_pdf_file(str(pdf_file))
        if not chunks:
            skipped_files += 1
            print(f"[ingest] Ignorado (0 chunks): {pdf_file.name}")
            continue

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = []

        for text in texts:
            emb = embed_text(text)
            embeddings.append(emb)

        vector_store.add_texts(texts=texts, metadatas=metadatas, embeddings=embeddings)
        total_chunks += len(chunks)
        processed_files += 1
        print(f"[ingest] OK {pdf_file.name}: {len(chunks)} chunks")

    vector_store.close()
    print(
        f"✅ PDFs lidos: {len(pdf_files)} | Com chunks: {processed_files} | "
        f"Sem chunks: {skipped_files} | Chunks inseridos: {total_chunks}"
    )


if __name__ == "__main__":
    main()
