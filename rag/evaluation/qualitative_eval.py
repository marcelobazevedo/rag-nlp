import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.graph.rag_graph import run_streaming_rag


RUBRIC_FIELDS = [
    "groundedness",
    "correction",
    "citations",
    "hallucination",
    "refusal",
]


def _load_items(dataset_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise ValueError("Dataset sem itens para avaliacao qualitativa.")
    return items


def _run_question(
    question: str, retrieval_mode: str, top_k: int
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    answer = ""
    sources: List[Dict[str, Any]] = []
    guard_data: Dict[str, Any] = {}

    for event in run_streaming_rag(question, retrieval_mode=retrieval_mode, top_k=top_k):
        event_type = event.get("type")
        if event_type == "token":
            answer += event.get("data", "")
        elif event_type == "guard":
            guard_data = event.get("data", {}) or {}
        elif event_type == "sources":
            payload = event.get("data", {}) or {}
            sources = payload.get("sources", []) or []

    return answer.strip(), sources, guard_data


def _build_row(
    index: int,
    item: Dict[str, Any],
    answer: str,
    sources: List[Dict[str, Any]],
    guard_data: Dict[str, Any],
    retrieval_mode: str,
    top_k: int,
) -> Dict[str, Any]:
    source_chunk_ids = [s.get("chunk_id") for s in sources if s.get("chunk_id")]
    source_doc_ids = [s.get("doc_id") for s in sources if s.get("doc_id")]
    source_citations = [
        f"[{s.get('doc_id')}#{s.get('chunk_id')}]" for s in sources if s.get("doc_id") and s.get("chunk_id")
    ]

    return {
        "item_id": index,
        "question": item.get("question", ""),
        "expected_behavior": item.get("expected_behavior", "answer"),
        "relevant": item.get("relevant", []),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "answer": answer,
        "source_count": len(sources),
        "source_doc_ids": source_doc_ids,
        "source_chunk_ids": source_chunk_ids,
        "source_citations": source_citations,
        "sources": sources,
        "evidence_sufficient": guard_data.get("evidence_sufficient", True),
        "refusal_reason": guard_data.get("refusal_reason", ""),
        "evidence_summary": guard_data.get("evidence_summary", {}),
        "rubric": {field: None for field in RUBRIC_FIELDS},
        "review_notes": "",
    }


def _write_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_id",
        "question",
        "expected_behavior",
        "retrieval_mode",
        "top_k",
        "relevant",
        "answer",
        "source_count",
        "source_doc_ids",
        "source_chunk_ids",
        "source_citations",
        "evidence_sufficient",
        "refusal_reason",
        "evidence_summary",
        "groundedness",
        "correction",
        "citations",
        "hallucination",
        "refusal",
        "review_notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "question": row["question"],
                    "expected_behavior": row["expected_behavior"],
                    "retrieval_mode": row["retrieval_mode"],
                    "top_k": row["top_k"],
                    "relevant": json.dumps(row["relevant"], ensure_ascii=False),
                    "answer": row["answer"],
                    "source_count": row["source_count"],
                    "source_doc_ids": json.dumps(row["source_doc_ids"], ensure_ascii=False),
                    "source_chunk_ids": json.dumps(row["source_chunk_ids"], ensure_ascii=False),
                    "source_citations": json.dumps(row["source_citations"], ensure_ascii=False),
                    "evidence_sufficient": row["evidence_sufficient"],
                    "refusal_reason": row["refusal_reason"],
                    "evidence_summary": json.dumps(row["evidence_summary"], ensure_ascii=False),
                    "groundedness": "",
                    "correction": "",
                    "citations": "",
                    "hallucination": "",
                    "refusal": "",
                    "review_notes": row["review_notes"],
                }
            )


def evaluate(
    dataset_path: Path,
    retrieval_mode: str,
    top_k: int,
    limit: int | None,
    jsonl_output: Path,
    csv_output: Path,
) -> None:
    items = _load_items(dataset_path)
    if limit is not None:
        items = items[:limit]

    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        question = (item.get("question") or "").strip()
        if not question:
            continue
        answer, sources, guard_data = _run_question(question, retrieval_mode=retrieval_mode, top_k=top_k)
        rows.append(_build_row(index, item, answer, sources, guard_data, retrieval_mode, top_k))

    _write_jsonl(rows, jsonl_output)
    _write_csv(rows, csv_output)

    print("\nAvaliacao qualitativa gerada")
    print(f"Itens processados: {len(rows)}")
    print(f"Modo de recuperacao: {retrieval_mode}")
    print(f"Top-k: {top_k}")
    print(f"JSONL: {jsonl_output}")
    print(f"CSV: {csv_output}")
    print("Rubrica esperada por item: groundedness, correction, citations, hallucination, refusal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera arquivo-base para avaliacao qualitativa manual do chatbot.")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Arquivo JSON com campo 'items'.",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["dense", "sparse", "hybrid"],
        help="Modo de recuperacao usado na avaliacao qualitativa.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        choices=[3, 5, 10],
        help="Numero de chunks recuperados por pergunta.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Numero maximo de perguntas a processar. Use 15 para atender o minimo do enunciado.",
    )
    parser.add_argument(
        "--jsonl-output",
        default="data/qualitative_eval_results.jsonl",
        help="Arquivo de saida JSONL com respostas, fontes e campos da rubrica.",
    )
    parser.add_argument(
        "--csv-output",
        default="data/qualitative_eval_results.csv",
        help="Arquivo de saida CSV para preenchimento manual da rubrica.",
    )
    args = parser.parse_args()

    evaluate(
        dataset_path=Path(args.dataset),
        retrieval_mode=args.mode,
        top_k=args.top_k,
        limit=args.limit,
        jsonl_output=Path(args.jsonl_output),
        csv_output=Path(args.csv_output),
    )
