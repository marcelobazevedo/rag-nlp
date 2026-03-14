import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Set

from rag.retrieval.retriever import HybridRetriever


def _to_id_set(raw_relevant: List) -> Set[str]:
    """
    Aceita strings (chunk_id completo como "doc::chunk_NNN") ou dicts com
    {"chunk_id": "..."} ou {"doc_id": "...", "chunk_id": "..."}.
    Usa chunk_id como chave única canônica (possui restrição UNIQUE no banco).
    """
    result: Set[str] = set()
    for item in raw_relevant:
        if isinstance(item, str):
            result.add(item)
            continue
        chunk_id = item.get("chunk_id")
        if chunk_id:
            result.add(chunk_id)
    return result


def _from_results(results: List[Dict]) -> Set[str]:
    """Usa o chunk_id de nível raiz (chave completa no banco) como identificador único."""
    ids = set()
    for row in results:
        # Prefere o chunk_id do nível da linha (forma completa "doc::chunk_NNN" do banco)
        # ao chunk_id do metadata, que pode estar na forma curta ("chunk_NNN").
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            md = row.get("metadata", {})
            chunk_id = md.get("chunk_id")
        if chunk_id:
            ids.add(chunk_id)
    return ids


def recall_at_k(relevant: Set[str], retrieved: Set[str]) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def evaluate(dataset_path: Path):
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise ValueError("Dataset sem itens para avaliação.")

    retriever = HybridRetriever()
    modes = ["dense", "sparse", "hybrid"]
    ks = [3, 5, 10]

    metrics: Dict[str, Dict[int, List[float]]] = {
        mode: {k: [] for k in ks} for mode in modes
    }

    for row in items:
        question = row.get("question", "").strip()
        if not question:
            continue
        relevant = _to_id_set(row.get("relevant", []))

        for mode in modes:
            for k in ks:
                results = retriever.get_mode(mode, question, k=k)
                retrieved = _from_results(results)
                metrics[mode][k].append(recall_at_k(relevant, retrieved))

    retriever.close()

    print("\nRecall@k médio por modo")
    print("mode\tk=3\tk=5\tk=10")
    for mode in modes:
        values = [mean(metrics[mode][k]) if metrics[mode][k] else 0.0 for k in ks]
        print(f"{mode}\t{values[0]:.4f}\t{values[1]:.4f}\t{values[2]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação Recall@k para dense/sparse/hybrid")
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Arquivo JSON com campo 'items' e rótulos relevantes por pergunta.",
    )
    args = parser.parse_args()

    evaluate(Path(args.dataset))
