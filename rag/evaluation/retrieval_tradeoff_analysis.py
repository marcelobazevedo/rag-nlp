import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Set

from rag.retrieval.retrieval_node import retrieve_docs


MODES = ["dense", "sparse", "hybrid"]
KS = [3, 5, 10]


def _to_id_set(raw_relevant: List[Any]) -> Set[str]:
    result: Set[str] = set()
    for item in raw_relevant:
        if isinstance(item, str):
            result.add(item)
            continue
        chunk_id = item.get("chunk_id")
        if chunk_id:
            result.add(chunk_id)
    return result


def _from_results(results: List[Dict[str, Any]]) -> Set[str]:
    ids = set()
    for row in results:
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


def _question_kind(question: str, relevant_count: int) -> str:
    q = (question or "").lower()
    if relevant_count >= 2:
        return "multi_chunk"
    if " e " in q and " lei " in q and "parecer" in q:
        return "multi_doc"
    return "direct"


def _load_answerable_items(dataset_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not items:
        raise ValueError("Dataset sem itens para analise.")

    answerable_items = []
    refuse_items = 0
    for item in items:
        question = (item.get("question") or "").strip()
        if not question:
            continue
        expected_behavior = (item.get("expected_behavior") or "").strip().lower()
        if expected_behavior == "refuse":
            refuse_items += 1
            continue
        relevant = _to_id_set(item.get("relevant", []))
        if not relevant:
            continue
        answerable_items.append(
            {
                "question": question,
                "relevant": sorted(relevant),
                "kind": _question_kind(question, len(relevant)),
            }
        )
    return answerable_items, refuse_items


def analyze(dataset_path: Path) -> Dict[str, Any]:
    items, refuse_items = _load_answerable_items(dataset_path)

    per_question: List[Dict[str, Any]] = []
    average_recall: Dict[str, Dict[str, float]] = {mode: {} for mode in MODES}
    wins: Dict[str, Dict[str, int]] = {
        "hybrid_vs_dense": {str(k): 0 for k in KS},
        "hybrid_vs_sparse": {str(k): 0 for k in KS},
        "dense_vs_hybrid": {str(k): 0 for k in KS},
        "sparse_vs_hybrid": {str(k): 0 for k in KS},
        "all_equal": {str(k): 0 for k in KS},
    }

    metrics: Dict[str, Dict[int, List[float]]] = {
        mode: {k: [] for k in KS} for mode in MODES
    }

    for item in items:
        row: Dict[str, Any] = {
            "question": item["question"],
            "kind": item["kind"],
            "relevant": item["relevant"],
            "results": {},
        }

        for mode in MODES:
            row["results"][mode] = {}
            for k in KS:
                results = retrieve_docs(
                    question=item["question"],
                    mode=mode,
                    top_k=k,
                    use_named_doc=False,
                    use_query_expansion=False,
                )["docs"]
                retrieved = sorted(_from_results(results))
                hit_set = sorted(set(item["relevant"]).intersection(retrieved))
                recall = recall_at_k(set(item["relevant"]), set(retrieved))
                metrics[mode][k].append(recall)
                row["results"][mode][str(k)] = {
                    "recall": recall,
                    "hit": bool(hit_set),
                    "matched_relevant": hit_set,
                    "retrieved": retrieved,
                }

        for k in KS:
            dense_r = row["results"]["dense"][str(k)]["recall"]
            sparse_r = row["results"]["sparse"][str(k)]["recall"]
            hybrid_r = row["results"]["hybrid"][str(k)]["recall"]

            if hybrid_r > dense_r:
                wins["hybrid_vs_dense"][str(k)] += 1
            elif dense_r > hybrid_r:
                wins["dense_vs_hybrid"][str(k)] += 1

            if hybrid_r > sparse_r:
                wins["hybrid_vs_sparse"][str(k)] += 1
            elif sparse_r > hybrid_r:
                wins["sparse_vs_hybrid"][str(k)] += 1

            if dense_r == sparse_r == hybrid_r:
                wins["all_equal"][str(k)] += 1

        per_question.append(row)

    for mode in MODES:
        for k in KS:
            average_recall[mode][str(k)] = mean(metrics[mode][k]) if metrics[mode][k] else 0.0

    return {
        "dataset": str(dataset_path),
        "answerable_items": len(items),
        "refuse_items_excluded": refuse_items,
        "evaluation_policy": {
            "use_query_expansion": False,
            "use_named_doc": False,
            "reason": "comparar dense/sparse/hybrid como retrievers puros, sem deixar query expansion ou named_doc mascararem os resultados experimentais",
        },
        "ks": KS,
        "modes": MODES,
        "average_recall": average_recall,
        "wins": wins,
        "per_question": per_question,
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Analise de trade-offs da recuperacao")
    lines.append("")
    lines.append(f"- Itens avaliados para retrieval: {report['answerable_items']}")
    lines.append(f"- Itens de recusa excluidos: {report['refuse_items_excluded']}")
    lines.append("- Politica de avaliacao: desativa query expansion e named_doc para medir dense, sparse e hybrid como retrievers puros e comparaveis.")
    lines.append("")
    lines.append("## Recall medio por modo")
    lines.append("")
    lines.append("| modo | k=3 | k=5 | k=10 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for mode in MODES:
        vals = report["average_recall"][mode]
        lines.append(f"| {mode} | {vals['3']:.4f} | {vals['5']:.4f} | {vals['10']:.4f} |")
    lines.append("")
    lines.append("## Comparacao do hybrid com os modos isolados")
    lines.append("")
    for k in KS:
        lines.append(f"### k={k}")
        lines.append("")
        lines.append(f"- `hybrid` melhor que `dense`: {report['wins']['hybrid_vs_dense'][str(k)]} perguntas")
        lines.append(f"- `dense` melhor que `hybrid`: {report['wins']['dense_vs_hybrid'][str(k)]} perguntas")
        lines.append(f"- `hybrid` melhor que `sparse`: {report['wins']['hybrid_vs_sparse'][str(k)]} perguntas")
        lines.append(f"- `sparse` melhor que `hybrid`: {report['wins']['sparse_vs_hybrid'][str(k)]} perguntas")
        lines.append(f"- todos empatados: {report['wins']['all_equal'][str(k)]} perguntas")
        lines.append("")

    lines.append("## Leitura curta dos trade-offs")
    lines.append("")
    lines.append("- `hybrid` tende a ajudar quando a pergunta mistura termos literais importantes com formulacoes mais semanticas, porque combina BM25 com embeddings.")
    lines.append("- `sparse` tende a se sair melhor quando o texto da pergunta repete termos muito especificos do corpus, como numero de parecer, expressao normativa ou redacao quase literal.")
    lines.append("- `dense` tende a ajudar mais em perguntas parafraseadas, em que o usuario nao usa exatamente os mesmos termos dos documentos.")
    lines.append("- `hybrid` pode piorar em alguns casos quando a fusao RRF traz ruido de um dos modos e rebaixa um chunk muito forte que apareceria mais acima em um retriever isolado.")
    lines.append("")
    lines.append("## Perguntas em que o hybrid perdeu para algum modo isolado")
    lines.append("")

    losses = []
    for item in report["per_question"]:
        for k in KS:
            hybrid_r = item["results"]["hybrid"][str(k)]["recall"]
            dense_r = item["results"]["dense"][str(k)]["recall"]
            sparse_r = item["results"]["sparse"][str(k)]["recall"]
            if dense_r > hybrid_r or sparse_r > hybrid_r:
                losses.append((item["question"], k, dense_r, sparse_r, hybrid_r))

    if not losses:
        lines.append("- Nao houve perdas do `hybrid` frente aos modos isolados no dataset avaliado.")
    else:
        for question, k, dense_r, sparse_r, hybrid_r in losses[:10]:
            lines.append(
                f"- `k={k}`: {question} | dense={dense_r:.2f}, sparse={sparse_r:.2f}, hybrid={hybrid_r:.2f}"
            )

    lines.append("")
    lines.append("## Perguntas em que o hybrid superou algum modo isolado")
    lines.append("")

    gains = []
    for item in report["per_question"]:
        for k in KS:
            hybrid_r = item["results"]["hybrid"][str(k)]["recall"]
            dense_r = item["results"]["dense"][str(k)]["recall"]
            sparse_r = item["results"]["sparse"][str(k)]["recall"]
            if hybrid_r > dense_r or hybrid_r > sparse_r:
                gains.append((item["question"], k, dense_r, sparse_r, hybrid_r))

    if not gains:
        lines.append("- Nao houve ganhos do `hybrid` frente aos modos isolados no dataset avaliado.")
    else:
        for question, k, dense_r, sparse_r, hybrid_r in gains[:10]:
            lines.append(
                f"- `k={k}`: {question} | dense={dense_r:.2f}, sparse={sparse_r:.2f}, hybrid={hybrid_r:.2f}"
            )

    return "\n".join(lines) + "\n"


def save_outputs(report: Dict[str, Any], json_output: Path, markdown_output: Path) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_output.write_text(_build_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera analise por pergunta e resumo de trade-offs para dense, sparse e hybrid."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Arquivo JSON com campo 'items' e chunks relevantes.",
    )
    parser.add_argument(
        "--json-output",
        default="data/retrieval_tradeoff_analysis.json",
        help="Arquivo JSON de saida com analise detalhada por pergunta.",
    )
    parser.add_argument(
        "--markdown-output",
        default="data/retrieval_tradeoffs.md",
        help="Arquivo Markdown de saida com resumo interpretativo.",
    )
    args = parser.parse_args()

    report = analyze(Path(args.dataset))
    save_outputs(report, Path(args.json_output), Path(args.markdown_output))

    print("\nAnalise de trade-offs gerada")
    print(f"JSON: {args.json_output}")
    print(f"Markdown: {args.markdown_output}")
