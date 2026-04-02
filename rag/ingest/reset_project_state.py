import argparse
import shutil
from pathlib import Path

from rag.ingest.truncate_data import truncate_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_FILES = [
    PROJECT_ROOT / "data" / "qualitative_eval_results.jsonl",
    PROJECT_ROOT / "data" / "qualitative_eval_results.csv",
    PROJECT_ROOT / "data" / "qualitative_rubric.md",
    PROJECT_ROOT / "data" / "retrieval_tradeoff_analysis.json",
    PROJECT_ROOT / "data" / "retrieval_tradeoffs.md",
]

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
SKIP_DIR_NAMES = {".git", ".venv"}


def remove_generated_files() -> list[Path]:
    removed = []
    for file_path in GENERATED_FILES:
        if file_path.exists():
            file_path.unlink()
            removed.append(file_path)
    return removed


def remove_local_caches() -> list[Path]:
    removed = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name in CACHE_DIR_NAMES:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def reset_project_state(remove_eval_artifacts: bool = True, remove_caches: bool = True) -> None:
    print("[reset] Truncando tabela 'dados'...")
    truncate_table()

    if remove_eval_artifacts:
        removed_files = remove_generated_files()
        if removed_files:
            print("[reset] Artefatos removidos:")
            for path in removed_files:
                print(f"  - {path.relative_to(PROJECT_ROOT)}")
        else:
            print("[reset] Nenhum artefato de avaliação para remover.")

    if remove_caches:
        removed_caches = remove_local_caches()
        if removed_caches:
            print("[reset] Caches removidos:")
            for path in removed_caches:
                print(f"  - {path.relative_to(PROJECT_ROOT)}")
        else:
            print("[reset] Nenhum cache local para remover.")

    print("[reset] Projeto resetado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trunca a base vetorial e remove artefatos regeneráveis do projeto."
    )
    parser.add_argument(
        "--keep-eval-artifacts",
        action="store_true",
        help="Mantém os artefatos de avaliação em data/.",
    )
    parser.add_argument(
        "--keep-caches",
        action="store_true",
        help="Mantém caches locais como __pycache__ e .pytest_cache.",
    )
    args = parser.parse_args()

    reset_project_state(
        remove_eval_artifacts=not args.keep_eval_artifacts,
        remove_caches=not args.keep_caches,
    )
