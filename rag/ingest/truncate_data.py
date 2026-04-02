import argparse

from rag.ingest.pgvector_store import PgVectorStore


def truncate_table() -> None:
    store = PgVectorStore()
    try:
        store.cur.execute("TRUNCATE TABLE dados RESTART IDENTITY;")
        store.conn.commit()
        print("Tabela 'dados' truncada com sucesso.")
        print("A sequência de IDs também foi reiniciada.")
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trunca a tabela 'dados' e reinicia a sequência de IDs.")
    parser.parse_args()
    truncate_table()
