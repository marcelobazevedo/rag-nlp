import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values, Json

class PgVectorStore:
    def __init__(self):
        load_dotenv()
        self.vector_dim = int(os.getenv("PGVECTOR_DIM", "768"))
        self.conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB")
        )
        self.cur = self.conn.cursor()

    def create_table(self):
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS dados (
                id SERIAL PRIMARY KEY,
                chunk_id TEXT UNIQUE,
                doc_id TEXT,
                titulo TEXT,
                fonte TEXT,
                data TEXT,
                tipo TEXT,
                text TEXT,
                metadata JSONB,
                embedding VECTOR({self.vector_dim})
            );
        """)
        self.conn.commit()

    def add_texts(self, texts, metadatas, embeddings):
        import json
        data = []
        for text, meta, emb in zip(texts, metadatas, embeddings):
            if emb is None or len(emb) != self.vector_dim:
                got = 0 if emb is None else len(emb)
                raise RuntimeError(
                    f"Dimensao de embedding invalida: esperado {self.vector_dim}, recebido {got}. "
                    "Ajuste PGVECTOR_DIM/.env e o modelo de embedding para a mesma dimensao antes da ingestao."
                )
            # Garante que meta seja dict
            if isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                except Exception:
                    meta_dict = {}
            else:
                meta_dict = meta
            # Extrai campos principais do metadata
            doc_id = meta_dict.get("doc_id") or meta_dict.get("pdf_name") or "doc_desconhecido"
            meta_chunk_id = meta_dict.get("chunk_id") or f"chunk_{meta_dict.get('chunk_index', '')}"
            # Chave unica de armazenamento para evitar colisao de chunk_id entre documentos.
            chunk_id = f"{doc_id}::{meta_chunk_id}"
            titulo = meta_dict.get("titulo") or meta_dict.get("pdf_name") or "Sem título"
            fonte = meta_dict.get("fonte") or f"Arquivo local: {meta_dict.get('pdf_name', 'desconhecido')}"
            data_field = meta_dict.get("data") or meta_dict.get("data_status") or ""
            tipo = meta_dict.get("tipo") or meta_dict.get("chunk_type") or "texto"
            data.append((chunk_id, doc_id, titulo, fonte, data_field, tipo, text, Json(meta_dict), emb))
        execute_values(
            self.cur,
            "INSERT INTO dados (chunk_id, doc_id, titulo, fonte, data, tipo, text, metadata, embedding) VALUES %s ON CONFLICT (chunk_id) DO NOTHING",
            data
        )
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()