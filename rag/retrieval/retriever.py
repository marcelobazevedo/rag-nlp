import os
import json
import unicodedata
import re
from dotenv import load_dotenv
import psycopg2
from typing import Dict, List
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from rag.graph.model_provider import embed_text

def _build_tokenizer():
    stemmer = RSLPStemmer()
    try:
        pt_stops = set(stopwords.words("portuguese"))
    except Exception:
        pt_stops = set()

    def _tokenize(text: str) -> list[str]:
        # normaliza unicode e lowercase
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        # mantém apenas letras e dígitos
        tokens = re.findall(r"[a-z0-9]+", text)
        # remove stopwords e aplica stemmer
        return [stemmer.stem(t) for t in tokens if t not in pt_stops and len(t) > 1]

    return _tokenize

_tokenize = _build_tokenizer()

class HybridRetriever:
	def __init__(self):
		load_dotenv()
		self.conn = psycopg2.connect(
			host=os.getenv("POSTGRES_HOST"),
			port=int(os.getenv("POSTGRES_PORT", 5432)),
			user=os.getenv("POSTGRES_USER"),
			password=os.getenv("POSTGRES_PASSWORD"),
			dbname=os.getenv("POSTGRES_DB"),
		)
		self.cur = self.conn.cursor()
		self.chunks: List[Dict] = []
		self.texts: List[str] = []
		self._load_chunks()
		tokenized = [_tokenize(c) for c in self.texts] if self.texts else [[""]]
		self.bm25 = BM25Okapi(tokenized)

	def _load_chunks(self):
		self.cur.execute("SELECT id, chunk_id, doc_id, text, metadata, embedding FROM dados")
		for row in self.cur.fetchall():
			emb = row[5]
			if isinstance(emb, str):
				try:
					emb = json.loads(emb)
				except Exception:
					emb = None
			metadata = row[4] or {}
			if isinstance(metadata, str):
				try:
					metadata = json.loads(metadata)
				except Exception:
					metadata = {}
			metadata.setdefault("chunk_id", row[1])
			metadata.setdefault("doc_id", row[2])
			chunk = {
				"id": row[0],
				"doc_id": row[2],
				"chunk_id": row[1],
				"text": row[3],
				"metadata": metadata,
				"embedding": emb,
			}
			self.texts.append(row[3])
			self.chunks.append(chunk)

	def dense_search(self, query: str, k: int = 10) -> List[dict]:
		if not self.chunks:
			return []
		query_emb = np.array(embed_text(query), dtype=np.float32)
		chunks_with_emb = [c for c in self.chunks if c.get("embedding") is not None]
		if not chunks_with_emb:
			return []

		scores = []
		for c in chunks_with_emb:
			emb = np.array(c["embedding"], dtype=np.float32)
			if emb.shape != query_emb.shape:
				scores.append(float("-inf"))
				continue
			scores.append(float(np.dot(query_emb, emb)))

		topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
		return [
			{
				"id": chunks_with_emb[i]["id"],
				"doc_id": chunks_with_emb[i]["doc_id"],
				"chunk_id": chunks_with_emb[i]["chunk_id"],
				"text": chunks_with_emb[i]["text"],
				"metadata": chunks_with_emb[i]["metadata"],
				"score": float(scores[i]),
			}
			for i in topk_idx
		]

	def bm25_search(self, query: str, k: int = 10) -> List[dict]:
		if not self.chunks:
			return []
		scores = self.bm25.get_scores(_tokenize(query))
		topk_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
		return [
			{
				"id": self.chunks[i]["id"],
				"doc_id": self.chunks[i]["doc_id"],
				"chunk_id": self.chunks[i]["chunk_id"],
				"text": self.chunks[i]["text"],
				"metadata": self.chunks[i]["metadata"],
				"score": float(scores[i]),
			}
			for i in topk_idx
		]

	def hybrid_search(self, query: str, k: int = 10, rrf_k: int = 60) -> List[dict]:
		# Pool interno grande garante recall: chunks relevantes podem estar além do rank k
		# em cada modo individualmente. 60 candidatos mínimos cobrem a maioria dos casos.
		internal_k = max(k * 8, 60)
		dense_results = self.dense_search(query, internal_k)
		sparse_results = self.bm25_search(query, internal_k)
		return reciprocal_rank_fusion([dense_results, sparse_results], k=k, rrf_k=rrf_k)

	def get_mode(self, mode: str, query: str, k: int = 10) -> List[dict]:
		if mode == "dense":
			return self.dense_search(query, k)
		if mode == "sparse":
			return self.bm25_search(query, k)
		if mode == "hybrid":
			return self.hybrid_search(query, k)
		raise ValueError(f"Modo de recuperação desconhecido: {mode}")

	def close(self):
		self.cur.close()
		self.conn.close()

def reciprocal_rank_fusion(results_lists: List[List[dict]], k: int = 10, rrf_k: int = 60) -> List[dict]:
	scores = {}
	for results in results_lists:
		for rank, item in enumerate(results):
			doc_key = item["id"]
			scores.setdefault(doc_key, {"item": item, "score": 0.0})
			scores[doc_key]["score"] += 1.0 / (rrf_k + rank + 1)
	ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
	fused = []
	for row in ranked[:k]:
		item = dict(row["item"])
		item["score"] = float(row["score"])
		fused.append(item)
	return fused

