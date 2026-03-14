# rag-nlp

Chatbot RAG com Streamlit para consulta jurídica da Lei 14.133/2021, pareceres e notas jurídicas.

Suporta recuperação **dense** (pgvector), **sparse** (BM25) e **hybrid** (Reciprocal Rank Fusion), com query expansion automática, citações obrigatórias por chunk e recusa quando não há evidência suficiente.

---

## Arquitetura

```
documentos/ (PDFs)
      │
      ▼
extract_text.py  →  pgvector_store.py  →  PostgreSQL + pgvector
                                                │
                        ┌───────────────────────┘
                        ▼
              retrieval_node.py  (query expansion + HybridRetriever)
                        │
                        ▼
              augmented_node.py  (geração com grounding)
                        │
                        ▼
                    app.py (Streamlit)
```

O fluxo de inferência é orquestrado pelo **LangGraph** (`rag_graph.py`). O provider de modelo (`model_provider.py`) é o único ponto de decisão entre modo local (Ollama) e OpenAI, lendo tudo do `.env`.

---

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) — gerenciador de pacotes e ambiente Python
- [Docker](https://docs.docker.com/) — para subir o Postgres com pgvector
- **Modo local:** [Ollama](https://ollama.com/) instalado e rodando, com os modelos baixados
- **Modo OpenAI:** chave de API válida (`OPENAI_API_KEY`)

---

## Configuração

### 1. Instalar dependências

```bash
uv sync
```

### 2. Criar o arquivo `.env`

```bash
cp .env_sample .env
```

Edite o `.env` conforme o modo desejado:

```dotenv
# ── Modo de operação ─────────────────────────────────────
# true  → usa Ollama (modelos locais)
# false → usa OpenAI (requer OPENAI_API_KEY)
MODELO_LOCAL=false

# ── Modelos locais (Ollama) ── usados quando MODELO_LOCAL=true
LLM_MODEL=ministral-3:14b
EMBEDDING_MODEL=nomic-embed-text:latest

# ── Modelos OpenAI ── usados quando MODELO_LOCAL=false
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# ── Banco de dados ───────────────────────────────────────
PGVECTOR_DIM=768     # deve ser igual à dimensão do modelo de embedding
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=rag
POSTGRES_PASSWORD=senha_forte_123
POSTGRES_DB=rag_db

# ── Chunking ─────────────────────────────────────────────
CHUNK_SIZE=900
CHUNK_OVERLAP=120
```

> **Importante sobre `PGVECTOR_DIM`:** a dimensão deve coincidir com o modelo de embedding escolhido.
> - `nomic-embed-text:latest` → `768`
> - `text-embedding-3-large` (OpenAI) → truncado para `768` automaticamente via parâmetro `dimensions`
>
> Se trocar de modelo, truncate a tabela antes de reingerir (ver seção abaixo).

### 3. Subir o banco de dados

```bash
docker compose up -d
```

O script `initdb/01-pgvector.sql` é executado automaticamente na primeira inicialização e instala a extensão `vector`. A tabela `dados` é criada pelo próprio script de ingestão.

---

## Ingestão de documentos

1. Coloque os arquivos PDF na pasta `documentos/`.
2. Execute a ingestão:

```bash
uv run python -m rag.ingest.extract_text
```

O script:
- Lê cada PDF com `pypdf`
- Infere metadados por regras (nome do arquivo + regex): `doc_id`, `titulo`, `fonte`, `data`, `tipo`
- Faz chunking com quebra em fronteiras naturais (artigos, capítulos, seções) e overlap configurável
- Gera embeddings via Ollama ou OpenAI (conforme `MODELO_LOCAL`)
- Insere no PostgreSQL com `chunk_id UNIQUE` (re-execuções são idempotentes por `ON CONFLICT DO NOTHING`)

### Trocar de modo (local ↔ OpenAI)

Embeddings gerados por modelos diferentes **não são intercambiáveis**. Ao trocar `MODELO_LOCAL`, truncate a tabela antes de reingerir:

```bash
docker exec -i postgres_pgvector_rag psql -U rag -d rag_db -c "TRUNCATE TABLE dados;"
uv run python -m rag.ingest.extract_text
```

---

## Executar o chatbot

```bash
uv run streamlit run app.py
```

Acesse em `http://localhost:8501`.

No menu lateral é possível configurar:
- **Modo de recuperação:** `dense` (vetorial), `sparse` (BM25) ou `hybrid` (RRF)
- **Top-k:** número de chunks recuperados (3, 5 ou 10)

Cada resposta exibe, em um expander, a query expandida e os detalhes da busca. As respostas contêm citações obrigatórias no formato `[doc_id#chunk_id]`.

---

## Avaliação de métricas (Recall@k)

### 1. Preparar o dataset

Crie `data/eval_dataset.json` com base no modelo `data/eval_dataset.sample.json`:

```json
{
  "items": [
    {
      "question": "Qual o prazo para publicação do edital?",
      "relevant": [
        "lei_14133_2021::chunk_042",
        "lei_14133_2021::chunk_043"
      ]
    }
  ]
}
```

Os `chunk_id`s devem estar no formato `doc_id::chunk_NNN`, conforme armazenado no banco.

### 2. Executar a avaliação

```bash
uv run python -m rag.evaluation.recall_eval --dataset data/eval_dataset.json
```

Saída: tabela com **Recall@3**, **Recall@5** e **Recall@10** para os modos `dense`, `sparse` e `hybrid`:

```
Recall@k médio por modo
mode    k=3     k=5     k=10
dense   0.3111  0.4000  0.4889
sparse  0.2444  0.3556  0.4444
hybrid  0.2889  0.3778  0.4556
```

---

## Estrutura do projeto

```
app.py                        # Interface Streamlit
pyproject.toml                # Dependências (uv)
docker-compose.yaml           # Postgres + pgvector
.env_sample                   # Modelo de variáveis de ambiente
documentos/                   # PDFs para ingestão
data/
  eval_dataset.json           # Dataset de avaliação
  eval_dataset.sample.json    # Exemplo de formato
initdb/
  01-pgvector.sql             # Instala extensão vector
rag/
  graph/
    model_provider.py         # Provider central: Ollama ↔ OpenAI
    rag_graph.py              # Grafo LangGraph
    utils.py                  # Utilitários
  ingest/
    extract_text.py           # Pipeline de ingestão de PDFs
    pgvector_store.py         # Persistência no PostgreSQL
  retrieval/
    retriever.py              # HybridRetriever (dense + BM25 + RRF)
    retrieval_node.py         # Nó de recuperação (query expansion)
  augmented/
    augmented_node.py         # Nó de geração com grounding
  evaluation/
    recall_eval.py            # Avaliação Recall@k
```

