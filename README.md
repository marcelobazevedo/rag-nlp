# rag-nlp

Aplicação RAG em Streamlit para consulta jurídica sobre a Lei 14.133/2021, pareceres e notas jurídicas. O projeto cobre ingestão de PDFs, chunking, indexação em PostgreSQL com pgvector, recuperação `dense`/`sparse`/`hybrid`, execução do chatbot e avaliação com `Recall@k` e rubrica qualitativa.


## Trilha implementada

Trilha implementada no projeto:

- ingestão de corpus em PDF
- chunking com fronteiras naturais e overlap
- indexação vetorial em PostgreSQL com pgvector
- recuperação `dense`, `sparse` (BM25) e `hybrid` (RRF)
- chatbot RAG em Streamlit com citações por chunk
- avaliação de retrieval com `Recall@k`
- análise comparativa entre `dense`, `sparse` e `hybrid`
- avaliação qualitativa com rubrica

## Visão geral da arquitetura

```text
documentos/ (PDFs)
  -> rag/ingest/extract_text.py
  -> rag/ingest/pgvector_store.py
  -> PostgreSQL + pgvector
  -> rag/retrieval/retrieval_node.py
  -> rag/augmented/augmented_node.py
  -> app.py
```

Arquivos principais:

- [app.py](/home/marcelo/Development/rag-nlp/app.py): interface Streamlit
- [extract_text.py](/home/marcelo/Development/rag-nlp/rag/ingest/extract_text.py): ingestão e chunking
- [pgvector_store.py](/home/marcelo/Development/rag-nlp/rag/ingest/pgvector_store.py): persistência no banco
- [retriever.py](/home/marcelo/Development/rag-nlp/rag/retrieval/retriever.py): `dense`, `sparse` e `hybrid`
- [retrieval_node.py](/home/marcelo/Development/rag-nlp/rag/retrieval/retrieval_node.py): lógica de recuperação
- [augmented_node.py](/home/marcelo/Development/rag-nlp/rag/augmented/augmented_node.py): geração da resposta
- [recall_eval.py](/home/marcelo/Development/rag-nlp/rag/evaluation/recall_eval.py): `Recall@k`
- [retrieval_tradeoff_analysis.py](/home/marcelo/Development/rag-nlp/rag/evaluation/retrieval_tradeoff_analysis.py): comparação entre modos
- [qualitative_eval.py](/home/marcelo/Development/rag-nlp/rag/evaluation/qualitative_eval.py): geração da base para avaliação qualitativa

## Pré-requisitos

- Docker e Docker Compose
- `uv`
- Ollama rodando no host se `MODELO_LOCAL=true`
- chave válida da OpenAI se `MODELO_LOCAL=false`

## Instalação e configuração

Crie o arquivo `.env`:

```bash
cp .env_sample .env
```

Exemplo:

```dotenv
MODELO_LOCAL=true

LLM_MODEL=ministral-3:14b
EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_URL=http://host.docker.internal:11434

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

PGVECTOR_DIM=768
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=rag
POSTGRES_PASSWORD=senha_forte_123
POSTGRES_DB=rag_db

STREAMLIT_BIND_ADDRESS=0.0.0.0
CHUNK_SIZE=900
CHUNK_OVERLAP=120
```

Instale as dependências Python:

```bash
uv sync
```

Se quiser garantir os recursos do NLTK manualmente:

```bash
uv run python -m nltk.downloader stopwords rslp
```

Suba os serviços:

```bash
docker compose up -d --build
docker compose ps
```

Endpoints locais:

- Streamlit em `http://127.0.0.1:8501`
- PostgreSQL em `127.0.0.1:5432`

Observação:

- `uv sync` prepara o ambiente Python
- para rodar localmente ainda é necessário banco com `pgvector`, `.env` válido e provider de modelo configurado

## Corpus e fontes

O corpus está incluído na pasta [documentos/](/home/marcelo/Development/rag-nlp/documentos).

Documentos atualmente presentes:

- Lei 14.133/2021: `L14133.pdf`
- notas jurídicas: `Nota_Juridica_1_2024.pdf`, `Nota_3_2024.pdf`
- pareceres: `Parecer_1_2023.pdf`, `Parecer_8_2024.pdf`, `Parecer_10_2025.pdf` e demais arquivos da mesma pasta

As referências de fonte utilizadas pelo projeto são os próprios PDFs do corpus, que são exibidos no app como `Fontes consultadas`.

No estado atual do repositório, o corpus já está versionado localmente. Não há script separado para download/geração desses arquivos.

## Scripts do projeto

### Ingestão + chunking

Executa extração de texto, inferência de metadados, chunking e geração de embeddings:

```bash
uv run python -m rag.ingest.extract_text
```

Arquivo principal:

- [extract_text.py](/home/marcelo/Development/rag-nlp/rag/ingest/extract_text.py)

### Indexação

A indexação está integrada ao pipeline de ingestão. Os chunks e embeddings são gravados na tabela `dados` do PostgreSQL com pgvector:

- [pgvector_store.py](/home/marcelo/Development/rag-nlp/rag/ingest/pgvector_store.py)

Para limpar a tabela antes de reingerir:

```bash
uv run python -m rag.ingest.truncate_data
```

Para fazer um reset mais completo do projeto, truncando a tabela, removendo artefatos regeneráveis de avaliação e limpando caches locais:

```bash
uv run python -m rag.ingest.reset_project_state
```

### Execução do chatbot

Com o banco populado:

```bash
uv run streamlit run app.py
```

No app, é possível alternar entre:

- `dense`
- `sparse`
- `hybrid`
- `top-k` da recuperação

### Avaliação

#### Recall@k

```bash
uv run python -m rag.evaluation.recall_eval --dataset data/eval_dataset.json
```

Esse script calcula `Recall@3`, `Recall@5` e `Recall@10` para:

- `dense`
- `sparse`
- `hybrid`

#### Trade-offs de retrieval

```bash
uv run python -m rag.evaluation.retrieval_tradeoff_analysis --dataset data/eval_dataset.json
```

Arquivos gerados:

- [retrieval_tradeoff_analysis.json](/home/marcelo/Development/rag-nlp/data/retrieval_tradeoff_analysis.json)
- [retrieval_tradeoffs.md](/home/marcelo/Development/rag-nlp/data/retrieval_tradeoffs.md)

#### Rubrica qualitativa

```bash
uv run python -m rag.evaluation.qualitative_eval --dataset data/eval_dataset.json --mode hybrid --top-k 5 --limit 15
```

Arquivos de apoio:

- [qualitative_rubric.md](/home/marcelo/Development/rag-nlp/data/qualitative_rubric.md)
- [qualitative_eval_results.jsonl](/home/marcelo/Development/rag-nlp/data/qualitative_eval_results.jsonl)
- [qualitative_eval_results.csv](/home/marcelo/Development/rag-nlp/data/qualitative_eval_results.csv)

## Como os experimentos foram organizados

### Chunking

Configuração lida do `.env`:

- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`

O chunking tenta quebrar primeiro em fronteiras naturais como artigos, capítulos e seções. Quando isso não é possível, usa overlap para preservar contexto entre trechos consecutivos.

### Retrieval

Os experimentos comparam três modos:

- `dense`: busca vetorial por embeddings
- `sparse`: busca lexical com BM25
- `hybrid`: fusão RRF entre dense e sparse

### Golden set

O conjunto de avaliação está em [eval_dataset.json](/home/marcelo/Development/rag-nlp/data/eval_dataset.json).

Ele foi estruturado com:

- `question`
- `relevant`: lista de `chunk_id`s esperados
- `expected_behavior: "refuse"` para perguntas fora do escopo do corpus

Esse formato foi adotado para avaliar retrieval com `Recall@k`. A qualidade da resposta final é analisada separadamente na rubrica qualitativa.

## Limitações atuais

- o fluxo principal de execução foi preparado para ambiente local/Docker, não para notebook de Colab
- a política de recusa por evidência insuficiente está descrita no prompt, mas não há um guard obrigatório integrado ao pipeline final
- o corpus está versionado no repositório, mas não há script separado para download das fontes
- a avaliação qualitativa gera os arquivos-base, mas o preenchimento da rubrica continua sendo manual

## Glossário rápido

- `dense`: retrieval vetorial por embeddings
- `sparse`: retrieval lexical com BM25
- `hybrid`: fusão RRF entre `dense` e `sparse`
- `RRF`: Reciprocal Rank Fusion, usada para combinar rankings de recuperação
- `Recall@k`: fração dos chunks esperados que aparece no top-k recuperado
- `golden set`: conjunto de perguntas com chunks relevantes esperados, usado para avaliação
- `grounding`: restrição de resposta ao contexto recuperado, com citações por chunk
- `recusa`: comportamento esperado quando a base não traz evidência suficiente para responder

## Perguntas prováveis da banca

### Como o golden set foi gerado?

O golden set foi montado manualmente a partir do corpus do projeto. Para cada pergunta respondível, foram identificados os `chunk_id`s que continham a evidência esperada. Para perguntas fora do escopo do corpus, foi usado `expected_behavior: "refuse"`.

### Por que o golden set usa chunks e não respostas prontas?

Porque o foco principal da avaliação automática é o retrieval. O projeto mede se a evidência correta aparece no top-k recuperado. A qualidade da resposta final é avaliada separadamente com a rubrica qualitativa.

### O que significa `Recall@k` neste projeto?

`Recall@k` mede quantos chunks esperados foram recuperados entre os `k` primeiros resultados. Exemplo: se a pergunta tem 2 chunks esperados e apenas 1 aparece no top-10, então `Recall@10 = 0.5`.

### Onde está implementado o BM25?

O retriever esparso está em [retriever.py](/home/marcelo/Development/rag-nlp/rag/retrieval/retriever.py), no método `bm25_search`. Ele faz busca lexical por coincidência de termos e complementa o retriever denso.

### O que significa grounding no chatbot?

Grounding significa que a resposta deve ser produzida com base apenas nos trechos recuperados da base. No projeto, isso é reforçado no prompt de geração, com citação obrigatória por chunk e sem uso de conhecimento externo.

### Como o chunking foi definido?

O projeto usa `CHUNK_SIZE=900` e `CHUNK_OVERLAP=120`, lidos do `.env`. O tamanho-base é por caractere, mas o algoritmo tenta quebrar primeiro em fronteiras naturais como artigos, capítulos e seções.

### Por que existe modo `dense`, `sparse` e `hybrid`?

Porque cada modo cobre um tipo de sinal diferente. `dense` ajuda mais em correspondência semântica, `sparse` ajuda em termos literais e expressões exatas, e `hybrid` tenta combinar os dois usando RRF.

## Execução completa

Fluxo completo para reproduzir o projeto:

```bash
cp .env_sample .env
docker compose up -d --build
uv sync
uv run python -m rag.ingest.extract_text
uv run python -m rag.evaluation.recall_eval --dataset data/eval_dataset.json
uv run python -m rag.evaluation.retrieval_tradeoff_analysis --dataset data/eval_dataset.json
uv run python -m rag.evaluation.qualitative_eval --dataset data/eval_dataset.json --mode hybrid --top-k 5 --limit 15
uv run streamlit run app.py
```

## Estrutura do projeto

```text
app.py
Dockerfile
docker-compose.yaml
.env_sample
README.md

documentos/
data/
  eval_dataset.json
  eval_dataset.sample.json
  qualitative_eval_results.csv
  qualitative_eval_results.jsonl
  qualitative_rubric.md
  retrieval_tradeoff_analysis.json
  retrieval_tradeoffs.md

rag/
  augmented/
  evaluation/
  graph/
  ingest/
  retrieval/
  settings.py
```
