# rag-nlp

Aplicação RAG em Streamlit para consulta jurídica sobre a Lei 14.133/2021, pareceres e notas jurídicas. O projeto faz ingestão de PDFs, indexação em PostgreSQL com pgvector, recuperação `dense`, `sparse` e `hybrid`, e geração de respostas com citações por chunk.

## Visão geral

Fluxo principal:

```text
PDFs em documentos/
  -> ingestão e chunking
  -> embeddings + armazenamento em PostgreSQL/pgvector
  -> recuperação de trechos relevantes
  -> resposta no app com citações
```

Arquivos centrais:

- [app.py](/home/marcelo/Development/rag-nlp/app.py): interface Streamlit
- [rag_graph.py](/home/marcelo/Development/rag-nlp/rag/graph/rag_graph.py): orquestração do fluxo RAG
- [extract_text.py](/home/marcelo/Development/rag-nlp/rag/ingest/extract_text.py): ingestão, metadados e chunking
- [retrieval_node.py](/home/marcelo/Development/rag-nlp/rag/retrieval/retrieval_node.py): recuperação e seleção do modo de busca
- [augmented_node.py](/home/marcelo/Development/rag-nlp/rag/augmented/augmented_node.py): geração da resposta final

## Pré-requisitos

- Docker e Docker Compose
- `uv`
- Ollama rodando no host se `MODELO_LOCAL=true`
- chave válida da OpenAI se `MODELO_LOCAL=false`

## Configuração

Crie o arquivo `.env`:

```bash
cp .env_sample .env
```

Variáveis principais:

- `MODELO_LOCAL=true` usa Ollama para LLM e embeddings
- `MODELO_LOCAL=false` usa OpenAI
- `PGVECTOR_DIM` precisa ser compatível com a dimensão do embedding escolhido
- `CHUNK_SIZE` e `CHUNK_OVERLAP` são lidos do `.env` e controlam a segmentação dos documentos

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

## Subir os serviços

```bash
docker compose up -d --build
docker compose ps
```

Endpoints locais:

- Streamlit em `http://127.0.0.1:8501`
- PostgreSQL em `127.0.0.1:5432`

## Instalar dependências locais

```bash
uv sync
```

Esse passo instala as dependências Python do projeto. Na primeira execução local, o código também tenta garantir automaticamente os recursos `stopwords` e `rslp` do NLTK, usados no tokenizer do modo `sparse`.

Para preparar o ambiente local completo, ainda é necessário:

- configurar o arquivo `.env`
- ter PostgreSQL com `pgvector`
- ter Ollama ativo ou uma chave válida da OpenAI

Se quiser baixar os recursos do NLTK manualmente antes da primeira execução:

```bash
uv run python -m nltk.downloader stopwords rslp
```

## Ingerir o corpus

Coloque os PDFs em [documentos/](/home/marcelo/Development/rag-nlp/documentos) e execute:

```bash
uv run python -m rag.ingest.extract_text
```

Durante a ingestão, o projeto:

- extrai texto com `pypdf`
- infere metadados básicos por nome de arquivo e conteúdo
- divide o texto em chunks
- gera embeddings
- grava os registros na tabela `dados`

Metadados principais por chunk:

- `doc_id`
- `titulo`
- `fonte`
- `data`
- `tipo`
- `numero_documento`
- `chunk_type`

### Chunking

Configuração definida no `.env`:

- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`

O chunking tenta quebrar primeiro em fronteiras naturais como artigos, capítulos e seções. Quando isso não é possível, usa overlap para preservar contexto entre trechos consecutivos.

## Limpar a base antes de reingerir

```bash
uv run python -m rag.ingest.truncate_data
```

Esse comando remove os registros da tabela `dados` e reinicia a sequência de IDs. Se você trocar o modelo de embedding ou alterar a dimensão do vetor, vale limpar a base antes de uma nova ingestão.

## Executar o aplicativo

Com os containers ativos e a base ingerida:

```bash
uv run streamlit run app.py
```

No app, é possível escolher:

- modo `dense`
- modo `sparse`
- modo `hybrid`
- `top-k` da recuperação

## Avaliação

O repositório inclui scripts para avaliação de recuperação e análise qualitativa, com dados de apoio na pasta [data/](/home/marcelo/Development/rag-nlp/data).

Exemplo de `Recall@k`:

```bash
uv run python -m rag.evaluation.recall_eval --dataset data/eval_dataset.json
```

Também estão disponíveis:

- [retrieval_tradeoff_analysis.py](/home/marcelo/Development/rag-nlp/rag/evaluation/retrieval_tradeoff_analysis.py)
- [qualitative_eval.py](/home/marcelo/Development/rag-nlp/rag/evaluation/qualitative_eval.py)

## Estrutura do projeto

```text
app.py
Dockerfile
docker-compose.yaml
.env_sample
README.md

documentos/
data/
rag/
  augmented/
  evaluation/
  graph/
  ingest/
  retrieval/
```

## Observações

- a recuperação híbrida usa fusão RRF entre resultados densos e esparsos
- a indexação usa `chunk_id` composto por documento e chunk para evitar colisões
- se a configuração de ambiente estiver incompleta, o projeto falha cedo com mensagens explícitas
- `uv sync` prepara o ambiente Python, mas a execução local ainda depende de banco, `.env` e provider de modelo
