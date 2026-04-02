FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY .python-version ./

# Instala apenas as dependencias travadas; o codigo do projeto sera montado depois.
RUN uv sync --frozen --no-dev --no-install-project

COPY rag ./rag
COPY app.py ./
COPY .streamlit ./.streamlit
COPY static ./static
COPY data ./data
COPY documentos ./documentos
COPY initdb ./initdb

RUN uv run --no-sync python -m nltk.downloader stopwords rslp

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
