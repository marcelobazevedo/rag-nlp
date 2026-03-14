import os

import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()


def _is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def is_local_model_enabled() -> bool:
    return _is_true(os.getenv("MODELO_LOCAL"))


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Variavel de ambiente obrigatoria ausente: {name}")
    return value


def _validate_mode_env() -> None:
    """Valida variaveis obrigatorias com mensagem amigavel por modo."""
    if is_local_model_enabled():
        missing = [name for name in ("LLM_MODEL", "EMBEDDING_MODEL") if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                "MODELO_LOCAL=true, mas faltam variaveis no .env: "
                + ", ".join(missing)
                + ". Defina essas chaves para usar modelos locais via Ollama."
            )
        return

    missing = [
        name
        for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_EMBEDDING_MODEL")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "MODELO_LOCAL=false, mas faltam variaveis no .env: "
            + ", ".join(missing)
            + ". Defina essas chaves para usar OpenAI."
        )


def llm_model_name() -> str:
    _validate_mode_env()
    if is_local_model_enabled():
        return _required_env("LLM_MODEL")
    return _required_env("OPENAI_MODEL")


def embedding_model_name() -> str:
    _validate_mode_env()
    if is_local_model_enabled():
        return _required_env("EMBEDDING_MODEL")
    return _required_env("OPENAI_EMBEDDING_MODEL")


def build_chat_model(temperature: float = 0.0):
    _validate_mode_env()
    model = llm_model_name()
    if is_local_model_enabled():
        return ChatOllama(model=model, temperature=temperature)
    return ChatOpenAI(model=model, temperature=temperature)


def generate_text(prompt: str, timeout: int = 15, max_tokens: int = 100) -> str:
    """Gera texto curto para tarefas internas, respeitando MODELO_LOCAL."""
    _validate_mode_env()
    if is_local_model_enabled():
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": llm_model_name(),
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return (response.json().get("response") or "").strip()

    client = OpenAI(api_key=_required_env("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model=llm_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return (completion.choices[0].message.content or "").strip()


def embed_text(text: str) -> list[float]:
    _validate_mode_env()
    if is_local_model_enabled():
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": embedding_model_name(), "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    client = OpenAI(api_key=_required_env("OPENAI_API_KEY"))
    # Mantem compatibilidade com a dimensao da coluna pgvector (ex.: 768).
    vector_dim = int(os.getenv("PGVECTOR_DIM", "768"))
    result = client.embeddings.create(
        model=embedding_model_name(),
        input=text,
        dimensions=vector_dim,
    )
    return result.data[0].embedding
