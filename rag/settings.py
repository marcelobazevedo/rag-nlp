import os

from dotenv import load_dotenv

load_dotenv()


def _required_int_env(name: str) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        raise RuntimeError(f"Variavel de ambiente obrigatoria ausente: {name}")

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Variavel de ambiente invalida para inteiro: {name}={raw_value!r}") from exc


CHUNK_SIZE = _required_int_env("CHUNK_SIZE")
CHUNK_OVERLAP = _required_int_env("CHUNK_OVERLAP")
