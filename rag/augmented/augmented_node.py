from typing import Dict

from langchain_core.runnables import RunnableConfig

from rag.graph.model_provider import build_chat_model
from rag.graph.utils import _format_docs


def generate_stream(state, config: RunnableConfig) -> Dict:
    """Nó que gera a resposta final em formato de stream."""
    print("Executando o nó de geração...")
    docs = state.get("docs", [])
    if not docs:
        return {"answer": iter(["não encontrei na base"])}

    context = _format_docs(docs)
    prompt = f"""
Você é um assistente jurídico com política estrita de grounding.

Regras obrigatórias:
1) Responda APENAS com base no CONTEXTO recuperado.
2) Cada afirmação factual deve conter citação no formato [doc_id#chunk_id].
3) Use "não encontrei na base" APENAS quando não houver NENHUMA evidência útil para responder a pergunta.
4) Se houver evidência parcial, responda com o que foi encontrado e deixe explícito o que não foi possível confirmar.
5) Não invente documentos, IDs, artigos ou datas.
6) Não use conhecimento externo.
7) Quando a pergunta pedir "em que documento", sempre informe o documento de origem usando doc_id/pdf_name presentes no contexto.
8) Seja objetivo e não contradiga a própria resposta (evite dizer "não encontrei" e depois citar evidência).

Pergunta: {state['question']}

CONTEXTO:
{context}
"""

    def stream_answer(input_prompt: str):
        llm = build_chat_model(temperature=0)
        for chunk in llm.stream(input_prompt):
            yield chunk.content

    answer_stream = stream_answer(prompt)
    return {"answer": answer_stream}
