from typing import Dict
from rag.graph.utils import _format_docs
from rag.graph.model_provider import build_chat_model
from langchain_core.runnables import RunnableConfig

def generate_stream(state, config: RunnableConfig) -> Dict:
    """Nó que gera a resposta final em formato de stream."""
    print("Executando o nó de geração...")
    docs = state.get("docs", [])
    if not docs:
        # Sem qualquer evidência recuperada, mantém fallback estrito.
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
    def stream_ollama(prompt):
        llm = build_chat_model(temperature=0)
        # O método stream retorna um gerador de chunks
        for chunk in llm.stream(prompt):
            yield chunk.content
    answer_stream = stream_ollama(prompt)
    return {"answer": answer_stream}
