from typing import Annotated, List, Dict, Any, Generator, TypedDict
from rag.retrieval.retrieval_node import retrieve
from rag.augmented.augmented_node import generate_stream
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig


# Definição do Estado do Grafo
class RAGState(TypedDict):
    question: str
    retrieval_mode: str
    top_k: int
    docs: List[Document]
    answer: Generator[str, None, None]
    generated_query: str
    generated_filter: str
    retrieval_modes: Dict[str, List[Dict[str, Any]]]
    messages: Annotated[list, add_messages]

def build_streaming_graph(collection_name: str = "dados", k: int = 5):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_stream)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

COMPILED_GRAPH = build_streaming_graph()

def run_streaming_rag(
    question: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
) -> Generator[Dict[str, Any], None, None]:
    """Executa o fluxo RAG e retorna eventos para o frontend."""

    run_config = RunnableConfig(
        run_name="Chat",
        tags=["live-demo", "licitacoes", retrieval_mode],
        metadata={"collection": "dados", "k": top_k, "mode": retrieval_mode, "user": "AnalistaJuridico"},
    )

    initial_state = {
        "question": question,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "messages": [],
    }
    final_state = {}

    for event in COMPILED_GRAPH.stream(initial_state, config=run_config):
        if "retrieve" in event:
            output = event["retrieve"]
            yield {
                "type": "details",
                "data": {
                    "query": output["generated_query"],
                    "filter": output["generated_filter"],
                    "mode": output.get("retrieval_mode", retrieval_mode),
                    "top_k": output.get("top_k", top_k),
                },
            }

        if "generate" in event:
            answer_stream = event["generate"]["answer"]
            for token in answer_stream:
                yield {"type": "token", "data": token}

        if END in event:
            final_state = event[END]

    docs = final_state.get("docs", [])
    sources = []
    for d in docs:
        md = d.get("metadata", {})
        sources.append(
            {
                "score": d.get("score"),
                "pdf_name": md.get("pdf_name"),
                "titulo": md.get("titulo"),
                "fonte": md.get("fonte"),
                "tipo": md.get("tipo"),
                "data": md.get("data"),
                "data_status": md.get("data_status"),
                "data_status_ano": md.get("data_status_ano"),
                "status_atual": md.get("status_atual"),
                "numero_documento": md.get("numero_documento"),
                "chunk_type": md.get("chunk_type"),
                "chunk_index": md.get("chunk_index"),
                "doc_id": md.get("doc_id", d.get("doc_id")),
                "chunk_id": md.get("chunk_id", d.get("chunk_id")),
            }
        )
    yield {"type": "sources", "data": sources}
