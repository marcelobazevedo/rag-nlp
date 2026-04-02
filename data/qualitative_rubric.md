# Rubrica de avaliacao qualitativa

Use esta rubrica para pelo menos 15 perguntas do golden set.

Preenchimento recomendado:
- `1` = atende ao criterio
- `0` = nao atende ao criterio

Critérios:

- `groundedness`: a resposta esta apoiada nos trechos recuperados?
- `correction`: a resposta esta correta dado o corpus?
- `citations`: as citacoes `[doc_id#chunk_id]` estao adequadas e coerentes com o conteudo?
- `hallucination`: a resposta evitou inventar informacoes fora da base?
- `refusal`: quando faltou evidencia suficiente, o chatbot recusou corretamente?

Campo livre:

- `review_notes`: observacoes curtas sobre falhas, bons exemplos ou ambiguidades.

Sugestao de uso:

1. Gere os arquivos com `qualitative_eval.py`.
2. Abra `data/qualitative_eval_results.csv`.
3. Preencha as cinco colunas da rubrica e o campo `review_notes`.
4. Use o JSONL quando precisar inspecionar a resposta completa e as fontes recuperadas.
