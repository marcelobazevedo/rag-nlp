# Relatório de Desenvolvimento, Validação e Avaliação do Sistema RAG

### Alunos:

* Alex Pereinha Maranhão
* Marcelo de Azevedo
* Murilo Aguiar


## 1. Visão geral

Este documento descreve o projeto `rag-nlp`, uma aplicação de Retrieval-Augmented Generation (RAG) voltada à consulta jurídica sobre a Lei 14.133/2021, pareceres e notas jurídicas. O sistema abrange o fluxo completo de ingestão de PDFs, segmentação em chunks, indexação vetorial em PostgreSQL com `pgvector`, recuperação nos modos `dense`, `sparse` e `hybrid`, geração de respostas com citações por trecho e avaliação quantitativa e qualitativa.

O objetivo central do trabalho foi desenvolver um chatbot jurídico capaz de responder com base em um corpus fechado, preservando rastreabilidade das fontes recuperadas e permitindo a comparação objetiva entre diferentes estratégias de recuperação.

No contexto das trilhas propostas pela disciplina, o projeto implementa a **Trilha A: Recuperação Híbrida (Sparse + Dense)**. O retriever **denso** foi adotado como baseline, e a melhoria avançada consiste na inclusão do retriever **esparso** e da fusão **híbrida** entre ambos.

## 2. Escopo do corpus e organização do projeto

O corpus utilizado está na pasta `documentos/` e reúne três grupos principais de fontes:

- texto legal da Lei 14.133/2021
- pareceres jurídicos
- notas jurídicas

No estado atual do repositório, o corpus já está versionado localmente e não depende de etapa externa de download. A arquitetura geral do sistema está organizada da seguinte forma:

```text
documentos/ (PDFs)
  -> rag/ingest/extract_text.py
  -> rag/ingest/pgvector_store.py
  -> PostgreSQL + pgvector
  -> rag/retrieval/retrieval_node.py
  -> rag/augmented/augmented_node.py
  -> app.py
```

Os arquivos centrais do projeto são:

- `rag/ingest/extract_text.py`: leitura dos PDFs, inferência de metadados, chunking e geração de embeddings
- `rag/ingest/pgvector_store.py`: persistência dos chunks e vetores
- `rag/retrieval/retriever.py`: implementação dos motores `dense`, `sparse` e `hybrid`
- `rag/retrieval/retrieval_node.py`: orquestração da recuperação, incluindo expansão de consulta
- `rag/augmented/augmented_node.py`: geração da resposta final
- `rag/evaluation/recall_eval.py`: cálculo de `Recall@k`
- `rag/evaluation/retrieval_tradeoff_analysis.py`: análise comparativa dos modos de recuperação
- `rag/evaluation/qualitative_eval.py`: geração dos artefatos-base da avaliação qualitativa

### 2.1 Aderência aos requisitos mínimos do corpus

O enunciado exige um corpus fechado com metadados mínimos por documento. No projeto, esses requisitos são atendidos da seguinte forma:

- `doc_id`: gerado e normalizado na ingestão
- `titulo`: derivado do nome-base do arquivo PDF
- `fonte`: registrada como descrição da origem do arquivo local
- `data`: inferida do texto quando disponível
- `tipo`: também é armazenado, embora no enunciado apareça como opcional

Dessa forma, o projeto atende ao conjunto mínimo de metadados solicitado para o corpus.

## 3. Desenvolvimento da ingestão e do chunking

O pipeline de ingestão foi implementado em `rag/ingest/extract_text.py`. O processo lê os PDFs com `pypdf`, extrai o texto integral, infere metadados do documento e gera chunks de forma determinística.

### 3.1 Leitura e normalização

Cada PDF é lido página a página e convertido em texto corrido. Em seguida, o pipeline busca inferir:

- tipo do documento (`lei`, `parecer`, `nota_juridica` ou `outro`)
- número do documento
- data textual, quando disponível
- órgão emissor
- assunto resumido

Também é gerado um `doc_id` normalizado, utilizado posteriormente na indexação e na citação dos resultados.

### 3.2 Segmentação por seções jurídicas

Antes da quebra por tamanho, o texto é dividido em seções com base em marcadores frequentes de documentos jurídicos, como:

- `OBJETO`
- `EMENTA`
- `INTRODUÇÃO`
- `FUNDAMENTAÇÃO`
- `ANÁLISE`
- `CONCLUSÃO`
- `PARECER`

Na ausência desses marcadores, o documento é tratado como uma única seção. Essa escolha reduz a mistura de partes com funções distintas dentro do mesmo chunk e melhora o valor informacional dos trechos recuperados.

### 3.3 Chunking com fronteiras naturais e overlap

Depois da separação por seções, o texto é subdividido com base em dois parâmetros do projeto:

- `CHUNK_SIZE=900`
- `CHUNK_OVERLAP=120`

O algoritmo tenta preservar fronteiras naturais encontradas por expressões como `Art.`, `CAPÍTULO` e `SEÇÃO`. Quando não há um ponto de quebra natural adequado, a divisão recorre ao overlap padrão. Na prática, isso busca equilibrar dois objetivos:

- manter coesão jurídica dentro de cada trecho
- evitar perda de contexto entre chunks consecutivos

Cada chunk recebe metadados estruturados, incluindo `doc_id`, `chunk_id`, `tipo`, `numero_documento`, `pdf_name`, `chunk_type` e `chunk_index`.

### 3.4 Tratamento de títulos, seções, listas e tabelas

O enunciado pede que o grupo documente como tratou títulos, seções, listas e tabelas, quando aplicável. No projeto:

- títulos e seções recebem tratamento explícito por meio da identificação de marcadores jurídicos como `OBJETO`, `EMENTA`, `FUNDAMENTAÇÃO` e `CONCLUSÃO`
- fronteiras naturais como `Art.`, `CAPÍTULO` e `SEÇÃO` são usadas para quebrar o texto de forma menos arbitrária
- listas e tabelas não recebem um parser especializado; quando presentes no PDF extraído, permanecem no fluxo linear do texto e são herdadas pelo chunk correspondente

Em síntese, houve tratamento estrutural para seções e títulos, mas não uma etapa dedicada à reconstrução semântica de tabelas ou listas complexas.

## 4. Indexação vetorial e modelos utilizados

Após a geração dos chunks, o pipeline cria embeddings para cada trecho e grava o resultado no PostgreSQL com `pgvector`. A função central para escolha do provedor de modelo está em `rag/graph/model_provider.py`.

O projeto suporta dois modos:

- modelo local via Ollama, com `LLM_MODEL` e `EMBEDDING_MODEL`
- modelo da OpenAI, com `OPENAI_MODEL` e `OPENAI_EMBEDDING_MODEL`

No README do projeto, o exemplo de configuração usa:

- `LLM_MODEL=ministral-3:14b`
- `EMBEDDING_MODEL=nomic-embed-text:latest`
- `OPENAI_MODEL=gpt-4.1-mini`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`

O uso do modelo é distribuído em três pontos principais:

- embeddings na ingestão
- embeddings na busca densa
- geração da resposta final

### 4.1 Parâmetros relevantes do retriever denso

Como a trilha escolhida exige comparação com o baseline, vale registrar objetivamente os principais parâmetros do modo denso:

- índice vetorial: PostgreSQL com `pgvector`
- dimensão vetorial: `PGVECTOR_DIM=768`
- embedding configurável por ambiente
- cálculo de score: produto escalar entre embedding da pergunta e embedding do chunk
- `top-k` configurável, com uso explícito de `k=3`, `k=5` e `k=10` na avaliação

## 5. Estratégias de recuperação implementadas

O sistema compara três estratégias de retrieval:

- `dense`: recuperação vetorial por similaridade entre embeddings
- `sparse`: recuperação lexical com BM25
- `hybrid`: combinação entre `dense` e `sparse` por Reciprocal Rank Fusion (RRF)

Essa organização atende exatamente ao que a Trilha A solicita na apresentação do sistema: exposição dos três modos `dense` (baseline), `sparse` e `hybrid`.

### 5.1 Recuperação densa

Na busca densa, a pergunta do usuário é convertida em embedding e comparada com os embeddings dos chunks armazenados no banco. O score é calculado por produto escalar entre o vetor da consulta e o vetor de cada chunk.

Essa abordagem tende a funcionar melhor quando a pergunta é parafraseada ou semanticamente próxima do conteúdo, mesmo sem repetir exatamente os mesmos termos do corpus.

### 5.2 Recuperação esparsa

Na busca esparsa, o projeto usa `BM25Okapi` com tokenização adaptada para português. Nesse ponto aparece o uso de `nltk`, especificamente para:

- carregar `stopwords` em português
- aplicar o `RSLPStemmer`

O objetivo é melhorar a recuperação lexical sobre o corpus jurídico, reduzindo impacto de palavras muito frequentes e aproximando variações morfológicas por radical.

Os principais parâmetros e decisões do modo `sparse` são:

- algoritmo: `BM25Okapi`
- tokenização por regex alfanumérica
- normalização para minúsculas
- remoção de acentos por normalização Unicode
- remoção de stopwords em português
- stemming com `RSLPStemmer`
- `top-k` configurável

### 5.3 Recuperação híbrida

No modo `hybrid`, os rankings do `dense` e do `sparse` são combinados por RRF. Essa fusão é aplicada no arquivo `rag/retrieval/retriever.py` e busca aproveitar o melhor dos dois sinais:

- capacidade semântica do embedding
- capacidade lexical do BM25

Os resultados mostram que essa combinação foi a estratégia mais forte no dataset atual do projeto.

Os principais parâmetros e comportamentos do modo híbrido são:

- método de fusão: `Reciprocal Rank Fusion`
- parâmetro de fusão: `rrf_k=60`
- conjunto candidato antes do corte final: `internal_k = max(k * 8, 60)`
- corte final: retorno dos `k` melhores itens após a fusão

### 5.4 Deduplicação na fusão

Um dos critérios mínimos da Trilha A é a deduplicação de chunks na combinação dos rankings. No projeto, isso é atendido na função de RRF: quando um mesmo item aparece nas listas `dense` e `sparse`, os scores são acumulados sobre uma única chave e o resultado final mantém apenas uma ocorrência daquele chunk no ranking fundido.

Na implementação atual, a deduplicação é feita pelo identificador interno do item recuperado no banco, o que evita duplicidade do mesmo chunk no ranking final.

## 6. Expansão de consulta e heurísticas de recuperação

O projeto inclui uma etapa explícita de melhoria da pergunta do usuário em `rag/retrieval/retrieval_node.py`, na função `_expand_query()`.

Nessa etapa, a pergunta original é enviada ao modelo com um prompt pedindo reescrita em terminologia jurídica técnica, incluindo sinônimos e termos legais correlatos. A consulta expandida não substitui completamente a consulta original: o sistema executa a recuperação com ambas e depois funde os resultados.

Além disso, o nó de recuperação implementa uma heurística de atalho por nome de documento. Quando a pergunta menciona diretamente um parecer, nota ou lei específica, o sistema tenta localizar os chunks daquele documento antes de seguir para a busca geral.

Essas duas decisões de projeto ajudam a recuperar melhor perguntas formuladas de forma mais livre e perguntas que citam documentos nominais do corpus.

É importante observar, porém, que as avaliações comparativas da Trilha A foram executadas com `use_query_expansion=False` e `use_named_doc=False`, para preservar uma comparação controlada entre os três modos de retrieval. Dessa forma, os resultados quantitativos refletem a diferença entre `dense`, `sparse` e `hybrid`, e não o efeito dessas heurísticas auxiliares.

## 7. Geração da resposta e grounding

Após a recuperação, a geração da resposta é feita em `rag/augmented/augmented_node.py`. O prompt estabelece uma política estrita de grounding, com regras como:

- responder apenas com base no contexto recuperado
- inserir citação factual no formato `[doc_id#chunk_id]`
- não inventar documentos, datas ou artigos
- assumir explicitamente quando o contexto é parcial

Essa camada é importante porque o projeto não se limita à recuperação: ele também tenta condicionar a LLM a produzir uma resposta jurídica rastreável, vinculada aos trechos efetivamente recuperados.

Há também um módulo complementar de verificação em `rag/augmented/evidence_guard.py`, que calcula sinais heurísticos de suficiência de evidência e pode consultar a LLM como juiz. Esse componente reforça a preocupação com recusa e confiabilidade, embora o próprio README registre que a política de recusa ainda não está integrada como guard obrigatório no pipeline principal.

### 7.1 Transparência dos trechos recuperados

O enunciado exige transparência sobre o contexto recuperado. Esse requisito está contemplado no projeto de duas formas:

- o app exibe as fontes consultadas ao usuário
- a interface de avaliação mostra os `chunk_id`s esperados, encontrados, faltantes e os rankings recuperados para `top-3`, `top-5` e `top-10`

Além disso, o fluxo de recuperação registra `generated_query`, `generated_filter`, `doc_id`, `chunk_id` e `score`, o que facilita a auditoria do comportamento do retriever. Assim, o projeto atende ao requisito de expor os chunks recuperados e seus identificadores, ao menos em log e em componentes auxiliares da interface.

### 7.2 Recusa adequada: estado atual

O enunciado trata a recusa adequada como requisito mínimo do pipeline RAG. No projeto, esse requisito foi parcialmente contemplado:

- o prompt de geração instrui a responder `"não encontrei na base"` quando não houver evidência útil
- existe um módulo de verificação de suficiência de evidência

## 8. Estratégia de avaliação

O projeto separa a avaliação em duas frentes:

- avaliação quantitativa do retrieval
- avaliação qualitativa da resposta

### 8.1 Golden set

O conjunto de avaliação está em `data/eval_dataset.json`. O dataset atual contém:

- 18 perguntas respondíveis com chunks relevantes anotados
- 2 perguntas marcadas com `expected_behavior: "refuse"`

As perguntas cobrem:

- consultas diretas sobre a Lei 14.133/2021
- perguntas sobre pareceres e notas jurídicas específicas
- ao menos um caso multi-documento
- casos fora do escopo, usados para testar comportamento de recusa

### 8.2 Métricas utilizadas

Na avaliação automática do retrieval, a métrica utilizada é `Recall@k`, com:

- `k=3`
- `k=5`
- `k=10`

Além disso, a análise comparativa calcula contagens de vitórias entre modos, como:

- `hybrid_vs_dense`
- `hybrid_vs_sparse`
- `dense_vs_hybrid`
- `sparse_vs_hybrid`
- `all_equal`

Na avaliação qualitativa, os artefatos gerados preveem uma rubrica com os campos:

- `groundedness`
- `correction`
- `citations`
- `hallucination`
- `refusal`

### 8.3 Aderência da avaliação ao enunciado

Em relação ao que o PDF exige:

- o golden set tem aproximadamente 20 perguntas, como solicitado
- há perguntas factuais diretas
- há perguntas que exigem múltiplos trechos
- há perguntas fora do corpus para testar recusa
- a avaliação quantitativa calcula `Recall@3`, `Recall@5` e `Recall@10` para os três modos da Trilha A

## 9. Resultados quantitativos do retrieval

Os números consolidados do projeto estão em `data/retrieval_tradeoff_analysis.json` e `data/retrieval_tradeoffs.md`.

Foram avaliadas 18 perguntas respondíveis, com 2 itens de recusa excluídos do cálculo de recall. A política de avaliação utilizada nesses testes desabilita, para fins de comparação controlada, tanto a expansão de consulta quanto o atalho por nome de documento.

### 9.1 Recall médio por modo

| modo | Recall@3 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| dense | 0.1296 | 0.1481 | 0.2241 |
| sparse | 0.0759 | 0.1565 | 0.2630 |
| hybrid | 0.2176 | 0.2685 | 0.4574 |

### 9.2 Leitura dos resultados

Os resultados indicam que o modo `hybrid` obteve o melhor recall médio em todos os valores de `k`. O ganho foi especialmente mais expressivo em `Recall@10`, no qual o híbrido alcançou `0.4574`, quase o dobro do `dense` e substancialmente acima do `sparse`.

Isso sugere que, no corpus jurídico deste projeto, a combinação entre busca lexical e busca vetorial foi mais robusta do que qualquer um dos modos isolados.

### 9.3 Comparação do hybrid com os modos isolados

Para `k=3`:

- `hybrid` superou `dense` em 5 perguntas
- `dense` não superou `hybrid`
- `hybrid` superou `sparse` em 8 perguntas
- `sparse` superou `hybrid` em 4 perguntas
- houve 6 empates totais

Para `k=5`:

- `hybrid` superou `dense` em 6 perguntas
- `dense` não superou `hybrid`
- `hybrid` superou `sparse` em 8 perguntas
- `sparse` superou `hybrid` em 5 perguntas
- houve 5 empates totais

Para `k=10`:

- `hybrid` superou `dense` em 11 perguntas
- `dense` superou `hybrid` em 1 pergunta
- `hybrid` superou `sparse` em 9 perguntas
- `sparse` superou `hybrid` em 4 perguntas
- houve 2 empates totais

Esses números reforçam que o híbrido não vence todos os casos individualmente, mas foi a estratégia mais consistente no agregado.

## 10. Observações qualitativas sobre os trade-offs

O resumo interpretativo versionado no projeto aponta quatro leituras importantes:

- `hybrid` tende a ajudar quando a pergunta mistura termos literais do corpus com formulações mais semânticas
- `sparse` tende a funcionar melhor quando a pergunta repete expressões normativas ou nomes muito específicos
- `dense` tende a ajudar mais quando a pergunta está parafraseada
- o `hybrid` pode piorar em alguns casos quando a fusão introduz ruído de um dos retrievers

Essas observações são compatíveis com a natureza do corpus. Em direito, muitas perguntas dependem de expressões literais, nomes de documentos, numeração normativa e vocabulário técnico. Ao mesmo tempo, perguntas formuladas de forma menos literal podem se beneficiar do sinal semântico da busca vetorial.

## 11. Avaliação qualitativa

O projeto também gera artefatos para avaliação qualitativa manual em:

- `data/qualitative_eval_results.jsonl`
- `data/qualitative_eval_results.csv`

Esses arquivos registram, para cada pergunta processada:

- texto da resposta gerada
- fontes recuperadas
- citações por chunk
- campos da rubrica qualitativa

Isso significa que o relatório pode afirmar com segurança que:

- o projeto implementou o fluxo de geração dos artefatos qualitativos
- o projeto separou os critérios exigidos pela disciplina
- o projeto está preparado para avaliação manual em pelo menos 15 perguntas


Os artefatos mostram que o sistema já produz respostas com:

- referência às fontes recuperadas
- identificação de `doc_id` e `chunk_id`
- distinção entre comportamento respondível e necessidade de cautela diante de evidência parcial

## 12. Limitações atuais do projeto

Com base na implementação e na documentação atual, as principais limitações são:

- a avaliação qualitativa depende de preenchimento manual posterior
- o desempenho absoluto de recall ainda é moderado, inclusive no melhor modo, o que mostra espaço para melhoria de chunking, reranking ou expansão de consulta


## 13. Conclusão técnica

O trabalho realizado entrega um pipeline RAG funcional e coerente para consulta jurídica sobre a Lei 14.133/2021 e documentos correlatos. O sistema cobre ingestão, indexação, recuperação, geração e avaliação, com organização clara dos componentes e preocupação explícita com grounding e rastreabilidade da resposta.

Do ponto de vista experimental, o principal resultado é que a recuperação `hybrid` foi superior às abordagens isoladas no dataset atual, alcançando os melhores valores médios de `Recall@3`, `Recall@5` e `Recall@10`. Esse achado justifica a adoção do híbrido como modo preferencial do projeto no estado atual.

Do ponto de vista de maturidade, o projeto já possui:

- corpus jurídico versionado
- pipeline reprodutível
- avaliação quantitativa formal
- estrutura pronta para avaliação qualitativa

Como próximos passos naturais, destacam-se:

- consolidar a rubrica qualitativa com revisão humana
- testar estratégias de reranking
- revisar perguntas de baixo recall para refinar chunking e recuperação
