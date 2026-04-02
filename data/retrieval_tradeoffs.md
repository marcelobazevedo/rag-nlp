# Analise de trade-offs da recuperacao

- Itens avaliados para retrieval: 18
- Itens de recusa excluidos: 2
- Politica de avaliacao: usa query expansion e desativa named_doc para manter comparabilidade entre dense, sparse e hybrid.

## Recall medio por modo

| modo | k=3 | k=5 | k=10 |
| --- | ---: | ---: | ---: |
| dense | 0.1481 | 0.2037 | 0.1944 |
| sparse | 0.0833 | 0.1315 | 0.2769 |
| hybrid | 0.2083 | 0.2639 | 0.3287 |

## Comparacao do hybrid com os modos isolados

### k=3

- `hybrid` melhor que `dense`: 6 perguntas
- `dense` melhor que `hybrid`: 3 perguntas
- `hybrid` melhor que `sparse`: 6 perguntas
- `sparse` melhor que `hybrid`: 3 perguntas
- todos empatados: 6 perguntas

### k=5

- `hybrid` melhor que `dense`: 5 perguntas
- `dense` melhor que `hybrid`: 1 perguntas
- `hybrid` melhor que `sparse`: 8 perguntas
- `sparse` melhor que `hybrid`: 5 perguntas
- todos empatados: 3 perguntas

### k=10

- `hybrid` melhor que `dense`: 7 perguntas
- `dense` melhor que `hybrid`: 1 perguntas
- `hybrid` melhor que `sparse`: 6 perguntas
- `sparse` melhor que `hybrid`: 5 perguntas
- todos empatados: 4 perguntas

## Leitura curta dos trade-offs

- `hybrid` tende a ajudar quando a pergunta mistura termos literais importantes com formulacoes mais semanticas, porque combina BM25 com embeddings.
- `sparse` tende a se sair melhor quando o texto da pergunta repete termos muito especificos do corpus, como numero de parecer, expressao normativa ou redacao quase literal.
- `dense` tende a ajudar mais em perguntas parafraseadas, em que o usuario nao usa exatamente os mesmos termos dos documentos.
- `hybrid` pode piorar em alguns casos quando a fusao RRF traz ruido de um dos modos e rebaixa um chunk muito forte que apareceria mais acima em um retriever isolado.

## Perguntas em que o hybrid perdeu para algum modo isolado

- `k=3`: Quais as modalidades de licitação existentes segundo a Lei 14.133/2021? | dense=0.67, sparse=0.00, hybrid=0.33
- `k=5`: Quais são os casos de inexigibilidade de licitação previstos na Lei 14.133/2021? | dense=0.50, sparse=0.00, hybrid=0.00
- `k=10`: Quais são os casos de inexigibilidade de licitação previstos na Lei 14.133/2021? | dense=0.50, sparse=0.00, hybrid=0.00
- `k=10`: O que a Lei 14.133/2021 estabelece sobre nulidade de licitação e seus efeitos jurídicos? | dense=0.33, sparse=0.67, hybrid=0.33
- `k=3`: Quais sanções administrativas podem ser aplicadas a responsáveis por infrações na Lei 14.133/2021? | dense=0.00, sparse=0.50, hybrid=0.25
- `k=5`: Quais sanções administrativas podem ser aplicadas a responsáveis por infrações na Lei 14.133/2021? | dense=0.00, sparse=0.50, hybrid=0.25
- `k=3`: Qual é o conteúdo e as conclusões do Parecer 8/2024 da AGU? | dense=0.33, sparse=0.00, hybrid=0.00
- `k=5`: O que o Parecer 10/2025 estabelece sobre a designação de gestores e fiscais de contratos? | dense=0.00, sparse=0.33, hybrid=0.00
- `k=10`: O que o Parecer 10/2025 estabelece sobre a designação de gestores e fiscais de contratos? | dense=0.00, sparse=1.00, hybrid=0.33
- `k=3`: O que o Parecer 8/2025 trata sobre pagamento antecipado em contratos administrativos? | dense=0.00, sparse=0.33, hybrid=0.00

## Perguntas em que o hybrid superou algum modo isolado

- `k=3`: Quais as modalidades de licitação existentes segundo a Lei 14.133/2021? | dense=0.67, sparse=0.00, hybrid=0.33
- `k=5`: Quais as modalidades de licitação existentes segundo a Lei 14.133/2021? | dense=0.67, sparse=0.33, hybrid=0.67
- `k=10`: Quais as modalidades de licitação existentes segundo a Lei 14.133/2021? | dense=0.67, sparse=0.33, hybrid=0.67
- `k=3`: Em quais hipóteses é dispensável a licitação conforme o artigo 75 da Lei 14.133/2021? | dense=1.00, sparse=0.00, hybrid=1.00
- `k=5`: Em quais hipóteses é dispensável a licitação conforme o artigo 75 da Lei 14.133/2021? | dense=1.00, sparse=0.00, hybrid=1.00
- `k=10`: Em quais hipóteses é dispensável a licitação conforme o artigo 75 da Lei 14.133/2021? | dense=1.00, sparse=0.00, hybrid=1.00
- `k=3`: Quais documentos são obrigatórios no processo de contratação direta segundo a Lei 14.133/2021? | dense=0.00, sparse=0.00, hybrid=0.50
- `k=5`: Quais documentos são obrigatórios no processo de contratação direta segundo a Lei 14.133/2021? | dense=0.50, sparse=0.00, hybrid=0.50
- `k=10`: Quais documentos são obrigatórios no processo de contratação direta segundo a Lei 14.133/2021? | dense=0.00, sparse=0.50, hybrid=0.50
- `k=3`: O que a Lei 14.133/2021 estabelece sobre nulidade de licitação e seus efeitos jurídicos? | dense=0.00, sparse=0.33, hybrid=0.33
