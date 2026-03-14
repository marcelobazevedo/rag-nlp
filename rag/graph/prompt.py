from langchain.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="numero_documento",
        description=(
            "- Numero identificador do documento (ex.: '14.133/2021', 'Parecer 10/2025').\n"
            "- Use este campo quando o usuario perguntar por lei, parecer ou nota especifica."
        ),
        type="string",
    ),
    AttributeInfo(
        name="tipo",
        description="Tipo do documento (ex.: 'lei', 'parecer', 'nota_juridica', 'outro').",
        type="string",
    ),
    AttributeInfo(
        name="data",
        description=("Data textual do documento no formato 'DD/MM/AAAA' (string), quando disponivel.\n"),
        type="string",
    ),
    AttributeInfo(
        name="assunto",
        description=(
            "Assunto principal do documento (ex.: fase preparatoria, contratacao direta, termo de referencia).\n"
            "- Pode ser usado para refinar busca por tema de licitacao.\n"
        ),
        type="string",
    ),
    AttributeInfo(
        name="pdf_name",
        description="Nome do arquivo PDF de origem.",
        type="string",
    ),
    AttributeInfo(
        name="chunk_type",
        description="Tipo do chunk: 'conteudo_principal', 'referencias_normativas' ou 'precedentes'.",
        type="string",
    ),
    AttributeInfo(
        name="chunk_index",
        description="Índice do chunk no documento.",
        type="integer",
    ),
]
document_content_description = """
    Colecao de trechos (chunks) de documentos juridicos sobre a Lei 14.133/2021,
    incluindo pareceres e notas juridicas, com metadados como numero_documento,
    tipo, assunto, data, nome do arquivo (pdf_name) e tipo de trecho (chunk_type).\n\n
"""

SYSTEM_PROMPT_JURIDICO = """
Voce e um Assistente Juridico Especialista em licitacoes publicas e Lei 14.133/2021.

## Contexto

Voce recebera uma pergunta "{question}" do usuario e um conjunto de trechos de documentos "{context}".

Sua diretriz principal e a FIDELIDADE AO TEXTO. Voce deve responder usando apenas os trechos exatos e literais dos documentos fornecidos no contexto.

Estruture sua resposta da seguinte maneira:

1.  **Introducao Direta**: Comece com uma frase objetiva respondendo a pergunta do usuario.

2.  **Apresentacao Organizada**: Para cada documento ou trecho relevante encontrado no contexto, crie uma secao clara e separada.

3.  **Formato de Citação**: Use o seguinte formato para cada seção:
    "**Conforme o documento [doc_id#chunk_id]:**"

4.  **Extracao Literal**: Abaixo do titulo, insira o trecho literal relevante do documento.

**Restrições Obrigatórias:**
- Fundamente TODA a sua resposta exclusivamente no contexto fornecido.
- Nao adicione opinioes, interpretacoes, exemplos ou informacoes externas de qualquer natureza.
"""
