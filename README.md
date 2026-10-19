# Ingestão e Busca Semântica com LangChain e Postgres

Desafio técnico do MBA em Engenharia de Software com IA (Full Cycle): aplicação de RAG (Retrieval-Augmented Generation) que ingere um PDF em um banco vetorial e responde perguntas sobre o conteúdo via linha de comando.

## O que o projeto faz

1. **Ingestão** (`src/ingest.py`)
   - Lê o PDF `document.pdf`;
   - Divide o conteúdo em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`);
   - Gera embeddings de cada chunk com o modelo `models/gemini-embedding-2`;
   - Armazena os vetores no **PostgreSQL + pgVector** (via `PGVector` do LangChain).

2. **Busca / Chat** (`src/chat.py` + `src/search.py`)
   - Recebe a pergunta no terminal;
   - Vetoriza a pergunta e busca os **10 resultados mais relevantes (k=10)** no banco vetorial;
   - Monta o prompt com o contexto recuperado e as regras de resposta fechada;
   - Chama o LLM (`gemini-3.1-flash-lite`) e retorna a resposta.

O prompt instrui o modelo a responder **somente com base no contexto**. Se a resposta não estiver no documento, deve retornar exatamente: `Não tenho informações necessárias para responder sua pergunta.`

## Tecnologias

- Python 3.11+
- LangChain (`langchain`, `langchain-community`, `langchain-google-genai`, `langchain-postgres`, `langchain-text-splitters`)
- PostgreSQL 17 + pgVector (imagem `pgvector/pgvector:pg17`, subida via Docker Compose)
- Google Gemini API (embeddings + LLM)

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   └── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## Pré-requisitos

- Docker + Docker Compose
- Python 3.11 ou superior
- API Key do Gemini (gratuita no Google AI Studio): https://aistudio.google.com/apikey

## Configuração

1. Crie e ative o ambiente virtual e instale as dependências:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Copie o template de variáveis de ambiente e preencha sua API key:

```
cp .env.example .env
```

No `.env`:

- `GOOGLE_API_KEY` — sua API key do Gemini;
- `GOOGLE_EMBEDDING_MODEL` — modelo de embeddings (padrão: `models/gemini-embedding-2`);
- `GOOGLE_LLM_MODEL` — modelo de resposta (padrão: `gemini-3.1-flash-lite`);
- `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME` e `PDF_PATH` — já apontam para o banco do Docker Compose, a coleção vetorial e o `document.pdf`.

> O arquivo `.env` não é versionado (veja `.gitignore`).

## Execução

Na raiz do projeto:

```
# 1. Sobe o banco de dados (PostgreSQL + pgVector)
docker compose up -d

# 2. Executa a ingestão do PDF
python src/ingest.py

# 3. Abre o chat no terminal
python src/chat.py
```

Exemplo de uso:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Faça sua pergunta:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para encerrar o chat, digite `sair` (ou `exit` / `quit`).

## Notas de implementação

- **Reingestão idempotente:** a coleção vetorial é recriada a cada execução da ingestão (`pre_delete_collection=True`), então rodar `python src/ingest.py` novamente não duplica registros.
- **Troca de modelo de embeddings:** modelos diferentes geram vetores com dimensões diferentes. Como a coleção é recriada na ingestão, basta rodar a ingestão novamente após trocar o modelo.
- **Modelos:** nomes e versões de modelos mudam com frequência — consulte a documentação oficial do Gemini (https://ai.google.dev/gemini-api/docs/models) para verificar os modelos disponíveis no momento. Os defaults usados aqui são modelos leves/baratos, suficientes para o volume deste desafio.
