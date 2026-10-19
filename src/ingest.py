import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def ingest_pdf():
    print(f"Carregando PDF: {PDF_PATH}")
    documents = PyPDFLoader(PDF_PATH).load()
    print(f"Páginas carregadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(
        f"Documento dividido em {len(chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})"
    )

    print(f"Gerando embeddings com o modelo: {GOOGLE_EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    print(f"Gravando vetores no PostgreSQL (collection: {PG_VECTOR_COLLECTION_NAME})...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        # Recria a coleção a cada execução: reingestão idempotente e
        # reconstrução automática caso o modelo de embeddings seja trocado.
        pre_delete_collection=True,
    )
    print("Ingestão concluída com sucesso!")


if __name__ == "__main__":
    ingest_pdf()
