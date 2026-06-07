import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medium-articles")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "4UHRUIN-gpt-5-mini")

EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.2"))
TOP_K = int(os.getenv("TOP_K", "10"))

CSV_PATH = os.getenv("CSV_PATH", "medium-english-50mb.csv")
INGEST_MAX_ARTICLES = os.getenv("INGEST_MAX_ARTICLES")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

PINECONE_CREATE_INDEX_IF_MISSING = (
    os.getenv("PINECONE_CREATE_INDEX_IF_MISSING", "false").lower() == "true"
)
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def validate_config_for_api():
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY")


def validate_config_for_ingest():
    validate_config_for_api()