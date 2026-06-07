
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    OVERLAP_RATIO,
    CSV_PATH,
    INGEST_MAX_ARTICLES,
    EMBEDDING_BATCH_SIZE,
    PINECONE_CREATE_INDEX_IF_MISSING,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_INDEX_HOST,
    validate_config_for_ingest,
)


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def split_text_to_chunks(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Approximate token chunking by words.
    In this project, chunk_size=512 means about 512 words per chunk.
    This keeps chunks safely below the assignment maximum of 1024 tokens in most cases.
    """
    words = text.split()

    if not words:
        return []

    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("Invalid chunk settings: step must be positive.")

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks

def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    return index

def get_openai_client() -> OpenAI:
    if OPENAI_BASE_URL:
        return OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=120.0,
            max_retries=5,
        )

    return OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=120.0,
        max_retries=5,
    )

def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def flush_batch(
    client: OpenAI,
    index,
    texts: List[str],
    payloads: List[Dict[str, Any]],
):
    if not texts:
        return

    embeddings = embed_texts(client, texts)

    vectors = []

    for embedding, payload in zip(embeddings, payloads):
        vectors.append(
            {
                "id": payload["vector_id"],
                "values": embedding,
                "metadata": payload["metadata"],
            }
        )

    index.upsert(
        vectors=vectors,
        namespace=PINECONE_NAMESPACE,
    )

    print(f"Upserted {len(vectors)} chunks to Pinecone.")


def main():
    validate_config_for_ingest()

    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    required_columns = ["title", "text", "url", "authors", "timestamp", "tags"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    if INGEST_MAX_ARTICLES:
        max_articles = int(INGEST_MAX_ARTICLES)
        df = df.head(max_articles)
        print(f"Using only first {max_articles} articles for testing.")

    client = get_openai_client()
    index = get_pinecone_index()

    pending_texts = []
    pending_payloads = []

    total_chunks = 0

    for article_id, row in df.iterrows():
        title = safe_str(row["title"])
        text = safe_str(row["text"])
        url = safe_str(row["url"])
        authors = safe_str(row["authors"])
        timestamp = safe_str(row["timestamp"])
        tags = safe_str(row["tags"])

        chunks = split_text_to_chunks(
            text=text,
            chunk_size=CHUNK_SIZE,
            overlap_ratio=OVERLAP_RATIO,
        )

        for chunk_id, chunk in enumerate(chunks):
            vector_id = f"article_{article_id}_chunk_{chunk_id}"

            metadata = {
                "article_id": str(article_id),
                "title": title,
                "authors": authors,
                "url": url,
                "timestamp": timestamp,
                "tags": tags,
                "chunk": chunk,
                "chunk_id": str(chunk_id),
            }

            pending_texts.append(chunk)
            pending_payloads.append(
                {
                    "vector_id": vector_id,
                    "metadata": metadata,
                }
            )

            total_chunks += 1

            if len(pending_texts) >= EMBEDDING_BATCH_SIZE:
                flush_batch(client, index, pending_texts, pending_payloads)
                pending_texts = []
                pending_payloads = []

    flush_batch(client, index, pending_texts, pending_payloads)

    print("Done.")
    print(f"Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    main()