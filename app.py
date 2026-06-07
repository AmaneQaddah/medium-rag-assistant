from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    CHUNK_SIZE,
    OVERLAP_RATIO,
    TOP_K,
    validate_config_for_api,
)


SYSTEM_PROMPT = """
You are a Medium-article assistant that answers questions strictly and only
based on the Medium articles dataset context provided to you (metadata
and article passages). You must not use any external knowledge, the open
internet, or information that is not explicitly contained in the retrieved
context. If the answer cannot be determined from the provided context,
respond: “I don’t know based on the provided Medium articles data.”
Always explain your answer using the given context, quoting or
paraphrasing the relevant article passage or metadata when helpful.

Additional rules:
- If the user asks for titles only, return only titles.
- If the user asks for exactly 3 articles, return exactly 3 distinct article titles if the context supports it.
- Do not invent authors, titles, URLs, timestamps, or article details.
- Prefer using article metadata when the question asks for title, author, tags, URL, or date.
""".strip()


app = FastAPI(title="Medium Article RAG Assistant")


class PromptRequest(BaseModel):
    question: str


def get_openai_client() -> OpenAI:
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)


def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def embed_query(client: OpenAI, question: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    return response.data[0].embedding


def normalize_pinecone_matches(query_result) -> List[Any]:
    if hasattr(query_result, "matches"):
        return query_result.matches

    if isinstance(query_result, dict):
        return query_result.get("matches", [])

    return []


def get_match_metadata(match) -> Dict[str, Any]:
    if hasattr(match, "metadata"):
        return match.metadata or {}

    if isinstance(match, dict):
        return match.get("metadata", {}) or {}

    return {}


def get_match_score(match) -> float:
    if hasattr(match, "score"):
        return float(match.score)

    if isinstance(match, dict):
        return float(match.get("score", 0.0))

    return 0.0


def retrieve_context(question: str) -> List[Dict[str, Any]]:
    client = get_openai_client()
    index = get_pinecone_index()

    query_embedding = embed_query(client, question)

    query_result = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )

    matches = normalize_pinecone_matches(query_result)

    context = []
    seen_articles = set()

    for match in matches:
        metadata = get_match_metadata(match)

        article_id = str(metadata.get("article_id", ""))

        # Helps avoid returning many chunks from the same article.
        if article_id in seen_articles:
            continue

        seen_articles.add(article_id)

        context.append(
            {
                "article_id": article_id,
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", ""),
                "url": metadata.get("url", ""),
                "timestamp": metadata.get("timestamp", ""),
                "tags": metadata.get("tags", ""),
                "chunk": metadata.get("chunk", ""),
                "score": get_match_score(match),
            }
        )

    return context


def build_user_prompt(question: str, context: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for i, item in enumerate(context, start=1):
        block = f"""
[Context {i}]
Article ID: {item.get("article_id", "")}
Title: {item.get("title", "")}
Authors: {item.get("authors", "")}
URL: {item.get("url", "")}
Timestamp: {item.get("timestamp", "")}
Tags: {item.get("tags", "")}
Passage:
{item.get("chunk", "")}
""".strip()

        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    return f"""
Retrieved Medium articles context:
{context_text}

User question:
{question}

Answer strictly based on the retrieved context.
""".strip()


def generate_answer(question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = get_openai_client()

    user_prompt = build_user_prompt(question, context)

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    response_text = completion.choices[0].message.content

    return {
        "response": response_text,
        "context": [
            {
                "article_id": item["article_id"],
                "title": item["title"],
                "chunk": item["chunk"],
                "score": item["score"],
            }
            for item in context
        ],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    try:
        validate_config_for_api()

        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        context = retrieve_context(question)

        if not context:
            return {
                "response": "I don’t know based on the provided Medium articles data.",
                "context": [],
                "Augmented_prompt": {
                    "System": SYSTEM_PROMPT,
                    "User": build_user_prompt(question, []),
                },
            }

        return generate_answer(question, context)

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }


@app.get("/")
def root():
    return {
        "message": "Medium Article RAG Assistant is running.",
        "endpoints": [
            "POST /api/prompt",
            "GET /api/stats",
        ],
    }