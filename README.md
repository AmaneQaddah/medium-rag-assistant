# Medium Article RAG API

This project implements a Retrieval-Augmented Generation (RAG) system over a Medium articles dataset.  
The system answers questions strictly based on retrieved Medium article chunks and metadata, without relying on external knowledge.

## Live Deployment

Base URL: `https://YOUR-VERCEL-URL.vercel.app`

Swagger Docs: `https://YOUR-VERCEL-URL.vercel.app/docs`

## Main Endpoint

### POST `/api/prompt`

Send a natural language question to query the system.

### Request Body

```json
{
  "question": "Your question here"
}
