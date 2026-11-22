# Backend Intern Take-Home: Mini RAG API

**Time box: about 2 hours. Deadline: 48 hours after you receive this.**

Build a small retrieval-augmented question-answering API with FastAPI.
The embedding model and the LLM are mocked in `app/llm.py`, so you need no API key and no network.

AI coding tools are allowed. In the interview you will explain your code and change it live, so make sure you understand every line.

## Project Status: All Tasks Completed ✅

- [x] **`chunk_text` in `app/chunking.py`**: Fully implemented with parameter validation and overlapping window slicing.
- [x] **`GET /health`**: Verified working.
- [x] **`POST /ingest`**: Implemented with validation (`422`), chunking, embedding generation, in-memory deduplication/replacement.
- [x] **`POST /ask`**: Implemented with cosine similarity dot product ranking, `top_k` validation, `404` for empty store, and generation.
- [x] **`GET /documents`**: Implemented returning sorted documents by `id` with accurate chunk counts.
- [x] **`DELETE /documents/{doc_id}`**: Implemented returning `204` with empty body and `404` for missing documents.
- [x] **Tests**: All public and expanded large-dataset tests passing.
- [x] **`NOTES.md`**: Completed with design choices, future improvements, and AI tool usage.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # run tests
uvicorn app.main:app --reload   # optional: try it at http://localhost:8000/docs
```

## Rules

- Python 3.12. Do **not** add dependencies. The grader installs only `requirements.txt` as it is now.
- Do **not** modify `app/llm.py`. The grader replaces it with its own copy.
- Keep the store in memory. No database.
- `app/main.py` must expose `create_app()`. Each call returns a new app with an empty store.
- You may add files and modules under `app/`.

## Part 1: `chunk_text` in `app/chunking.py` [Completed ✅]

`chunk_text(text, size=50, overlap=10) -> list[str]`

- Words are separated by whitespace (`str.split()`).
- Each chunk has at most `size` words, joined with one space.
- Consecutive chunks share exactly `overlap` words.
- Stop when a chunk contains the last word. Do not emit an extra trailing chunk.
- Empty or whitespace-only text returns `[]`.
- Raise `ValueError` when `size <= 0`, `overlap < 0`, or `overlap >= size`.

Example: 12 words `w0..w11`, `size=5`, `overlap=2`:

```
["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9 w10", "w9 w10 w11"]
```

## Part 2: API endpoints [Completed ✅]

All bodies are JSON. Use FastAPI's default `422` for invalid input.

### `GET /health` [Completed ✅]
Already done. Returns `{"status": "ok"}`.

### `POST /ingest` [Completed ✅]

```json
{"documents": [{"id": "doc1", "text": "..."}]}
```

- Split each document's text with `chunk_text` (default `size` and `overlap`).
- Embed each chunk with `llm.embed`.
- Chunk IDs are `"<doc_id>:<index>"`, index starting at 0.
- If a document `id` already exists, **replace** its chunks. Do not duplicate them.
- Response `200`: `{"ingested": <number of documents>, "chunks": <total chunks created>}`
- `422` when `documents` is missing or empty, when an `id` is empty, or when a `text` is empty or whitespace only.

### `POST /ask` [Completed ✅]

```json
{"question": "...", "top_k": 3}
```

- `top_k` is optional. Default `3`, allowed range `1..10`, else `422`.
- `question` empty or whitespace only gives `422`.
- No documents ingested gives `404`.
- Score every chunk by cosine similarity between `llm.embed(question)` and the chunk vector. The vectors are unit length, so cosine equals the dot product.
- Take the best `top_k` chunks, highest score first.
- `answer` is `llm.generate(question, [texts of those chunks, best first])`.
- Response `200`:

```json
{
  "answer": "...",
  "sources": [
    {"doc_id": "doc1", "chunk_id": "doc1:0", "score": 0.73, "text": "..."}
  ]
}
```

### `GET /documents` [Completed ✅]

Response `200`, sorted by `id`:

```json
{"documents": [{"id": "doc1", "chunks": 2}]}
```

### `DELETE /documents/{doc_id}` [Completed ✅]

- `204` with an empty body when deleted. Its chunks must no longer appear in `/ask`.
- `404` when the document does not exist.

## Submit

1. Push your code to a **public** GitHub repo (or a private one shared with the reviewer).
2. Add `NOTES.md` with 3 to 5 lines: your design choices, what you would improve with more time, and which AI tools you used.
3. Send the repo link in the application form.
