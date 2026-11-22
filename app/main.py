from fastapi import FastAPI, HTTPException, Response, status

from app import llm
from app.chunking import chunk_text
from app.models import IngestRequest, AskRequest



def create_app() -> FastAPI:
    """Return a NEW app with an EMPTY in-memory store.

    The tests call create_app() once per test, so state must not leak
    between app instances (no module-level store).
    """
    app = FastAPI()

    # In-memory document store scoped to this app instance:
    # doc_id -> list of chunk dicts: {"chunk_id": str, "text": str, "embedding": list[float]}
    store: dict[str, list[dict]] = {}
    app.state.store = store

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest")
    def ingest(payload: IngestRequest):
        total_chunks = 0
        for doc in payload.documents:
            chunks = chunk_text(doc.text)
            chunk_records = []
            for idx, chunk_str in enumerate(chunks):
                chunk_id = f"{doc.id}:{idx}"
                embedding = llm.embed(chunk_str)
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "embedding": embedding,
                })
            # If doc.id exists, replace its chunks
            store[doc.id] = chunk_records
            total_chunks += len(chunk_records)

        return {"ingested": len(payload.documents), "chunks": total_chunks}

    @app.post("/ask")
    def ask(payload: AskRequest):
        if not store:
            raise HTTPException(status_code=404, detail="No documents ingested")

        query_vec = llm.embed(payload.question)

        all_scored_chunks = []
        for doc_id, chunk_records in store.items():
            for chunk in chunk_records:
                # Vectors are unit length, so cosine similarity == dot product
                score = sum(q * c for q, c in zip(query_vec, chunk["embedding"]))
                all_scored_chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk["chunk_id"],
                    "score": round(score, 4),
                    "text": chunk["text"],
                })

        # Sort highest score first
        all_scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = all_scored_chunks[:payload.top_k]

        contexts = [c["text"] for c in top_chunks]
        answer = llm.generate(payload.question, contexts)

        return {
            "answer": answer,
            "sources": top_chunks,
        }
    @app.get("/documents")
    def get_documents():
        sorted_ids = sorted(store.keys())
        documents = [{"id": doc_id, "chunks": len(store[doc_id])} for doc_id in sorted_ids]
        return {"documents": documents}

    @app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(doc_id: str):
        if doc_id not in store:
            raise HTTPException(status_code=404, detail="Document not found")
        del store[doc_id]
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
