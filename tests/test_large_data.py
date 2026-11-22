"""Tests with larger datasets and comprehensive edge case validations."""

import pytest
from app.chunking import chunk_text
from app.main import create_app, AskRequest, IngestRequest
from app.models import DocumentItem
from fastapi import HTTPException


@pytest.fixture
def app_instance():
    return create_app()


def get_endpoint(app, path: str, method: str = "GET"):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", []):
            return route.endpoint
    raise ValueError(f"Route {method} {path} not found")


def test_chunk_text_large_data():
    # 500 words with default size=50, overlap=10 (step=40)
    # Expected chunks:
    # 0: 0..50
    # 1: 40..90
    # 2: 80..130
    # ...
    # 11: 440..490
    # 12: 480..500 (20 words)
    words = [f"token{i}" for i in range(500)]
    text = " ".join(words)
    chunks = chunk_text(text)

    assert len(chunks) == 13
    assert chunks[0].split()[0] == "token0"
    assert chunks[0].split()[-1] == "token49"
    assert chunks[1].split()[0] == "token40"
    assert chunks[-1].split()[-1] == "token499"

    # Consecutive chunks must share exactly overlap=10 words
    for i in range(len(chunks) - 1):
        c1_words = chunks[i].split()
        c2_words = chunks[i + 1].split()
        overlap_words_c1 = c1_words[-10:]
        overlap_words_c2 = c2_words[:10]
        assert overlap_words_c1 == overlap_words_c2


def test_chunk_text_exact_boundary():
    # 130 words with size=50, overlap=10 (step=40):
    # Chunk 0: 0..50
    # Chunk 1: 40..90
    # Chunk 2: 80..130 -> reaches last word exactly
    # Total chunks should be 3, no trailing chunk
    words = [f"word{i}" for i in range(130)]
    chunks = chunk_text(" ".join(words), size=50, overlap=10)
    assert len(chunks) == 3
    assert chunks[-1].split()[-1] == "word129"


def test_large_corpus_ingest_and_retrieval(app_instance):
    ingest = get_endpoint(app_instance, "/ingest", "POST")
    ask = get_endpoint(app_instance, "/ask", "POST")
    get_docs = get_endpoint(app_instance, "/documents", "GET")
    delete_doc = get_endpoint(app_instance, "/documents/{doc_id}", "DELETE")

    # Ingest 50 documents, each with around 150 words (each produces ~4 chunks -> ~200 chunks total)
    docs = []
    for doc_idx in range(50):
        if doc_idx == 27:
            # Inject a distinct target topic in document 27
            doc_words = ["quantum", "teleportation", "qubits", "entanglement"] * 40
        elif doc_idx == 42:
            # Inject another distinct target topic
            doc_words = ["astronomy", "telescope", "galaxy", "supernova"] * 40
        else:
            doc_words = [f"filler{doc_idx}_{w}" for w in range(150)]

        docs.append(DocumentItem(id=f"doc_{doc_idx:03d}", text=" ".join(doc_words)))

    payload = IngestRequest(documents=docs)
    ingest_res = ingest(payload)

    assert ingest_res["ingested"] == 50
    assert ingest_res["chunks"] > 180

    # Verify /documents returns all 50 sorted by id
    docs_res = get_docs()
    assert len(docs_res["documents"]) == 50
    ids = [d["id"] for d in docs_res["documents"]]
    assert ids == sorted(ids)
    assert ids[0] == "doc_000"
    assert ids[-1] == "doc_049"

    # Query for the target topic: document 27 should be the top hit
    ask_res = ask(AskRequest(question="Tell me about quantum teleportation and entanglement", top_k=5))
    assert len(ask_res["sources"]) == 5
    top_hit = ask_res["sources"][0]
    assert top_hit["doc_id"] == "doc_027"
    assert top_hit["score"] > 0.5
    assert "quantum" in top_hit["text"]

    # Delete doc_027
    del_res = delete_doc("doc_027")
    assert del_res.status_code == 204

    # Document 27 must no longer exist in /documents
    docs_after_delete = get_docs()["documents"]
    assert len(docs_after_delete) == 49
    assert "doc_027" not in [d["id"] for d in docs_after_delete]

    # Repeating the query must no longer return doc_027
    ask_after_delete = ask(AskRequest(question="Tell me about quantum teleportation and entanglement", top_k=5))
    returned_doc_ids = [s["doc_id"] for s in ask_after_delete["sources"]]
    assert "doc_027" not in returned_doc_ids


def test_replace_large_document(app_instance):
    ingest = get_endpoint(app_instance, "/ingest", "POST")
    get_docs = get_endpoint(app_instance, "/documents", "GET")

    # Ingest a doc with 300 words (~8 chunks)
    long_text = " ".join(f"word{i}" for i in range(300))
    ingest(IngestRequest(documents=[DocumentItem(id="doc_large", text=long_text)]))

    docs_info = get_docs()["documents"]
    assert len(docs_info) == 1
    initial_chunks = docs_info[0]["chunks"]
    assert initial_chunks >= 7

    # Replace with a short text (1 chunk)
    ingest(IngestRequest(documents=[DocumentItem(id="doc_large", text="Short replacement text.")]))

    docs_info_after = get_docs()["documents"]
    assert len(docs_info_after) == 1
    assert docs_info_after[0]["chunks"] == 1
