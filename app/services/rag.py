from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
VECTOR_DIR = Path(__file__).resolve().parents[2] / "vector_store"
COLLECTION_NAME = "cloudsense_aws_knowledge"
HF_EMBEDDING_MODEL = os.getenv(
    "CLOUDSENSE_HF_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

import chromadb
from sentence_transformers import SentenceTransformer

_encoder = None
_collection = None


def load_documents() -> List[Dict[str, str]]:
    documents = []
    if not KB_DIR.exists():
        raise RuntimeError(f"Knowledge base directory not found: {KB_DIR}")
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            documents.append({"source": path.name, "text": text})
    if not documents:
        raise RuntimeError("Knowledge base is empty")
    return documents


def chunk_text(text: str, chunk_size: int = 450) -> List[str]:
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current, current_len = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(HF_EMBEDDING_MODEL)
    return _encoder


def _get_collection():
    global _collection
    if _collection is None:
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": HF_EMBEDDING_MODEL,
            },
        )
    return _collection


def build_index(force: bool = False) -> Dict[str, int]:
    collection = _get_collection()
    docs = load_documents()
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            chunks.append((doc["source"], i, chunk))

    if force and collection.count():
        existing = collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)

    if collection.count() == 0 or force:
        encoder = _get_encoder()
        texts = [x[2] for x in chunks]
        embeddings = encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        if texts:
            ids = [f"{source}::{idx}" for source, idx, _ in chunks]
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {"source": source, "chunk_id": idx}
                    for source, idx, _ in chunks
                ],
            )

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "indexed_vectors": collection.count(),
    }


def ensure_index() -> Dict[str, int]:
    collection = _get_collection()
    docs = load_documents()
    if collection.count() == 0:
        return build_index()

    expected_chunks = sum(len(chunk_text(d["text"])) for d in docs)
    if collection.count() != expected_chunks:
        return build_index(force=True)

    return {
        "documents": len(docs),
        "chunks": expected_chunks,
        "indexed_vectors": collection.count(),
    }


def retrieve(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    question = (question or "").strip()
    if not question:
        raise ValueError("question cannot be empty")

    stats = ensure_index()
    if stats["indexed_vectors"] == 0:
        raise RuntimeError("Vector database contains no indexed knowledge")

    collection = _get_collection()
    encoder = _get_encoder()
    query_embedding = encoder.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(1, min(top_k, 10)),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    output = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append({
            "source": metadata.get("source", "unknown"),
            "chunk_id": metadata.get("chunk_id", 0),
            "text": text,
            "score": round(similarity, 4),
            "retrieval": "huggingface_embedding+chromadb",
        })

    if not output:
        raise RuntimeError("Vector database returned no evidence")
    return output


def format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        raise RuntimeError("No RAG evidence was retrieved")
    return "\n\n".join(
        f"[Evidence ID: {c['source']}#{c['chunk_id']} | Similarity: {c.get('score', 0)}]\n{c['text']}"
        for c in chunks
    )


def retrieval_status() -> Dict[str, Any]:
    stats = ensure_index()
    return {
        "vector_database": "ChromaDB PersistentClient",
        "collection": COLLECTION_NAME,
        "embedding_provider": "Hugging Face / Sentence Transformers",
        "embedding_model": HF_EMBEDDING_MODEL,
        "document_count": stats["documents"],
        "chunk_count": stats["chunks"],
        "indexed_vectors": stats["indexed_vectors"],
        "persistent_path": str(VECTOR_DIR),
        "semantic_search": True,
    }
