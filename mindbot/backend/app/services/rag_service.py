import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
from openai import OpenAI

from app.core.config import settings, FAISS_DIR

client = OpenAI(api_key=settings.openai_api_key)

INDEX_FILE = FAISS_DIR / "mindbot.index"
META_FILE = FAISS_DIR / "mindbot_meta.json"


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def embed_texts(texts: List[str]) -> np.ndarray:
    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key missing")

    resp = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    vectors = np.array([item.embedding for item in resp.data], dtype=np.float32)
    return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


@dataclass
class ChunkRecord:
    text: str
    source: str
    chunk_id: int


class FAISSStore:
    def __init__(self, index=None, records=None):
        self.index = index
        self.records: List[ChunkRecord] = records or []

    @classmethod
    def load(cls):
        if INDEX_FILE.exists() and META_FILE.exists():
            index = faiss.read_index(str(INDEX_FILE))
            records_data = json.loads(META_FILE.read_text(encoding="utf-8"))
            records = [ChunkRecord(**r) for r in records_data]
            return cls(index=index, records=records)

        return cls(index=None, records=[])

    def save(self):
        if self.index is None:
            return
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        META_FILE.write_text(
            json.dumps([r.__dict__ for r in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_chunks(self, texts: List[str], source: str):
        if not texts:
            return

        vectors = embed_texts(texts)
        dim = vectors.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        start_id = len(self.records)

        for i, t in enumerate(texts):
            self.records.append(ChunkRecord(text=t, source=source, chunk_id=start_id + i))

        self.save()

    def search(self, query: str, top_k: int = 4) -> List[Tuple[str, str, float]]:
        if self.index is None or len(self.records) == 0:
            return []

        qvec = embed_query(query).reshape(1, -1)
        scores, idxs = self.index.search(qvec, top_k)
        results = []

        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if idx < len(self.records):
                rec = self.records[idx]
                results.append((rec.text, rec.source, float(score)))
        return results


_store = None


def get_store() -> FAISSStore:
    global _store
    if _store is None:
        _store = FAISSStore.load()
    return _store


def add_document_chunks(chunks: List[str], source: str):
    store = get_store()
    store.add_chunks(chunks, source)


def retrieve_context(query: str, top_k: int = None) -> str:
    store = get_store()
    top_k = top_k or settings.top_k_docs
    results = store.search(query, top_k=top_k)
    if not results:
        return ""

    formatted = []
    for i, (text, source, score) in enumerate(results, start=1):
        formatted.append(f"[{i}] Source: {source}\n{text}")
    return "\n\n".join(formatted)