import os
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import DOCS_DIR

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: Optional[SentenceTransformer] = None
_index_cache: dict = {}


def _get_embed_model(model_name: str = _EMBED_MODEL_NAME) -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model '{model_name}'...")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _load_doc_index(doc_id: str):
    """Return (faiss_index, mapping) for *doc_id*, loading from disk only once."""
    if doc_id not in _index_cache:
        idx_path = os.path.join(DOCS_DIR, doc_id, "faiss_index.idx")
        map_path = os.path.join(DOCS_DIR, doc_id, "mapping.npy")
        _index_cache[doc_id] = (
            faiss.read_index(idx_path),
            np.load(map_path, allow_pickle=True).item(),
        )
    return _index_cache[doc_id]


def _invalidate_doc_cache(doc_id: str) -> None:
    """Remove *doc_id* from the in-memory cache (call after delete / update)."""
    _index_cache.pop(doc_id, None)


def create_embeddings(texts: List[str], model_name: str = _EMBED_MODEL_NAME) -> np.ndarray:
    model = _get_embed_model(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


def retrieve_similar_text(
    query: str,
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
    model_name: str = _EMBED_MODEL_NAME,
) -> List[str]:
    """
    Retrieve the top-k most similar chunks for *query*.

    Parameters
    ----------
    doc_ids : list[str] | None
        If a non-empty list, search only those documents' indices.
        If None or empty, search across ALL ingested documents.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)
    model = _get_embed_model(model_name)
    query_embedding = model.encode([query]).astype("float32")

    if doc_ids:
        search_ids = [
            doc_id for doc_id in doc_ids
            if os.path.isfile(os.path.join(DOCS_DIR, doc_id, "faiss_index.idx"))
        ]
    else:
        search_ids = [
            entry for entry in os.listdir(DOCS_DIR)
            if os.path.isfile(os.path.join(DOCS_DIR, entry, "faiss_index.idx"))
        ]
        if not search_ids:
            raise FileNotFoundError("No documents ingested yet. Use /api/upload_and_ingest first.")

    candidate_k = min(top_k * 3, 15)
    all_candidates: List[str] = []

    for doc_id in search_ids:
        idx_path = os.path.join(DOCS_DIR, doc_id, "faiss_index.idx")
        if not os.path.isfile(idx_path):
            continue
        index, mapping = _load_doc_index(doc_id)
        k = min(candidate_k, index.ntotal)
        _, indices = index.search(query_embedding, k)
        all_candidates.extend(mapping[i] for i in indices[0] if i in mapping)

    return all_candidates
