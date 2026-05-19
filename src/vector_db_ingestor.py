import os
import re
import json
import shutil
import uuid
import PyPDF2
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Singleton embedding model — loaded once, reused for all calls
# ---------------------------------------------------------------------------
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model: Optional[SentenceTransformer] = None

def _get_embed_model(model_name: str = _EMBED_MODEL_NAME) -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model '{model_name}'…")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model

# ---------------------------------------------------------------------------
# Sentence-boundary regex for chunking
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# ---------------------------------------------------------------------------
# Document store — each document lives in its own sub-directory
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR  = os.path.join(_BASE_DIR, "documents")

def _doc_dir(doc_id: str) -> str:
    return os.path.join(DOCS_DIR, doc_id)

def _ensure_docs_dir() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)

def _require_doc(doc_id: str) -> str:
    """Return the document directory or raise FileNotFoundError."""
    d = _doc_dir(doc_id)
    if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "metadata.json")):
        raise FileNotFoundError(f"Document '{doc_id}' not found.")
    return d


# ---------------------------------------------------------------------------
# Public document-store API
# ---------------------------------------------------------------------------

def list_documents() -> List[Dict[str, Any]]:
    """Return metadata for all stored documents, sorted newest-first."""
    _ensure_docs_dir()
    docs = []
    for entry in os.listdir(DOCS_DIR):
        meta_path = os.path.join(DOCS_DIR, entry, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                docs.append(json.load(f))
    return sorted(docs, key=lambda d: d.get("uploaded_at", ""), reverse=True)


def get_document(doc_id: str) -> Dict[str, Any]:
    """
    Return full document detail: metadata + pages (list of str) + chunks (list of str).
    """
    d = _require_doc(doc_id)
    with open(os.path.join(d, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(d, "pages.json"), "r", encoding="utf-8") as f:
        pages = json.load(f)
    with open(os.path.join(d, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {"metadata": metadata, "pages": pages, "chunks": chunks}


def delete_document(doc_id: str) -> None:
    """Permanently delete all files associated with a document."""
    d = _require_doc(doc_id)
    shutil.rmtree(d)


def update_document_chunks(doc_id: str, updated_chunks: List[str]) -> None:
    """
    Replace a document's chunks with *updated_chunks*, re-embed them,
    rebuild the FAISS index, and persist everything to disk.
    """
    d = _require_doc(doc_id)

    embeddings = create_embeddings(updated_chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = create_faiss_index(embeddings)
    mapping = {i: chunk for i, chunk in enumerate(updated_chunks)}

    faiss.write_index(index, os.path.join(d, "faiss_index.idx"))
    np.save(os.path.join(d, "mapping.npy"), mapping)
    with open(os.path.join(d, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    # Update metadata
    with open(os.path.join(d, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["num_chunks"]    = len(updated_chunks)
    metadata["last_edited_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Text extraction & chunking
# ---------------------------------------------------------------------------

def extract_pages_from_pdf(pdf_source: Union[str, bytes]) -> List[str]:
    """
    Extracts text from each page in the PDF file.
    The source can be a file path (str) or file content (bytes).
    """
    pages = []
    try:
        if isinstance(pdf_source, str):
            file_stream = open(pdf_source, "rb")
        elif isinstance(pdf_source, bytes):
            file_stream = io.BytesIO(pdf_source)
        else:
            raise TypeError("PDF source must be a file path (str) or bytes.")

        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            page_text = page.extract_text()
            pages.append(page_text if page_text else "")
    finally:
        if 'file_stream' in locals() and hasattr(file_stream, 'close'):
            file_stream.close()

    return pages


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150) -> List[str]:
    """
    Split *text* into overlapping chunks aligned to sentence boundaries.
    """
    sentences = _SENTENCE_END.split(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len: int = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        s_len = len(sentence) + 1

        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))
            overlap_buf: List[str] = []
            overlap_len: int = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_buf.insert(0, s)
                overlap_len += len(s) + 1
            current = overlap_buf
            current_len = overlap_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Embedding & indexing
# ---------------------------------------------------------------------------

def create_embeddings(texts: List[str], model_name: str = _EMBED_MODEL_NAME) -> np.ndarray:
    model = _get_embed_model(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=32)


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_pdf_to_vector_db(
    pdf_source: Union[str, bytes],
    filename: str = "document",
) -> str:
    """
    Full ingestion pipeline:
    1. Extract text page-by-page.
    2. Chunk the full text with sentence-aware overlap.
    3. Embed chunks and build FAISS index.
    4. Persist everything under DOCS_DIR/{doc_id}/.

    Returns
    -------
    doc_id : str  — unique identifier for the ingested document.
    """
    print(f"Extracting pages from '{filename}'...")
    pages = extract_pages_from_pdf(pdf_source)
    full_text = "\n\n".join(p for p in pages if p.strip())
    if not full_text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    print("Chunking document text...")
    chunks = chunk_text(full_text)
    print(f"  → {len(chunks)} chunks created (chunk_size=600, overlap=150).")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = create_faiss_index(embeddings)
    mapping = {i: chunk for i, chunk in enumerate(chunks)}

    # Persist
    doc_id = str(uuid.uuid4())
    d = _doc_dir(doc_id)
    os.makedirs(d, exist_ok=True)

    faiss.write_index(index, os.path.join(d, "faiss_index.idx"))
    np.save(os.path.join(d, "mapping.npy"), mapping)

    # Save original PDF bytes for later viewing
    if isinstance(pdf_source, bytes):
        with open(os.path.join(d, "original.pdf"), "wb") as f:
            f.write(pdf_source)
    elif isinstance(pdf_source, str) and os.path.isfile(pdf_source):
        import shutil as _shutil
        _shutil.copy2(pdf_source, os.path.join(d, "original.pdf"))

    with open(os.path.join(d, "pages.json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    with open(os.path.join(d, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    metadata = {
        "doc_id":       doc_id,
        "filename":     filename,
        "uploaded_at":  datetime.now(timezone.utc).isoformat(),
        "num_pages":    len(pages),
        "num_chunks":   len(chunks),
        "last_edited_at": None,
    }
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Ingestion of '{filename}' completed — doc_id={doc_id}")
    return doc_id


def get_document_pdf_path(doc_id: str) -> str:
    """Return the path to the original PDF file, or raise FileNotFoundError."""
    d = _require_doc(doc_id)
    pdf_path = os.path.join(d, "original.pdf")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Original PDF for document '{doc_id}' is not available.")
    return pdf_path


def run_ingestion(pdf_source: Union[str, bytes], filename: str = "document") -> str:
    """Entry point: runs the full PDF ingestion and returns doc_id."""
    print(f"Starting ingestion process for '{filename}'...")
    return ingest_pdf_to_vector_db(pdf_source, filename)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

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
    _ensure_docs_dir()
    model = _get_embed_model(model_name)
    query_embedding = model.encode([query]).astype("float32")

    if doc_ids:
        search_ids = [
            d for d in doc_ids
            if os.path.isfile(os.path.join(DOCS_DIR, d, "faiss_index.idx"))
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

    for did in search_ids:
        idx_path = os.path.join(DOCS_DIR, did, "faiss_index.idx")
        map_path = os.path.join(DOCS_DIR, did, "mapping.npy")
        if not os.path.isfile(idx_path):
            continue
        index   = faiss.read_index(idx_path)
        mapping = np.load(map_path, allow_pickle=True).item()
        k = min(candidate_k, index.ntotal)
        _, indices = index.search(query_embedding, k)
        all_candidates.extend(mapping[i] for i in indices[0] if i in mapping)

    return all_candidates


if __name__ == "__main__":
    if os.path.exists("sample.pdf"):
        run_ingestion("sample.pdf", "sample.pdf")
    else:
        print("Place a 'sample.pdf' file in the 'src' directory to run a test ingestion.")

