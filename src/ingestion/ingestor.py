import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Union

import faiss
import numpy as np

from src.chunking.chunker import chunk_text
from src.embedding.vector_store import create_embeddings, create_faiss_index, _invalidate_doc_cache
from src.ingestion.file_reader import extract_pages_from_file
from src.utils import DOCS_DIR, EXT_TO_MIME, MIME_TO_EXT, SUPPORTED_EXTENSIONS


def _doc_dir(doc_id: str) -> str:
    return os.path.join(DOCS_DIR, doc_id)


def _ensure_docs_dir() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)


def _require_doc(doc_id: str) -> str:
    """Return the document directory or raise FileNotFoundError."""
    doc_dir = _doc_dir(doc_id)
    if not os.path.isdir(doc_dir) or not os.path.isfile(os.path.join(doc_dir, "metadata.json")):
        raise FileNotFoundError(f"Document '{doc_id}' not found.")
    return doc_dir


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
    doc_dir = _require_doc(doc_id)
    with open(os.path.join(doc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(os.path.join(doc_dir, "pages.json"), "r", encoding="utf-8") as f:
        pages = json.load(f)
    with open(os.path.join(doc_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {"metadata": metadata, "pages": pages, "chunks": chunks}


def delete_document(doc_id: str) -> None:
    """Permanently delete all files associated with a document."""
    doc_dir = _require_doc(doc_id)
    shutil.rmtree(doc_dir)
    _invalidate_doc_cache(doc_id)


def update_document_chunks(doc_id: str, updated_chunks: List[str]) -> None:
    """
    Replace a document's chunks with *updated_chunks*, re-embed them,
    rebuild the FAISS index, and persist everything to disk.
    """
    doc_dir = _require_doc(doc_id)

    embeddings = create_embeddings(updated_chunks)
    embeddings = np.array(embeddings).astype("float32")
    index = create_faiss_index(embeddings)
    mapping = {i: chunk for i, chunk in enumerate(updated_chunks)}

    faiss.write_index(index, os.path.join(doc_dir, "faiss_index.idx"))
    np.save(os.path.join(doc_dir, "mapping.npy"), mapping)
    _invalidate_doc_cache(doc_id)
    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(updated_chunks, f, ensure_ascii=False, indent=2)

    with open(os.path.join(doc_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["num_chunks"] = len(updated_chunks)
    metadata["last_edited_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def ingest_document_to_vector_db(
    file_source: Union[str, bytes],
    filename: str = "document",
    content_type: str = "",
) -> str:
    """
    Full ingestion pipeline for any supported document format:
    1. Extract text page-by-page / section-by-section.
    2. Chunk the full text with sentence-aware overlap.
    3. Embed chunks and build FAISS index.
    4. Persist everything under DOCS_DIR/{doc_id}/.

    Returns
    -------
    doc_id : str  - unique identifier for the ingested document.
    """
    ext = os.path.splitext(filename.lower())[1]
    if not ext and content_type:
        ext = MIME_TO_EXT.get(content_type, "")

    print(f"Extracting pages from '{filename}' (type={ext or content_type})...")
    pages = extract_pages_from_file(file_source, filename=filename, content_type=content_type)
    full_text = "\n\n".join(page for page in pages if page.strip())
    if not full_text.strip():
        raise ValueError("No text could be extracted from the document.")

    print("Chunking document text...")
    chunks = chunk_text(full_text)
    print(f"  -> {len(chunks)} chunks created (chunk_size=600, overlap=150).")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = create_faiss_index(embeddings)
    mapping = {i: chunk for i, chunk in enumerate(chunks)}

    doc_id = str(uuid.uuid4())
    doc_dir = _doc_dir(doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(doc_dir, "faiss_index.idx"))
    np.save(os.path.join(doc_dir, "mapping.npy"), mapping)

    original_ext = ext if ext in SUPPORTED_EXTENSIONS else ".bin"
    original_filename = f"original{original_ext}"
    if isinstance(file_source, bytes):
        with open(os.path.join(doc_dir, original_filename), "wb") as f:
            f.write(file_source)
    elif isinstance(file_source, str) and os.path.isfile(file_source):
        shutil.copy2(file_source, os.path.join(doc_dir, original_filename))

    with open(os.path.join(doc_dir, "pages.json"), "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    with open(os.path.join(doc_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    file_type = ext.lstrip(".") if ext else "unknown"
    metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "last_edited_at": None,
    }
    with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Ingestion of '{filename}' completed - doc_id={doc_id}")
    return doc_id


def ingest_pdf_to_vector_db(pdf_source: Union[str, bytes], filename: str = "document") -> str:
    return ingest_document_to_vector_db(pdf_source, filename=filename)


def run_ingestion(
    file_source: Union[str, bytes],
    filename: str = "document",
    content_type: str = "",
) -> str:
    """Entry point: runs the full document ingestion and returns doc_id."""
    print(f"Starting ingestion process for '{filename}'...")
    return ingest_document_to_vector_db(file_source, filename=filename, content_type=content_type)


def get_document_file_path(doc_id: str) -> tuple:
    """
    Return (file_path, mime_type) for the original uploaded file, or raise FileNotFoundError.
    Searches for any 'original.*' file in the document directory.
    """
    doc_dir = _require_doc(doc_id)
    for ext, mime in EXT_TO_MIME.items():
        candidate = os.path.join(doc_dir, f"original{ext}")
        if os.path.isfile(candidate):
            return candidate, mime
    raise FileNotFoundError(f"Original file for document '{doc_id}' is not available.")


def get_document_pdf_path(doc_id: str) -> str:
    """Return the path to the original file (any format), or raise FileNotFoundError."""
    path, _ = get_document_file_path(doc_id)
    return path
