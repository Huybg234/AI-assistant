"""
Utility helpers shared across the application.
"""

import os
import logging
from typing import Any, Dict

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".rtf"}
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "application/rtf",
    "text/rtf",
}
MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
    "application/rtf": ".rtf",
    "text/rtf": ".rtf",
}
EXT_TO_MIME = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".rtf": "application/rtf",
}
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(_BASE_DIR, "documents")

logger = logging.getLogger(__name__)


def format_output(thought_chain: list) -> str:
    """Join a list of thought-chain strings into a single formatted block."""
    return "\n".join(thought_chain)


def handle_data_transformation(data: str) -> str:
    """Strip leading/trailing whitespace from a raw string."""
    return data.strip()


def log_request(metadata: Dict[str, Any], endpoint: str) -> None:
    """
    Emit a standardised INFO log line for an incoming request.

    Parameters
    ----------
    metadata : dict from Preprocessor.preprocess_conversation
    endpoint : the API endpoint name (e.g. "/api/qa")
    """
    logger.info(
        f"[{metadata.get('request_id', 'N/A')}] "
        f"endpoint={endpoint} "
        f"ts={metadata.get('timestamp', 'N/A')} "
        f"messages={metadata.get('message_count', 0)}"
    )
