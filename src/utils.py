"""
Utility helpers shared across the application.
"""

import logging
from typing import Any, Dict

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
