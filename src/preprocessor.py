"""
Preprocessor module — Stage 1 of the pipeline.

Responsibilities:
- Validate JSON structure
- Strip unnecessary control / special characters
- Normalize text (collapse whitespace, optionally lowercase)
- Attach request metadata (request_id, timestamp) for logging/tracing
"""

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


class Preprocessor:
    # Non-printable control characters (keep \t, \n, \r)
    _CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
    # Two or more horizontal spaces / tabs → single space
    _EXTRA_SPACES = re.compile(r"[ \t]{2,}")
    # Three or more consecutive newlines → double newline
    _EXTRA_NEWLINES = re.compile(r"\n{3,}")

    # ------------------------------------------------------------------ #
    # Text cleaning helpers
    # ------------------------------------------------------------------ #

    def clean_text(self, text: str) -> str:
        """Remove control chars and collapse redundant whitespace."""
        text = self._CTRL_CHARS.sub("", text)
        text = self._EXTRA_SPACES.sub(" ", text)
        text = self._EXTRA_NEWLINES.sub("\n\n", text)
        return text.strip()

    def normalize_query(self, text: str) -> str:
        """
        Normalize a short user query:
        clean + convert to lowercase for prompt consistency.
        """
        return self.clean_text(text).lower()

    # ------------------------------------------------------------------ #
    # Conversation preprocessing
    # ------------------------------------------------------------------ #

    def preprocess_conversation(
        self,
        conversation: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Sanitize every message in *conversation* and generate request metadata.

        Returns
        -------
        cleaned_conversation : list of {role, content} dicts
        metadata             : {request_id, timestamp, message_count}
        """
        metadata: Dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": len(conversation),
        }

        cleaned: List[Dict[str, str]] = []
        for msg in conversation:
            role = msg.get("role", "").strip().lower()
            content = self.clean_text(msg.get("content", ""))
            cleaned.append({"role": role, "content": content})

        return cleaned, metadata

    # ------------------------------------------------------------------ #
    # JSON validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_json(raw: str) -> Any:
        """
        Parse *raw* as JSON.
        Raises ValueError with a descriptive message on failure.
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format: {exc}") from exc
