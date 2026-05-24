"""
Postprocessor module — Stage 4 of the pipeline.

Responsibilities:
- Extract the main text content from the raw Gemini/LLM response object.
- Run a safety check (detect policy-violating or empty/blocked content).
- Truncate the response if it exceeds the configured character limit.
- Format the cleaned result into a standard internal dict ready for the HTTP layer.
"""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum number of characters allowed in a single response.
# Responses that exceed this limit are truncated with a notice appended.
MAX_RESPONSE_CHARS: int = 10_000

# Patterns that indicate potentially unsafe or policy-violating content.
# Gemini already enforces its own safety filters; these are an extra guardrail
# for patterns that could slip through in edge cases.
_UNSAFE_PATTERNS = [
    re.compile(r"\b(how to (make|build|create) (a )?(bomb|explosive|weapon))\b", re.IGNORECASE),
    re.compile(r"\b(step[s]? (to )?(hack|exploit|bypass))\b", re.IGNORECASE),
    re.compile(r"\b(personal(ly identifiable)?|credit card|social security)\s+number\b", re.IGNORECASE),
]

# Phrases Gemini inserts when it refuses a request due to safety policies.
_BLOCKED_PHRASES = [
    "i cannot fulfill this request",
    "i'm not able to help with that",
    "i can't assist with that",
    "this request violates",
    "i'm unable to provide",
]


class SafetyViolationError(ValueError):
    """Raised when postprocessed content fails the safety check."""


class Postprocessor:
    """
    Post-processes the raw LLM response through four sequential steps:
    1. Content extraction
    2. Safety check
    3. Truncation
    4. Standard formatting
    """

    def __init__(self, max_chars: int = MAX_RESPONSE_CHARS):
        self.max_chars = max_chars

    # ------------------------------------------------------------------ #
    # Step 1 — Content extraction
    # ------------------------------------------------------------------ #

    def extract_content(self, raw_response: Any) -> str:
        """
        Pull the plain-text answer out of a LangChain AIMessage (or any
        object with a `.content` attribute).  Falls back to str() if needed.
        """
        if hasattr(raw_response, "content") and isinstance(raw_response.content, str):
            return raw_response.content

        # Some wrappers nest content inside a list of dicts
        if hasattr(raw_response, "content") and isinstance(raw_response.content, list):
            parts = [
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_response.content
            ]
            return "".join(parts)

        return str(raw_response)

    # ------------------------------------------------------------------ #
    # Step 2 — Safety check
    # ------------------------------------------------------------------ #

    def safety_check(self, text: str) -> None:
        """
        Raise SafetyViolationError if the text:
        - is empty (Gemini silently blocked the request), or
        - contains a known refusal/block phrase, or
        - matches one of the unsafe regex patterns.
        """
        stripped = text.strip()

        if not stripped:
            raise SafetyViolationError(
                "Empty response received — the request may have been blocked by Gemini's safety filters."
            )

        lower = stripped.lower()
        for phrase in _BLOCKED_PHRASES:
            if phrase in lower:
                raise SafetyViolationError(
                    f"Response blocked by safety policy (matched phrase: '{phrase}')."
                )

        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(stripped):
                raise SafetyViolationError(
                    f"Response contains unsafe content (matched pattern: '{pattern.pattern}')."
                )

        logger.debug("Safety check passed.")

    # ------------------------------------------------------------------ #
    # Step 3 — Truncation
    # ------------------------------------------------------------------ #

    def truncate(self, text: str) -> str:
        """
        If *text* exceeds *self.max_chars*, truncate it and append a notice.
        The truncation boundary respects word boundaries when possible.
        """
        if len(text) <= self.max_chars:
            return text

        cutoff = text.rfind(" ", 0, self.max_chars)
        if cutoff == -1:
            cutoff = self.max_chars

        truncated = text[:cutoff]
        notice = f"\n\n[Response truncated at {self.max_chars:,} characters.]"
        logger.warning(
            f"Response truncated: original={len(text)} chars, limit={self.max_chars}."
        )
        return truncated + notice

    # ------------------------------------------------------------------ #
    # Step 4 — Standard formatting
    # ------------------------------------------------------------------ #

    def format_response(self, text: str) -> Dict[str, Any]:
        """
        Wrap the cleaned text in the standard internal response envelope.
        The HTTP layer will merge this with routing-specific keys and metadata.
        """
        return {
            "answer": text,
            "char_count": len(text),
            "truncated": len(text) < len(text),  # will be overridden by process()
        }

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def process(self, raw_response: Any) -> Dict[str, Any]:
        """
        Run the full four-step post-processing pipeline.

        Returns
        -------
        dict with keys:
            answer      – cleaned, safe, (possibly truncated) response text
            char_count  – final character count
            truncated   – True if the original response was truncated
        """
        # 1. Extract
        text = self.extract_content(raw_response)
        original_len = len(text)
        logger.info(f"Postprocessor: raw content length = {original_len} chars.")

        # 2. Safety check
        self.safety_check(text)

        # 3. Truncate
        text = self.truncate(text)

        # 4. Format
        was_truncated = len(text) < original_len
        result = {
            "answer": text,
            "char_count": len(text),
            "truncated": was_truncated,
        }
        logger.info(
            f"Postprocessor: final length={result['char_count']}, "
            f"truncated={was_truncated}."
        )
        return result
