"""
InContextLearningModule — pipeline orchestrator.

Coordinates the full four-stage pipeline:
  Stage 1 – Preprocess  : clean text, normalise, attach request metadata.
  Stage 2 – Enrich      : inject few-shot examples (ICL), CoT instructions,
                           and RAG context into the prompt.
  Stage 3 – LLM Call    : package enriched prompt into a Gemini API request
                           with configurable temperature / max_tokens / top_p.
  Stage 4 – Postprocess : extract content, run safety check, truncate if
                           needed, and wrap in a standard response envelope.

Returns a dict { answer, metadata } consumed directly by the HTTP layer.
"""

import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from src.pipeline.postprocessor import Postprocessor, SafetyViolationError
from src.pipeline.preprocessor import Preprocessor
from src.pipeline.prompt_enricher import PromptEnricher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="app.log",
    filemode="a",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM hyper-parameters — overridable via environment variables
# ---------------------------------------------------------------------------
_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))


class InContextLearningModule:
    def __init__(self):
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file or environment variables."
            )

        # Stage 3 — Gemini LLM with explicit generation parameters
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=_TEMPERATURE,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
            top_p=_TOP_P,
        )
        logger.info(
            f"LLM initialised — model=gemini-2.5-flash, "
            f"temperature={_TEMPERATURE}, max_tokens={_MAX_OUTPUT_TOKENS}, "
            f"top_p={_TOP_P}"
        )

        self.preprocessor = Preprocessor()
        self.prompt_enricher = PromptEnricher()
        self.postprocessor = Postprocessor()

    def generate_response_from_conversation(
        self, conversation: list, type: str, doc_ids: Optional[List[str]] = None, doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline: Preprocess → Enrich → LLM Call → Postprocess.

        Parameters
        ----------
        conversation : list of {role, content} dicts
        type         : "documentQA" | "documentSummary" | "documentExtraction"
        doc_ids      : restrict RAG to specific documents (None = search all docs)
        doc_id       : (deprecated) single doc restriction, converted to doc_ids
        """
        # Backward-compat: allow callers that still pass a single doc_id
        if doc_id and not doc_ids:
            doc_ids = [doc_id]
        pipeline_start = time.perf_counter()

        # ── Stage 1: Preprocess ─────────────────────────────────────────────
        cleaned_conversation, metadata = self.preprocessor.preprocess_conversation(
            conversation
        )
        req_id = metadata["request_id"]
        logger.info(
            f"[{req_id}] Pipeline start — type={type}, "
            f"ts={metadata['timestamp']}, messages={metadata['message_count']}"
        )

        # The user's actual request is always the last message
        user_input = cleaned_conversation[-1]["content"]

        # ── Stage 2: Enrich prompt (ICL + RAG) ─────────────────────────────
        try:
            enriched_prompt = self.prompt_enricher.enrich(user_input, type, doc_ids=doc_ids)
            logger.info(f"[{req_id}] Prompt enriched successfully.")
        except ValueError as exc:
            logger.error(f"[{req_id}] Prompt enrichment failed: {exc}")
            raise

        # ── Stage 3: LLM call (Gemini) ──────────────────────────────────────
        try:
            logger.info(
                f"[{req_id}] Invoking Gemini "
                f"(temperature={_TEMPERATURE}, max_tokens={_MAX_OUTPUT_TOKENS}, "
                f"top_p={_TOP_P})…"
            )
            raw_response = self.llm.invoke(enriched_prompt)
            logger.info(f"[{req_id}] Raw response received from Gemini.")
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"[{req_id}] Gemini API call failed: {exc}")
            raise

        # ── Stage 4: Postprocess ─────────────────────────────────────────────
        try:
            post_result = self.postprocessor.process(raw_response)
            logger.info(
                f"[{req_id}] Postprocessing complete — "
                f"chars={post_result['char_count']}, "
                f"truncated={post_result['truncated']}."
            )
        except SafetyViolationError as exc:
            logger.warning(f"[{req_id}] Safety violation: {exc}")
            raise

        # ── Assemble final response ──────────────────────────────────────────
        processing_time_ms = int((time.perf_counter() - pipeline_start) * 1000)
        logger.info(f"[{req_id}] Pipeline finished in {processing_time_ms} ms.")

        return {
            "answer": post_result["answer"],
            "metadata": {
                "request_id": req_id,
                "timestamp": metadata["timestamp"],
                "processing_time_ms": processing_time_ms,
                "truncated": post_result["truncated"],
                "char_count": post_result["char_count"],
            },
        }