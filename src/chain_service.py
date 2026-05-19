"""
Chain-of-Thought (CoT) service.

Provides structured, step-by-step reasoning instructions for each
conversation type.  Used by PromptEnricher to build the CoT prefix
injected into every prompt.
"""

from typing import Dict, List


# Ordered reasoning steps for each supported task type
_COT_STEPS: Dict[str, List[str]] = {
    "documentQA": [
        "Analyze the Question: Understand exactly what the user is asking.",
        "Scan the Context: Carefully read every retrieved text snippet.",
        "Synthesize: Formulate an answer using *only* information from the context.",
        "Verify: If the answer is not present in the context, state clearly that "
        "the information is unavailable. Do not draw on external knowledge.",
    ],
    "documentSummary": [
        "Identify Core Themes: Read the full text to find main topics, "
        "arguments, and conclusions.",
        "Extract Key Points: Select the most important sentences per theme.",
        "Synthesize: Draft a concise, coherent summary that connects the key points.",
        "Final Output: Provide only the polished, final summary.",
    ],
    "documentExtraction": [
        "Identify the Target: Determine exactly what type of information to extract "
        "(e.g. names, dates, figures, locations).",
        "Scan the Document: Locate every instance of that information in the text.",
        "Format: Present the results as a clear, structured list.",
        "Final Output: Provide only the extracted data — no explanations.",
    ],
}

_ROLE_DESCRIPTIONS: Dict[str, str] = {
    "documentQA": (
        "You are an AI Document Assistant. "
        "Answer questions based *only* on the provided document context. "
        "Always respond in Vietnamese."
    ),
    "documentSummary": (
        "You are an AI Document Assistant specialising in summarisation. "
        "Always respond in Vietnamese."
    ),
    "documentExtraction": (
        "You are an AI Document Assistant that extracts specific information "
        "from documents. "
        "Always respond in Vietnamese."
    ),
}


class ChainService:
    """Builds Chain-of-Thought reasoning instruction blocks."""

    def get_supported_types(self) -> List[str]:
        return list(_COT_STEPS.keys())

    def build_cot_prefix(self, conversation_type: str) -> str:
        """
        Return a fully formatted CoT reasoning instruction block for the
        given *conversation_type*.

        Raises ValueError for unsupported types.
        """
        if conversation_type not in _COT_STEPS:
            raise ValueError(
                f"Unsupported conversation_type: '{conversation_type}'. "
                f"Supported: {self.get_supported_types()}"
            )

        role_desc = _ROLE_DESCRIPTIONS[conversation_type]
        steps = _COT_STEPS[conversation_type]

        numbered_steps = "\n".join(
            f"{i + 1}. **{step.split(':')[0]}:**{step[len(step.split(':')[0]):]}"
            for i, step in enumerate(steps)
        )

        return (
            f"{role_desc}\n\n"
            "Please reason through your answer step by step:\n"
            f"{numbered_steps}"
        )