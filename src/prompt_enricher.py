"""
Prompt Enricher module — Stage 2 of the pipeline.

Responsibilities:
1. In-Context Learning (ICL):
   - Inject few-shot examples (curated per task type)
   - Prepend Chain-of-Thought (CoT) reasoning instructions (via ChainService)
2. Retrieval-Augmented Generation (RAG):
   - For 'documentQA': query the FAISS vector DB and inject retrieved chunks
     as additional context into the prompt before the user's question.

Returns a fully-formed, enriched prompt string ready for the LLM.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from sentence_transformers import CrossEncoder

from src.chain_service import ChainService
from src.vector_db_ingestor import retrieve_similar_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton cross-encoder reranker
# ---------------------------------------------------------------------------
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker once and reuse it for all requests."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model '{_RERANKER_MODEL}'…")
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker

# ------------------------------------------------------------------ #
# Few-shot example banks — one bank per conversation type
# ------------------------------------------------------------------ #

_FEW_SHOT_EXAMPLES: Dict[str, List[Dict[str, str]]] = {
    "documentQA": [
        # --- Factual / definition question ---
        {
            "input": (
                "Question: What is the main purpose of the 'Orion' project?\n\n"
                "Context: The 'Orion' project is a strategic initiative focused on "
                "reducing operational costs by 15% through supply chain automation."
            ),
            "output": (
                "The main purpose of the 'Orion' project is to reduce operational "
                "costs by 15% by automating the supply chain."
            ),
        },
        # --- Answer not present in context ---
        {
            "input": (
                "Question: Who is the CEO of the company?\n\n"
                "Context: The quarterly report was signed by Jane Doe, the Chief "
                "Financial Officer."
            ),
            "output": (
                "The provided document context does not mention the CEO. It only "
                "states that Jane Doe is the Chief Financial Officer."
            ),
        },
        # --- Numerical / financial question ---
        {
            "input": (
                "Question: What was the company's net revenue in Q3 2024?\n\n"
                "Context: In Q3 2024, the company reported a gross revenue of "
                "$4.2 billion. After deducting returns, allowances, and discounts "
                "totalling $310 million, net revenue stood at $3.89 billion, "
                "representing a 12% year-over-year increase."
            ),
            "output": (
                "The company's net revenue in Q3 2024 was $3.89 billion, which "
                "represents a 12% increase compared to the same period the previous year."
            ),
        },
        # --- Causal / 'why' question ---
        {
            "input": (
                "Question: Why did the clinical trial for Drug X fail?\n\n"
                "Context: The Phase III clinical trial for Drug X was terminated "
                "early after an independent safety monitoring board identified a "
                "statistically significant increase in cardiovascular adverse events "
                "in the treatment group (8.3%) versus the placebo group (1.2%). "
                "The trial enrolled 2,400 participants across 14 countries."
            ),
            "output": (
                "The clinical trial for Drug X failed because an independent safety "
                "board found a significantly higher rate of cardiovascular adverse "
                "events in patients receiving the drug (8.3%) compared to those on "
                "a placebo (1.2%), leading to early termination of the trial."
            ),
        },
        # --- Multi-condition / 'how' question ---
        {
            "input": (
                "Question: How does the authentication system work?\n\n"
                "Context: The platform uses a two-factor authentication (2FA) "
                "process. First, users enter their username and password. If the "
                "credentials are valid, a one-time passcode (OTP) is sent to the "
                "user's registered email or mobile number. The OTP expires after "
                "5 minutes. Access is only granted after the OTP is successfully "
                "verified."
            ),
            "output": (
                "The authentication system uses two-factor authentication (2FA). "
                "A user first submits their username and password. Upon successful "
                "credential validation, a one-time passcode (OTP) is sent to their "
                "registered email or phone. The OTP is valid for 5 minutes, and "
                "access is granted only after the OTP is verified."
            ),
        },
        # --- Date / timeline question ---
        {
            "input": (
                "Question: When is the product scheduled to launch?\n\n"
                "Context: Following the completion of beta testing in November 2024, "
                "the engineering team plans to enter a code freeze on December 1, 2024. "
                "The official public launch is scheduled for January 15, 2025, "
                "pending regulatory approval from the relevant authorities."
            ),
            "output": (
                "The product is scheduled to launch on January 15, 2025, subject to "
                "regulatory approval. A code freeze is planned for December 1, 2024, "
                "following the conclusion of beta testing in November 2024."
            ),
        },
        # --- Comparison question ---
        {
            "input": (
                "Question: What is the difference between Plan A and Plan B?\n\n"
                "Context: Plan A offers unlimited cloud storage, priority customer "
                "support, and access to all premium features at $49/month. Plan B "
                "provides 100 GB of storage, standard support, and access to basic "
                "features only at $19/month. Both plans include a 14-day free trial."
            ),
            "output": (
                "Plan A ($49/month) includes unlimited cloud storage, priority support, "
                "and all premium features. Plan B ($19/month) offers 100 GB storage, "
                "standard support, and basic features only. Both plans share a 14-day "
                "free trial."
            ),
        },
        # --- Legal / policy question ---
        {
            "input": (
                "Question: What happens if a user violates the terms of service?\n\n"
                "Context: Section 7.3 states: 'Any user found to be in breach of "
                "these Terms of Service may, at the Company's sole discretion, have "
                "their account suspended temporarily or permanently terminated without "
                "prior notice. The Company reserves the right to pursue legal action "
                "for damages incurred as a result of such violations.'"
            ),
            "output": (
                "According to Section 7.3 of the Terms of Service, users who violate "
                "the terms may have their account temporarily suspended or permanently "
                "terminated without prior notice, at the Company's discretion. The "
                "Company also reserves the right to seek legal action for any damages "
                "caused by such violations."
            ),
        },
        # --- Technical specification question ---
        {
            "input": (
                "Question: What are the minimum system requirements to run the software?\n\n"
                "Context: System Requirements — Minimum: OS: Windows 10 (64-bit) or "
                "macOS 12.0; CPU: Intel Core i5 (8th gen) or AMD Ryzen 5 3000 series; "
                "RAM: 8 GB; Storage: 20 GB free SSD space; GPU: NVIDIA GTX 1060 or "
                "equivalent with 4 GB VRAM. An active internet connection is required "
                "for activation."
            ),
            "output": (
                "The minimum system requirements are: Windows 10 64-bit or macOS 12.0; "
                "an Intel Core i5 (8th gen) or AMD Ryzen 5 3000-series CPU; 8 GB RAM; "
                "20 GB of free SSD storage; and an NVIDIA GTX 1060 (or equivalent) GPU "
                "with 4 GB VRAM. An internet connection is also required for activation."
            ),
        },
        # --- Partially available information ---
        {
            "input": (
                "Question: What are the salary and benefits for the Senior Engineer role?\n\n"
                "Context: The Senior Engineer position offers a competitive salary "
                "commensurate with experience, along with comprehensive health insurance, "
                "20 days of paid annual leave, and a flexible remote-work policy. "
                "Specific salary figures are discussed during the interview process."
            ),
            "output": (
                "The document does not state a specific salary figure for the Senior "
                "Engineer role; it will be discussed during interviews. However, the "
                "benefits include comprehensive health insurance, 20 days of paid annual "
                "leave, and a flexible remote-work policy."
            ),
        },
    ],

    "documentSummary": [
        # --- Market report ---
        {
            "input": (
                "Text: The solar panel market saw a 20% growth in the last quarter, "
                "driven by new government incentives and falling production costs. "
                "Key players are expanding manufacturing capacity to meet demand. "
                "However, challenges remain in grid integration."
            ),
            "output": (
                "The solar panel market grew by 20% last quarter due to government "
                "incentives and lower production costs. While major companies are "
                "increasing output, integrating new capacity into the power grid "
                "remains a key challenge."
            ),
        },
        # --- Scientific / research paper ---
        {
            "input": (
                "Text: This study investigated the effect of daily 30-minute aerobic "
                "exercise on cognitive function in adults aged 60–75. A randomised "
                "controlled trial was conducted over 12 weeks with 180 participants "
                "split evenly into an exercise group and a control group. Results "
                "showed the exercise group achieved significantly higher scores on "
                "memory recall tests (p < 0.01) and executive function assessments "
                "(p < 0.05). No significant difference was found in processing speed. "
                "The findings suggest that moderate aerobic exercise may meaningfully "
                "improve certain cognitive domains in older adults."
            ),
            "output": (
                "A 12-week randomised controlled trial involving 180 adults aged 60–75 "
                "found that 30 minutes of daily aerobic exercise significantly improved "
                "memory recall and executive function compared to a control group. "
                "Processing speed was unaffected. The study suggests moderate aerobic "
                "exercise can benefit key aspects of cognitive function in older adults."
            ),
        },
        # --- Business / financial report ---
        {
            "input": (
                "Text: TechCorp Inc. reported full-year 2024 results today. Total revenue "
                "reached $2.1 billion, a 18% increase from 2023, driven by strong "
                "performance in its cloud services segment, which grew 34% year-over-year. "
                "Net income was $310 million, up from $240 million in 2023. However, "
                "the hardware division experienced a 9% revenue decline due to weakening "
                "consumer demand and component shortages. The company announced a $500 "
                "million share buyback programme and raised its 2025 revenue guidance "
                "to $2.4–$2.5 billion. The CFO noted ongoing cost reduction efforts "
                "targeting $80 million in annual savings."
            ),
            "output": (
                "TechCorp Inc. posted strong 2024 results with total revenue of $2.1 billion "
                "(+18% YoY) and net income of $310 million (+29% YoY), led by a 34% surge "
                "in its cloud services segment. Its hardware division declined 9% due to "
                "weak demand and supply chain issues. The company launched a $500 million "
                "buyback, raised 2025 revenue guidance to $2.4–$2.5 billion, and is "
                "targeting $80 million in annual cost savings."
            ),
        },
        # --- Legal / contract document ---
        {
            "input": (
                "Text: This Service Level Agreement (SLA) between CloudHost Ltd. "
                "(Provider) and Acme Corp. (Client) defines the terms of service "
                "availability and support. The Provider guarantees a monthly uptime "
                "of 99.9%, equating to no more than 43.8 minutes of downtime per month. "
                "In the event of downtime exceeding this threshold, the Client is eligible "
                "for service credits of 10% of the monthly fee per hour of excess downtime, "
                "capped at 30% of the total monthly fee. The Provider will respond to "
                "critical support tickets within 1 hour and resolve them within 4 hours. "
                "This agreement is valid for 12 months from the signing date and renews "
                "automatically unless terminated with 30 days' written notice."
            ),
            "output": (
                "This SLA between CloudHost Ltd. and Acme Corp. guarantees 99.9% monthly "
                "uptime (max 43.8 minutes downtime). Excess downtime triggers service "
                "credits of 10% of the monthly fee per hour, capped at 30%. Critical "
                "support issues must be responded to within 1 hour and resolved within 4 "
                "hours. The agreement runs for 12 months with automatic renewal unless "
                "either party gives 30 days' written notice."
            ),
        },
        # --- Policy / HR document ---
        {
            "input": (
                "Text: The company's Remote Work Policy, effective from March 1, 2025, "
                "allows eligible employees to work from home up to three days per week, "
                "subject to manager approval. Employees must be available during core "
                "hours of 9:00 AM to 3:00 PM in their local time zone and must attend "
                "all mandatory in-person meetings. A reliable internet connection of at "
                "least 25 Mbps is required. Employees are responsible for maintaining a "
                "secure and distraction-free workspace. The company will provide a one-time "
                "home office setup allowance of $500. Non-compliance may result in "
                "revocation of remote work privileges."
            ),
            "output": (
                "Effective March 1, 2025, eligible employees may work remotely up to three "
                "days per week with manager approval. Core working hours are 9 AM–3 PM "
                "local time, and in-person meetings remain mandatory. A 25 Mbps internet "
                "connection and a secure workspace are required. The company offers a "
                "one-time $500 home office allowance. Non-compliance can lead to loss of "
                "remote work privileges."
            ),
        },
        # --- Technical documentation ---
        {
            "input": (
                "Text: The DataSync API provides developers with a real-time data "
                "synchronisation service between client applications and cloud storage. "
                "It supports RESTful and WebSocket interfaces. Authentication is handled "
                "via OAuth 2.0 with JWT tokens that expire after 3600 seconds. Rate "
                "limiting is enforced at 1,000 requests per minute per API key. "
                "The API guarantees eventual consistency and supports conflict resolution "
                "via a last-write-wins strategy. Data is encrypted in transit using "
                "TLS 1.3 and at rest using AES-256. The current stable version is v3.2, "
                "and v2.x will reach end-of-life on June 30, 2025."
            ),
            "output": (
                "The DataSync API enables real-time data sync between apps and cloud "
                "storage via REST and WebSocket interfaces. It uses OAuth 2.0 / JWT "
                "authentication (tokens expire in 3600 s) with a rate limit of 1,000 "
                "requests/min per key. Data is secured with TLS 1.3 in transit and "
                "AES-256 at rest. Consistency is eventual with a last-write-wins conflict "
                "strategy. Stable version is v3.2; v2.x is end-of-life on June 30, 2025."
            ),
        },
    ],

    "documentExtraction": [
        # --- Project names ---
        {
            "input": (
                "Request: Extract all project names from the text.\n\n"
                "Text: The report covers the status of Project Phoenix and Project "
                "Gemini. We are also planning the initial phase of Project Titan."
            ),
            "output": "- Project Phoenix\n- Project Gemini\n- Project Titan",
        },
        # --- Monetary figures ---
        {
            "input": (
                "Request: Extract all monetary values mentioned in the text.\n\n"
                "Text: The infrastructure upgrade budget is capped at $1.2 million. "
                "Marketing has been allocated $340,000 for Q1 campaigns. The CEO's "
                "compensation package totals $850,000 annually, and the emergency "
                "reserve fund holds $500,000."
            ),
            "output": (
                "- $1,200,000 — infrastructure upgrade budget\n"
                "- $340,000 — Q1 marketing campaigns\n"
                "- $850,000 — annual CEO compensation package\n"
                "- $500,000 — emergency reserve fund"
            ),
        },
        # --- Dates and deadlines ---
        {
            "input": (
                "Request: Extract all dates and their associated events from the text.\n\n"
                "Text: The project kick-off meeting is scheduled for March 3, 2025. "
                "The first milestone review will take place on April 14, 2025. Beta "
                "testing is set to begin on June 1, 2025, and the final product launch "
                "is planned for September 30, 2025. A post-launch review is scheduled "
                "for October 15, 2025."
            ),
            "output": (
                "- March 3, 2025 — Project kick-off meeting\n"
                "- April 14, 2025 — First milestone review\n"
                "- June 1, 2025 — Beta testing begins\n"
                "- September 30, 2025 — Final product launch\n"
                "- October 15, 2025 — Post-launch review"
            ),
        },
        # --- People and roles ---
        {
            "input": (
                "Request: Extract all people and their roles or titles from the text.\n\n"
                "Text: The annual report was presented by Dr. Sarah Chen, Chief Executive "
                "Officer. The financial section was prepared by Mr. David Okafor, CFO. "
                "Technical oversight was provided by Ms. Priya Nair, VP of Engineering. "
                "Legal compliance was confirmed by James Müller, General Counsel."
            ),
            "output": (
                "- Dr. Sarah Chen — Chief Executive Officer (CEO)\n"
                "- Mr. David Okafor — Chief Financial Officer (CFO)\n"
                "- Ms. Priya Nair — VP of Engineering\n"
                "- James Müller — General Counsel"
            ),
        },
        # --- Email addresses ---
        {
            "input": (
                "Request: Extract all email addresses from the text.\n\n"
                "Text: For billing inquiries, contact billing@techcorp.com. Technical "
                "support requests should be sent to support@techcorp.com. The project "
                "manager, Alice Wong, can be reached at a.wong@techcorp.com. Media "
                "inquiries should be directed to press@techcorp.com."
            ),
            "output": (
                "- billing@techcorp.com — billing inquiries\n"
                "- support@techcorp.com — technical support\n"
                "- a.wong@techcorp.com — project manager (Alice Wong)\n"
                "- press@techcorp.com — media inquiries"
            ),
        },
        # --- Percentages and KPIs ---
        {
            "input": (
                "Request: Extract all percentage figures and what they refer to.\n\n"
                "Text: Customer satisfaction scores improved by 12% following the "
                "new support rollout. Churn rate decreased from 8.4% to 5.1% year-over-year. "
                "Employee engagement rose to 76%, while operational costs were reduced "
                "by 9.3% through process automation. Market share increased marginally "
                "from 22% to 24%."
            ),
            "output": (
                "- 12% — improvement in customer satisfaction scores\n"
                "- 8.4% → 5.1% — churn rate reduction (year-over-year)\n"
                "- 76% — employee engagement rate\n"
                "- 9.3% — operational cost reduction via automation\n"
                "- 22% → 24% — market share increase"
            ),
        },
        # --- Locations / addresses ---
        {
            "input": (
                "Request: Extract all locations or addresses mentioned in the text.\n\n"
                "Text: The company's global headquarters are located at 100 Innovation "
                "Drive, San Francisco, CA 94105. The European office is based in "
                "Amsterdam, Netherlands. Manufacturing operations are run from a facility "
                "in Ho Chi Minh City, Vietnam. A new distribution hub is being opened "
                "in Dubai, UAE by Q2 2025."
            ),
            "output": (
                "- 100 Innovation Drive, San Francisco, CA 94105 — global headquarters\n"
                "- Amsterdam, Netherlands — European office\n"
                "- Ho Chi Minh City, Vietnam — manufacturing facility\n"
                "- Dubai, UAE — new distribution hub (opening Q2 2025)"
            ),
        },
        # --- Action items / obligations ---
        {
            "input": (
                "Request: Extract all action items and who is responsible for each.\n\n"
                "Text: Following today's board meeting, the following actions were agreed: "
                "The Finance team will submit a revised budget proposal by Friday. "
                "HR is to complete the new hire onboarding checklist before end of month. "
                "The CTO must deliver a cybersecurity risk report to the board within "
                "two weeks. Marketing is responsible for launching the Q2 campaign no "
                "later than April 1st."
            ),
            "output": (
                "- Finance team: submit revised budget proposal by Friday\n"
                "- HR: complete new hire onboarding checklist by end of month\n"
                "- CTO: deliver cybersecurity risk report to the board within two weeks\n"
                "- Marketing: launch Q2 campaign by April 1st"
            ),
        },
    ],
}

_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

_SUFFIX = "\n---\n\n{user_input}\n\n---"


# ------------------------------------------------------------------ #
# Prompt Enricher
# ------------------------------------------------------------------ #


class PromptEnricher:
    """
    Builds a fully enriched prompt by combining:
    - CoT prefix (via ChainService)
    - Few-shot examples (ICL)
    - RAG context (for documentQA)
    """

    def __init__(self):
        self._chain_service = ChainService()

    # ---- private helpers ------------------------------------------ #

    def _build_few_shot_template(
        self, conversation_type: str
    ) -> FewShotPromptTemplate:
        if conversation_type not in _FEW_SHOT_EXAMPLES:
            raise ValueError(
                f"No few-shot examples defined for type: '{conversation_type}'"
            )
        cot_prefix = self._chain_service.build_cot_prefix(conversation_type)
        return FewShotPromptTemplate(
            examples=_FEW_SHOT_EXAMPLES[conversation_type],
            example_prompt=_EXAMPLE_PROMPT,
            prefix=cot_prefix,
            suffix=_SUFFIX,
            input_variables=["user_input"],
        )

    def _retrieve_rag_context(self, query: str, top_k: int = 5, doc_ids: Optional[List[str]] = None) -> str:
        """
        Two-stage retrieval:
        1. Retrieve *candidate_k* chunks from FAISS (dense vector search).
        2. Rerank all candidates with a cross-encoder and return the top *top_k*.
        """
        candidate_k = max(top_k * 3, 15)
        try:
            candidates = retrieve_similar_text(query, top_k=candidate_k, doc_ids=doc_ids)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            return "[No relevant information found in the document.]"

        if not candidates:
            return "[No relevant information found in the document.]"

        try:
            reranker = _get_reranker()
            pairs = [(query, chunk) for chunk in candidates]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for _, chunk in ranked[:top_k]]
            logger.info(
                f"RAG: retrieved {len(candidates)} candidates, "
                f"reranked → top {len(top_chunks)} chunks returned."
            )
        except Exception as exc:
            logger.warning(f"Reranking failed, falling back to FAISS order: {exc}")
            top_chunks = candidates[:top_k]

        return "\n---\n".join(top_chunks)

    # ---- public API ----------------------------------------------- #

    def enrich(self, user_input: str, conversation_type: str, doc_ids: Optional[List[str]] = None) -> str:
        """
        Build the fully enriched prompt string.

        Parameters
        ----------
        user_input        : cleaned user query / document text
        conversation_type : "documentQA" | "documentSummary" | "documentExtraction"
        doc_ids           : restrict RAG retrieval to specific documents (None = all docs)
        """
        template = self._build_few_shot_template(conversation_type)

        if conversation_type == "documentQA":
            rag_context = self._retrieve_rag_context(user_input, doc_ids=doc_ids)
            final_input = f"Question: {user_input}\n\nContext:\n{rag_context}"
        else:
            final_input = user_input

        enriched_prompt = template.format(user_input=final_input)
        logger.debug(
            f"[PromptEnricher] Enriched prompt (type={conversation_type}):\n"
            f"{enriched_prompt[:300]}…"
        )
        return enriched_prompt
