"""
Pydantic schemas for LLM-generated misinformation explanations.

Fields produced per claim:
  - output:     Short user-facing explanation of why the post may be misleading.
  - reasoning:  Plain-language justification of the true label.
  - issue_type: One of six structured cause categories (see ISSUE_TYPE_DEFINITIONS).
"""

from typing import Literal
from pydantic import BaseModel, Field

# Canonical taxonomy of misclassification causes, shared between the schema,
# the system prompt, and any downstream analysis code.
ISSUE_TYPES = (
    "Misleading framing",
    "Overgeneralisation",
    "Factual error",
    "Preliminary/uncertain",
    "Missing context",
    "Conspiracy/junk source",
    "Other",
)

IssueTypeLiteral = Literal[
    "Misleading framing",
    "Overgeneralisation",
    "Factual error",
    "Preliminary/uncertain",
    "Missing context",
    "Conspiracy/junk source",
    "Other",
]

# Human-readable definitions used verbatim in the system prompt.
ISSUE_TYPE_DEFINITIONS = {
    "Misleading framing": (
        "The claim creates a false impression through selective emphasis, loaded "
        "language, false dilemmas, or implications not supported by the evidence — "
        "even if individual facts stated are technically accurate."
    ),
    "Overgeneralisation": (
        "The claim makes an absolute or sweeping statement that is only partially "
        "true. It ignores important exceptions, nuance, or context, often using "
        "language like 'always', 'never', 'all', or 'guaranteed'."
    ),
    "Factual error": (
        "The claim contains a statement that is demonstrably incorrect, fabricated, "
        "or originates from a satirical or fictional source presented as real news."
    ),
    "Preliminary/uncertain": (
        "The claim presents early-stage, speculative, or contested research as "
        "settled fact. The underlying evidence is limited, preliminary, or subject "
        "to ongoing scientific debate."
    ),
    "Missing context": (
        "The claim is technically accurate in isolation but omits crucial context "
        "that would materially change how a reader interprets it. The omitted "
        "information is necessary for a fair and complete understanding."
    ),
    "Conspiracy/junk source": (
        "The claim originates from or is primarily amplified by unreliable, dubious, "
        "or known misinformation sources such as clickbait sites, fake-news outlets, "
        "or conspiracy communities."
    ),
    "Other": (
        "The primary cause of potential misclassification does not clearly fit any "
        "of the above categories."
    ),
}


class MisinfoExplanation(BaseModel):
    """Structured LLM output for a single misclassified claim."""

    output: str = Field(
        description=(
            "A concise user-facing explanation (2–4 sentences) of why this post "
            "may be misleading or how it should be interpreted. Focus on the "
            "specific issue identified in issue_type."
        )
    )
    reasoning: str = Field(
        description=(
            "A brief plain-language justification (1–2 sentences) of why the true "
            "label is correct. Must be fully consistent with the assigned label "
            "(true / false / mixture)."
        )
    )
    issue_type: IssueTypeLiteral = Field(
        description=(
            "The single best-fitting cause category from the taxonomy defined in "
            "the system prompt. Choose the one that most directly explains why the "
            "claim is potentially misleading."
        )
    )


class IssueTypeOnly(BaseModel):
    """Lightweight schema used for backfilling issue_type on existing outputs."""

    issue_type: IssueTypeLiteral = Field(
        description=(
            "The single best-fitting cause category from the taxonomy defined in "
            "the system prompt."
        )
    )