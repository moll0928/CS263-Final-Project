"""
LLM pipeline for generating misinformation explanations.

Reads the misclassified-top-200 CSVs produced by the BERT classifier, calls a
strong LLM for each row, and appends two columns to the output:
  - llm_output    : user-facing explanation of why the post is misleading
  - llm_reasoning : plain-language justification of the true label

Supports Google Gemini and OpenAI via structured (JSON) output with retries.
"""

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from schemas_misinfo import MisinfoExplanation, ISSUE_TYPE_DEFINITIONS

# ---------------------------------------------------------------------------
# Retry / rate-limit helpers (adapted from llm_baseline/llm_client.py)
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BASE_DELAY = 5.0  # seconds


def _parse_retry_after(error_message: str) -> Optional[float]:
    patterns = [
        r"retry\s+in\s+(\d+(?:\.\d+)?)s",
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)",
        r"retry[_\s-]?after[:\s]+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*seconds?",
        r"(\d+(?:\.\d+)?)s\b",
    ]
    for pat in patterns:
        m = re.search(pat, str(error_message), re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _is_rate_limit(error: Exception) -> bool:
    s = str(error).lower()
    return any(k in s for k in [
        "rate limit", "rate_limit", "ratelimit",
        "quota exceeded", "resource_exhausted",
        "too many requests", "429",
    ])


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Gemini
    "gemini-2.5-pro":                        {"input": 1.25, "output": 10.00},
    "gemini-2.5-pro-preview-06-05":          {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash-preview-05-20":        {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash":                      {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite":                 {"input": 0.075, "output": 0.30},
    # OpenAI
    "gpt-4o-2024-08-06":                     {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-2024-07-18":                {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini-2025-04-14":               {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano-2025-04-14":               {"input": 0.10, "output": 0.40},
}

GEMINI_MODELS = {m for m in MODEL_COSTS if m.startswith("gemini")}
OPENAI_MODELS = {m for m in MODEL_COSTS if m.startswith("gpt")}


@dataclass
class CostTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    total_cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, model: str):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_calls += 1
        costs = MODEL_COSTS.get(model, {"input": 0.5, "output": 1.5})
        self.total_cost_usd += (
            prompt_tokens / 1_000_000 * costs["input"]
            + completion_tokens / 1_000_000 * costs["output"]
        )

    def summary(self) -> str:
        total = self.total_prompt_tokens + self.total_completion_tokens
        return (
            f"{self.total_calls} calls | "
            f"{total:,} tokens "
            f"({self.total_prompt_tokens:,} in + {self.total_completion_tokens:,} out) | "
            f"${self.total_cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# LLM client (Gemini + OpenAI, structured output)
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.cost_tracker = CostTracker()

        if model in GEMINI_MODELS or model.startswith("gemini"):
            self.provider = "gemini"
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Set GOOGLE_API_KEY or pass --api-key")
            from google import genai
            self._client = genai.Client(api_key=key)

        else:
            self.provider = "openai"
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY or pass --api-key")
            from openai import OpenAI
            self._client = OpenAI(api_key=key)

    def explain(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
    ) -> MisinfoExplanation:
        """Call the LLM and return a MisinfoExplanation. Retries on rate limits."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(messages, temperature)
                else:
                    return self._call_openai(messages, temperature)
            except Exception as e:
                last_error = e
                if _is_rate_limit(e):
                    wait = _parse_retry_after(str(e)) or BASE_DELAY * (2 ** attempt)
                    print(f"  Rate limit – waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    raise
        raise last_error

    # -- Gemini ---------------------------------------------------------------

    def _call_gemini(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> MisinfoExplanation:
        from google.genai import types

        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(types.Content(
                    role="user" if msg["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=msg["content"])],
                ))

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=MisinfoExplanation,
            ),
        )
        usage = response.usage_metadata
        self.cost_tracker.add(
            usage.prompt_token_count or 0,
            usage.candidates_token_count or 0,
            self.model,
        )
        parsed_dict = json.loads(response.text)
        return MisinfoExplanation(**parsed_dict)

    # -- OpenAI ---------------------------------------------------------------

    def _call_openai(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> MisinfoExplanation:
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=MisinfoExplanation,
            temperature=temperature,
        )
        usage = response.usage
        self.cost_tracker.add(
            usage.prompt_tokens,
            usage.completion_tokens,
            self.model,
        )
        return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_issue_type_block() -> str:
    """Render the issue-type taxonomy section of the system prompt."""
    lines = ["ISSUE TYPE TAXONOMY", "=" * 40]
    for name, definition in ISSUE_TYPE_DEFINITIONS.items():
        lines.append(f'\n• "{name}"\n  {definition}')
    lines.append("=" * 40)
    return "\n".join(lines)


SYSTEM_PROMPT = f"""\
You are an expert fact-checker and misinformation analyst.

You will be given a social media claim or news headline together with its
verified true label (true / false / mixture) and optional context such as
a fact-check explanation or background article.

Your task is to produce three fields:

1. output  – A concise user-facing explanation (2–4 sentences) of *why* this
             post may be misleading or how it should be interpreted. Ground the
             explanation in the specific issue_type you identify. Write for a
             general audience; avoid technical jargon.

2. reasoning – A brief plain-language justification (1–2 sentences) of why the
               true label applies. This must be fully consistent with the
               assigned label (true / false / mixture).

3. issue_type – The single best-fitting cause category from the taxonomy below.
               Choose the one that MOST DIRECTLY explains why the claim is
               potentially misleading. If none fit well, use "Other".

{_build_issue_type_block()}
"""


def build_intel_messages(row: dict) -> List[Dict[str, str]]:
    """Build prompt for the LIAR-New / INTEL dataset row."""
    claim = row.get("text", "").strip()
    true_label = row.get("true_label", "unknown")
    llm_reasoning = row.get("reasoning", "").strip()

    user_content = f"Claim: {claim}\nTrue label: {true_label}"
    if llm_reasoning:
        user_content += f"\n\nAdditional context (prior LLM analysis):\n{llm_reasoning}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


def build_pubhealth_messages(row: dict, max_context_chars: int = 2000) -> List[Dict[str, str]]:
    """Build prompt for the PUBHEALTH dataset row."""
    claim = row.get("claim", "").strip()
    true_label = row.get("true_label", "unknown")
    main_text = (row.get("main_text") or "").strip()
    explanation = (row.get("explanation") or "").strip()

    # Prefer shorter explanation first, fall back to main_text
    context = explanation or main_text
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "…"

    user_content = f"Claim: {claim}\nTrue label: {true_label}"
    if context:
        user_content += f"\n\nFact-check context:\n{context}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_csv(
    input_path: str,
    output_path: str,
    dataset: str,          # "intel" or "pubhealth"
    client: LLMClient,
    temperature: float = 0.3,
    resume: bool = True,
) -> None:
    """
    Process all rows in input_path, appending llm_output / llm_reasoning
    columns, and write to output_path.

    If resume=True and output_path already exists, rows that already have
    a non-empty llm_output are skipped (allows resuming after interruption).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read all input rows
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Add new columns if not present
    out_fields = list(fieldnames)
    for col in ("llm_output", "llm_reasoning"):
        if col not in out_fields:
            out_fields.append(col)

    # Load already-processed rows if resuming
    done_indices: set = set()
    if resume and output_path.exists():
        with open(output_path, newline="", encoding="utf-8-sig") as f:
            done_reader = csv.DictReader(f)
            for done_row in done_reader:
                if done_row.get("llm_output", "").strip():
                    idx = done_row.get("idx", "")
                    done_indices.add(idx)
        print(f"  Resuming: {len(done_indices)} rows already processed.")

    # Open output in append mode if resuming, else write mode
    mode = "a" if (resume and output_path.exists() and done_indices) else "w"
    out_f = open(output_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=out_fields, extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    total = len(rows)
    skipped = 0
    for i, row in enumerate(rows):
        row_idx = row.get("idx", str(i))

        # Skip already-done rows
        if row_idx in done_indices:
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] idx={row_idx}", end=" ", flush=True)

        try:
            if dataset == "intel":
                messages = build_intel_messages(row)
            else:
                messages = build_pubhealth_messages(row)

            result = client.explain(messages, temperature=temperature)
            row["llm_output"] = result.output
            row["llm_reasoning"] = result.reasoning
            print("✓")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            row["llm_output"] = ""
            row["llm_reasoning"] = ""

        writer.writerow(row)
        out_f.flush()

    out_f.close()
    print(f"\nDone. Processed {total - skipped} rows ({skipped} skipped/resumed).")
    print(f"Cost tracker: {client.cost_tracker.summary()}")
    print(f"Output written to: {output_path}")