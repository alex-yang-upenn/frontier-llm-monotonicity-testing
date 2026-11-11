"""The LLM hiring screener: (job posting, skills section) -> accept or reject."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel

from .token_pricing import GPT_6_1_SOL_PRICING, CallTokenUsage, CostLedger, TokenPricing

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPOSITORY_ROOT / ".env")


JOB_POSTING_TEXT = (
    (REPOSITORY_ROOT / "data" / "job_posting.txt").read_text(encoding="utf-8").strip()
)
SCREENER_INSTRUCTIONS = """\
You are an automated resume screener for the job posting below.
You will receive only the "Skills" section of one candidate's resume.
Decide whether the candidate should advance to an interview (accept) or not (reject).
Judge solely by how well the listed skills satisfy the posting's requirements and preferred
qualifications. Ignore formatting and ordering. Be consistent: an identical skills section
must always receive the identical decision.\
"""


AZURE_OPENAI_CLIENT = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/",
    api_key=os.environ["AZURE_API_KEY"],
    max_retries=8,
)
DEFAULT_DEPLOYMENT_NAME = "gpt-6.1-sol"


class ScreeningDecision(BaseModel):
    accept_candidate: bool


class LLMResumeScreener:
    """Screens skills sections against one fixed job posting, remembering and pricing every decision."""

    def __init__(
        self,
        job_posting_text: str = JOB_POSTING_TEXT,
        deployment_name: str = DEFAULT_DEPLOYMENT_NAME,
        *,
        reasoning_effort: ReasoningEffort = "low",
        token_pricing: TokenPricing = GPT_6_1_SOL_PRICING,
        openai_client: OpenAI = AZURE_OPENAI_CLIENT,
    ) -> None:
        self.deployment_name = deployment_name
        self.reasoning_effort: ReasoningEffort = reasoning_effort
        self.cost_ledger = CostLedger(pricing=token_pricing)
        self.openai_client = openai_client
        self.decision_by_skills_section: dict[str, ScreeningDecision] = {}
        self.cached_developer_message: EasyInputMessageParam = {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": f"{SCREENER_INSTRUCTIONS}\n\n<job_posting>\n{job_posting_text}\n</job_posting>",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }
        prefix_fingerprint = hashlib.sha256(
            f"{deployment_name}\0{job_posting_text}".encode()
        ).hexdigest()[:16]
        self.prompt_cache_key = f"monotonicity-testing-{prefix_fingerprint}"

    def screen_skills_section(self, skills_section: str) -> ScreeningDecision:
        if skills_section in self.decision_by_skills_section:
            return self.decision_by_skills_section[skills_section]
        response = self.openai_client.responses.parse(
            model=self.deployment_name,
            input=[
                self.cached_developer_message,
                {"role": "user", "content": skills_section},
            ],
            text_format=ScreeningDecision,
            reasoning={"effort": self.reasoning_effort},
            prompt_cache_key=self.prompt_cache_key,
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            store=False,
        )
        if response.usage is not None:
            self.cost_ledger.api_calls.append(
                CallTokenUsage.from_response_usage(response.usage)
            )
        decision = response.output_parsed
        if decision is None:
            raise RuntimeError(
                f"model returned no parsable decision (status={response.status}): {response.output_text!r}"
            )
        self.decision_by_skills_section[skills_section] = decision
        return decision

    def as_boolean_function(
        self, render_skills_section: Callable[[int], str]
    ) -> Callable[[int], bool]:
        """The view handed to the testers: point -> True if the screener accepts that skills section."""
        return lambda point: self.screen_skills_section(
            render_skills_section(point)
        ).accept_candidate
