"""The LLM hiring screener: (job posting, hypercube point) -> accept or reject."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam
from openai.types.shared import ReasoningEffort
from pydantic import BaseModel
from tqdm import tqdm

from .skills_catalog import SkillsCatalog
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
DEFAULT_WORKER_THREAD_COUNT = 4


class ScreeningDecision(BaseModel):
    accept_candidate: bool


class LLMResumeScreener:
    """Screens hypercube points against one fixed job posting, remembering and pricing every decision."""

    def __init__(
        self,
        skills_catalog: SkillsCatalog,
        job_posting_text: str = JOB_POSTING_TEXT,
        deployment_name: str = DEFAULT_DEPLOYMENT_NAME,
        *,
        reasoning_effort: ReasoningEffort = "low",
        token_pricing: TokenPricing = GPT_6_1_SOL_PRICING,
        openai_client: OpenAI = AZURE_OPENAI_CLIENT,
        worker_thread_count: int = DEFAULT_WORKER_THREAD_COUNT,
    ) -> None:
        self.skills_catalog = skills_catalog
        self.deployment_name = deployment_name
        self.worker_thread_count = worker_thread_count
        self.reasoning_effort: ReasoningEffort = reasoning_effort
        self.cost_ledger = CostLedger(pricing=token_pricing)
        self.openai_client = openai_client
        self.decision_by_point: dict[int, ScreeningDecision] = {}
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

    def accepts(self, points: Sequence[int]) -> list[bool]:
        new_points = list(
            dict.fromkeys(
                point for point in points if point not in self.decision_by_point
            )
        )
        finished_calls: list[tuple[int, ScreeningDecision, CallTokenUsage | None]] = []
        with ThreadPoolExecutor(max_workers=self.worker_thread_count) as executor:
            point_by_future = {
                executor.submit(self._request_decision, point): point
                for point in new_points
            }
            try:
                for future in tqdm(
                    as_completed(point_by_future),
                    total=len(point_by_future),
                    desc="LLM calls",
                    leave=False,
                ):
                    finished_calls.append((point_by_future[future], *future.result()))
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        for point, decision, usage in finished_calls:
            self.decision_by_point[point] = decision
            if usage is not None:
                self.cost_ledger.api_calls.append(usage)
        return [self.decision_by_point[point].accept_candidate for point in points]

    def _request_decision(
        self, point: int
    ) -> tuple[ScreeningDecision, CallTokenUsage | None]:
        start_seconds = time.perf_counter()
        response = self.openai_client.responses.parse(
            model=self.deployment_name,
            input=[
                self.cached_developer_message,
                {
                    "role": "user",
                    "content": self.skills_catalog.render_skills_section(point),
                },
            ],
            text_format=ScreeningDecision,
            reasoning={"effort": self.reasoning_effort},
            prompt_cache_key=self.prompt_cache_key,
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            store=False,
        )
        usage = (
            CallTokenUsage.from_response_usage(
                response.usage, compute_seconds=time.perf_counter() - start_seconds
            )
            if response.usage is not None
            else None
        )
        decision = response.output_parsed
        if decision is None:
            raise RuntimeError(
                f"model returned no parsable decision (status={response.status}): {response.output_text!r}"
            )
        return decision, usage
