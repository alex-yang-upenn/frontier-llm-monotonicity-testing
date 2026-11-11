"""Token accounting and dollar cost of the LLM calls.

Every input token is billed as exactly one of: uncached, cache read, or cache write.
Output tokens include the model's hidden reasoning tokens.
"""

from __future__ import annotations

from openai.types.responses import ResponseUsage
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, computed_field

TOKENS_PER_MILLION = 1_000_000


class TokenPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_usd_per_million_tokens: NonNegativeFloat
    cache_read_usd_per_million_tokens: NonNegativeFloat
    cache_write_usd_per_million_tokens: NonNegativeFloat
    output_usd_per_million_tokens: NonNegativeFloat


# GPT-6.1 Sol (short context), Global deployment, from https://azure.microsoft.com/pricing/details/azure-openai/
# (retrieved 2026-11-11). Our prompts (job posting + skills section) are well within short context.
GPT_6_1_SOL_PRICING = TokenPricing(
    input_usd_per_million_tokens=2.00,
    cache_read_usd_per_million_tokens=0.10,
    cache_write_usd_per_million_tokens=2.50,
    output_usd_per_million_tokens=10.00,
)


class CallTokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: NonNegativeInt
    cache_read_tokens: NonNegativeInt
    cache_write_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    reasoning_tokens: NonNegativeInt

    @classmethod
    def from_response_usage(cls, usage: ResponseUsage) -> CallTokenUsage:
        return cls(
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.input_tokens_details.cached_tokens,
            cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cache_read_tokens - self.cache_write_tokens

    def cost_usd(self, pricing: TokenPricing) -> float:
        return (
            self.uncached_input_tokens * pricing.input_usd_per_million_tokens
            + self.cache_read_tokens * pricing.cache_read_usd_per_million_tokens
            + self.cache_write_tokens * pricing.cache_write_usd_per_million_tokens
            + self.output_tokens * pricing.output_usd_per_million_tokens
        ) / TOKENS_PER_MILLION

    def cost_without_caching_usd(self, pricing: TokenPricing) -> float:
        return (
            self.input_tokens * pricing.input_usd_per_million_tokens
            + self.output_tokens * pricing.output_usd_per_million_tokens
        ) / TOKENS_PER_MILLION


class CostLedger(BaseModel):
    """Usage of every API call."""

    pricing: TokenPricing
    api_calls: list[CallTokenUsage] = Field(default_factory=list)

    @computed_field
    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd(self.pricing) for call in self.api_calls)

    @computed_field
    @property
    def cost_without_caching_usd(self) -> float:
        return sum(call.cost_without_caching_usd(self.pricing) for call in self.api_calls)
