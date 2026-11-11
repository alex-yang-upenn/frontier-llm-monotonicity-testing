"""Goldreich, Goldwasser, Lehman, Ron, Samorodnitsky. "Testing Monotonicity." Combinatorica, 2000. Algorithm 1.

Points of the hypercube are int bitmasks: bit i is coordinate i.

Repeat `sample_count` times: pick a uniformly random edge of the hypercube and query both
endpoints. Report "not monotone" if any sampled edge is violated.

(Thm. 2): a function at distance >= distance_threshold from monotone has at least a distance_threshold / dimension
fraction of violated edges, so with the paper's dimension / distance_threshold samples one is
found with probability >= 1 - 1/e.

Unlike the paper we do not stop at the first violation: every edge is sampled, so the fraction of violated edges
can also be estimated.
"""

from __future__ import annotations

import math
import random
from enum import StrEnum
from typing import Annotated, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    computed_field,
    validate_call,
)
from tqdm import tqdm

from .token_pricing import CostLedger

UnitIntervalFloat = Annotated[float, Field(gt=0, le=1)]


class TesterVerdict(StrEnum):
    LIKELY_MONOTONE = "likely monotone"
    NOT_MONOTONE = "not monotone"


class MonotonicityTestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: int
    distance_threshold: float
    verdict: TesterVerdict
    violated_edges: list[tuple[int, int]]
    """(lower point, upper point) bitmasks of each violating sample: f(lower) = 1 but f(upper) = 0."""
    sample_count: int
    elapsed_seconds: float
    seconds_per_sample: float
    cost: CostLedger | None
    """Cost of the API calls made by this run alone, if a ledger was given."""

    @computed_field
    @property
    def violated_edge_fraction(self) -> float:
        """Unbiased estimate of the fraction of all hypercube edges that are violated."""
        return len(self.violated_edges) / self.sample_count

    @computed_field
    @property
    def distance_lower_estimate(self) -> float:
        """Goldreich et al., Prop. 3: distance >= violated edge fraction / 2."""
        return self.violated_edge_fraction / 2

    @computed_field
    @property
    def distance_upper_estimate(self) -> float:
        """Goldreich et al., Thm. 2: distance <= dimension * violated edge fraction (and always <= 1/2)."""
        return min(0.5, self.dimension * self.violated_edge_fraction)


class EdgeTester:
    @validate_call
    def sample_count(
        self, dimension: PositiveInt, distance_threshold: UnitIntervalFloat
    ) -> int:
        return math.ceil(dimension / distance_threshold)

    def run_test(
        self,
        boolean_function: Callable[[int], bool],
        dimension: int,
        distance_threshold: float,
        random_generator: random.Random,
        cost_ledger: CostLedger | None = None,
    ) -> MonotonicityTestResult:
        sample_count = self.sample_count(dimension, distance_threshold)
        violated_edges: list[tuple[int, int]] = []
        first_call_index = len(cost_ledger.api_calls) if cost_ledger else 0
        progress_bar = tqdm(range(sample_count), desc="Edge tester")
        for _ in progress_bar:
            flipped_bit = 1 << random_generator.randrange(dimension)
            lower_point = random_generator.getrandbits(dimension) & ~flipped_bit
            if boolean_function(lower_point) and not boolean_function(
                lower_point | flipped_bit
            ):
                violated_edges.append((lower_point, lower_point | flipped_bit))
                progress_bar.set_postfix(violations=len(violated_edges))
        elapsed_seconds = progress_bar.format_dict["elapsed"]

        return MonotonicityTestResult(
            dimension=dimension,
            distance_threshold=distance_threshold,
            verdict=(
                TesterVerdict.NOT_MONOTONE
                if violated_edges
                else TesterVerdict.LIKELY_MONOTONE
            ),
            violated_edges=violated_edges,
            sample_count=sample_count,
            elapsed_seconds=elapsed_seconds,
            seconds_per_sample=elapsed_seconds / sample_count,
            cost=cost_ledger.calls_since(first_call_index) if cost_ledger else None,
        )
