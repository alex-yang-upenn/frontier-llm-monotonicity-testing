"""Fattal, Ron. "Approximating the Distance to Monotonicity in High Dimensions." ACM TALG, 2010. Algorithm 1.

Points of the hypercube are int bitmasks: bit i is coordinate i.

Split the coordinates into consecutive blocks of `subcube_dimension` coordinates (the last
block may be shorter).Repeat `sample_count` times:
 - pick a random block, fix every other coordinate at random
 - compute the exact distance to monotonicity of the function on that small subcube
 - return the mean of those distances minus a small bias correction.

Guarantee (Lemma 3.2 with Lemma 3.1), with probability >= 2/3:
    true_distance / (2 * block_count) - additive_error <= estimate <= true_distance.

We use the paper's constants: ceil(18 / additive_error^2) samples, and a bias
correction of additive_error / 3. Subcube distances are computed exactly (`exact_distance.py`).

Each sample costs up to 2^subcube_dimension queries. All subcubes are drawn first and every point of every subcube
is queried in one concurrent batch; the exact distances are then computed from those answers.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from typing import Annotated, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    computed_field,
    validate_call,
)
from .token_pricing import CostLedger
from .fattal_ron_subcube_exact_distance import exact_distance_to_monotonicity

UnitIntervalFloat = Annotated[float, Field(gt=0, le=1)]


class DistanceEstimate(BaseModel):
    """Fattal & Ron, Def. 1: with probability >= 2/3,

    true_distance / approximation_factor - additive_error <= estimated_distance <= true_distance.
    """

    model_config = ConfigDict(frozen=True)

    dimension: int
    estimated_distance: float
    approximation_factor: float
    additive_error: float
    sample_count: int
    mean_subcube_distance: float
    total_wall_clock_seconds: float
    cost: CostLedger | None

    @computed_field
    @property
    def seconds_per_sample(self) -> float | None:
        """Total compute time of this run's API calls / sample_count, if a ledger was given."""
        return self.cost.total_compute_seconds / self.sample_count if self.cost else None

    @computed_field
    @property
    def guaranteed_distance_range(self) -> tuple[float, float]:
        """Contains the true distance with probability >= 2/3."""
        upper_bound = self.approximation_factor * (
            self.estimated_distance + self.additive_error
        )
        return self.estimated_distance, min(0.5, upper_bound)


class FattalRonEstimator(BaseModel):
    model_config = ConfigDict(frozen=True)

    subcube_dimension: PositiveInt

    @validate_call
    def sample_count(self, additive_error: UnitIntervalFloat) -> int:
        return math.ceil(18 / additive_error**2)

    def approximation_factor(self, dimension: int) -> float:
        return 2 * math.ceil(dimension / self.subcube_dimension)

    def estimate_distance(
        self,
        boolean_function: Callable[[Sequence[int]], list[bool]],
        dimension: int,
        additive_error: float,
        random_generator: random.Random,
        cost_ledger: CostLedger | None = None,
    ) -> DistanceEstimate:
        sample_count = self.sample_count(additive_error)
        # Generate subcubes, described by (fixed_bits, block_start, block_size)
        subcubes: list[tuple[int, int, int]] = []
        for _ in range(sample_count):
            # Randomly choose a subcube of dimension `subcube_dimension`
            block_start = random_generator.randrange(
                0, dimension, self.subcube_dimension
            )
            block_size = min(self.subcube_dimension, dimension - block_start)
            block_bits = (2**block_size - 1) << block_start

            # Randomly fix the other coordinates of the hypercube
            fixed_bits = random_generator.getrandbits(dimension) & ~block_bits

            subcubes.append((fixed_bits, block_start, block_size))

        # Pre-compute all the points of all sampled subcubes and query the boolean function on them in one batch.
        first_call_index = len(cost_ledger.api_calls) if cost_ledger else 0
        start_seconds = time.perf_counter()
        points = [
            fixed_bits | subcube_point << block_start
            for fixed_bits, block_start, block_size in subcubes
            for subcube_point in range(2**block_size)
        ]
        value_by_point = dict(zip(points, boolean_function(points)))

        sum_of_subcube_distances = 0.0
        for fixed_bits, block_start, block_size in subcubes:
            # Create a boolean function on the subcube where the block's coordinates vary and the rest stay fixed.
            # Uses the pre-computed lookup table, rather than calling the boolean function again.
            restricted_function = lambda subcube_point: value_by_point[
                fixed_bits | subcube_point << block_start
            ]
            sum_of_subcube_distances += exact_distance_to_monotonicity(
                restricted_function, block_size
            )
        total_wall_clock_seconds = time.perf_counter() - start_seconds
        mean_subcube_distance = sum_of_subcube_distances / sample_count

        return DistanceEstimate(
            dimension=dimension,
            estimated_distance=max(0.0, mean_subcube_distance - additive_error / 3),
            approximation_factor=self.approximation_factor(dimension),
            additive_error=additive_error,
            sample_count=sample_count,
            mean_subcube_distance=mean_subcube_distance,
            total_wall_clock_seconds=total_wall_clock_seconds,
            cost=cost_ledger.calls_since(first_call_index) if cost_ledger else None,
        )
