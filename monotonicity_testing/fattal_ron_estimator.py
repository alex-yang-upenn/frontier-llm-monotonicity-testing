"""Fattal, Ron. "Approximating the Distance to Monotonicity in High Dimensions." ACM TALG, 2010. Algorithm 1.

Points of the hypercube are int bitmasks: bit i is coordinate i.

Split the coordinates into consecutive blocks of `subcube_dimension` coordinates (the last
block may be shorter). Repeat `sample_count` times: pick a random block, fix every other
coordinate at random, and compute the exact distance to monotonicity of the function on
that small subcube. Return the mean of those distances minus a small bias correction.

Guarantee (Lemma 3.2 with Lemma 3.1), with probability >= 2/3:
    true_distance / (2 * block_count) - additive_error <= estimate <= true_distance.

The paper's constants are used as is: ceil(18 / additive_error^2) samples, and a bias
correction of additive_error / 3. Subcube distances are computed exactly (`exact_distance.py`).

Each sample costs up to 2^subcube_dimension queries; the full dimension can be large.
"""

from __future__ import annotations

import math
import random
from typing import Annotated, Callable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    computed_field,
    validate_call,
)

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
        boolean_function: Callable[[int], bool],
        dimension: int,
        additive_error: float,
        random_generator: random.Random,
    ) -> DistanceEstimate:
        sample_count = self.sample_count(additive_error)
        sum_of_subcube_distances = 0.0
        for _ in range(sample_count):
            # Randomly choose a subcube of dimension `subcube_dimension`
            block_start = random_generator.randrange(
                0, dimension, self.subcube_dimension
            )
            block_size = min(self.subcube_dimension, dimension - block_start)
            block_bits = (2**block_size - 1) << block_start

            # Randomly fix the other coordinates of the hypercube
            fixed_bits = random_generator.getrandbits(dimension) & ~block_bits

            # Create a boolean function on the subcube where the block's coordinates vary and the rest stay fixed.
            restricted_function = lambda subcube_point: boolean_function(
                fixed_bits | subcube_point << block_start
            )
            sum_of_subcube_distances += exact_distance_to_monotonicity(
                restricted_function, block_size
            )
        mean_subcube_distance = sum_of_subcube_distances / sample_count

        return DistanceEstimate(
            dimension=dimension,
            estimated_distance=max(0.0, mean_subcube_distance - additive_error / 3),
            approximation_factor=self.approximation_factor(dimension),
            additive_error=additive_error,
            sample_count=sample_count,
            mean_subcube_distance=mean_subcube_distance,
        )
