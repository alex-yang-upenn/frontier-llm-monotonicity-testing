"""Exact distance to monotonicity of a small subcube restriction (step 3 of Fattal-Ron Algorithm 1).

Fattal & Ron, Lemma 2.1: the distance equals (minimum vertex cover of the violation graph) / 2^dimension,
where the violation graph joins every pair lower_point < upper_point with f(lower_point) = 1 and
f(upper_point) = 0. That graph is bipartite (lower points, where f = 1, vs upper points, where f = 0), so
by Konig's theorem the minimum vertex cover has the size of a maximum matching, which we find with augmenting paths.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Callable


def exact_distance_to_monotonicity(
    boolean_function: Callable[[int], bool], dimension: int
) -> float:
    """Given a (restricted) boolean function on a subcube of the hypercube, compute its exact distance to monotonicity by
    computing every output of the function and finding a maximum matching in the violation graph.

    Points are int bitmasks; lower_point < upper_point means lower_point's bits are a subset of upper_point's.
    """
    point_count = 2**dimension
    values = [boolean_function(point) for point in range(point_count)]

    # Any point where f = 0 can be the upper end of a violation
    upper_point_candidates = [
        bitmask for bitmask in range(point_count) if not values[bitmask]
    ]

    violation_graph = {
        lower_point: [
            upper_point
            for upper_point in upper_point_candidates
            if lower_point & upper_point == lower_point
        ]
        for lower_point in range(point_count)
        if values[lower_point]  # Any point where f = 1 can be the lower end
    }
    upper_point_matched_to: dict[int, int] = {}
    lower_point_matched_to: dict[int, int] = {}
    maximum_matching_size = sum(
        find_augmenting_path(
            lower_point,
            violation_graph,
            lower_point_matched_to,
            upper_point_matched_to,
        )
        for lower_point, neighbors in violation_graph.items()
        if neighbors
    )
    return maximum_matching_size / point_count


def find_augmenting_path(
    start_lower_point: int,
    violation_graph: Mapping[int, list[int]],
    lower_point_matched_to: dict[int, int],
    upper_point_matched_to: dict[int, int],
) -> bool:
    """Kuhn's algorithm step (iterative depth-first search). Grows the matching by one if possible."""
    reached_from: dict[int, int] = {}  # upper point -> lower point it was reached from
    lower_points_on_path = [start_lower_point]
    unexplored_neighbors: list[Iterator[int]] = [
        iter(violation_graph[start_lower_point])
    ]
    while unexplored_neighbors:
        for upper_point in unexplored_neighbors[-1]:
            if upper_point in reached_from:
                continue
            reached_from[upper_point] = lower_points_on_path[-1]
            if upper_point not in upper_point_matched_to:
                flip_path(
                    upper_point,
                    reached_from,
                    lower_point_matched_to,
                    upper_point_matched_to,
                )
                return True
            next_lower_point = upper_point_matched_to[upper_point]
            lower_points_on_path.append(next_lower_point)
            unexplored_neighbors.append(iter(violation_graph[next_lower_point]))
            break
        else:
            unexplored_neighbors.pop()
            lower_points_on_path.pop()
    return False


def flip_path(
    end_upper_point: int,
    reached_from: Mapping[int, int],
    lower_point_matched_to: dict[int, int],
    upper_point_matched_to: dict[int, int],
) -> None:
    upper_point: int | None = end_upper_point
    while upper_point is not None:
        lower_point = reached_from[upper_point]
        previously_matched_upper_point = lower_point_matched_to.get(lower_point)
        lower_point_matched_to[lower_point] = upper_point
        upper_point_matched_to[upper_point] = lower_point
        upper_point = previously_matched_upper_point
