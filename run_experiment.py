"""Test an LLM hiring screener for monotonicity. Calls the LLM (costs money).

python run_experiment.py --dimension 10
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from monotonicity_testing.edge_tester import EdgeTester
from monotonicity_testing.fattal_ron_estimator import FattalRonEstimator
from monotonicity_testing.screening_client import (
    DEFAULT_DEPLOYMENT_NAME,
    LLMResumeScreener,
)
from monotonicity_testing.skills_catalog import SkillsCatalog

RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"

# Epsilon parameter. Edge tester catches screeners at least this far from monotone (with probability >= 1 - 1/e).
DISTANCE_THRESHOLD = 0.05
# Delta parameter of Fattal-Ron, estimate is within this additive error, like a +-5% survey margin of error.
ADDITIVE_ERROR = 0.05
# Fattal-Ron checks 2^SUBCUBE_DIMENSION resumes per sample.
SUBCUBE_DIMENSION = 2

RANDOM_SEED = "6eeac86f2322c9eb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dimension", type=int, default=35, help="dimension n of the boolean hypercube"
    )
    parser.add_argument("--deployment-name", default=DEFAULT_DEPLOYMENT_NAME)
    arguments = parser.parse_args()

    skills_catalog = SkillsCatalog.load_from_file(dimension=arguments.dimension)
    screening_client = LLMResumeScreener(deployment_name=arguments.deployment_name)
    screener_accepts = screening_client.as_boolean_function(
        skills_catalog.render_skills_section
    )
    random_generator = random.Random(RANDOM_SEED)

    test_result = EdgeTester().run_test(
        screener_accepts, arguments.dimension, DISTANCE_THRESHOLD, random_generator
    )
    distance_estimate = FattalRonEstimator(
        subcube_dimension=SUBCUBE_DIMENSION
    ).estimate_distance(
        screener_accepts, arguments.dimension, ADDITIVE_ERROR, random_generator
    )

    results = json.dumps(
        {
            "arguments": vars(arguments),
            "stretch_skill_names": [
                skill.name for skill in skills_catalog.stretch_skills
            ],
            "test_result": test_result.model_dump(mode="json"),
            "distance_estimate": distance_estimate.model_dump(mode="json"),
            "api_call_count": len(screening_client.cost_ledger.api_calls),
            "cost": screening_client.cost_ledger.model_dump(
                mode="json", exclude={"api_calls"}
            ),
        },
        indent=2,
    )
    print(results)
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    (RESULTS_DIRECTORY / f"run_{datetime.now():%Y%m%d_%H%M%S}.json").write_text(results)


if __name__ == "__main__":
    main()
