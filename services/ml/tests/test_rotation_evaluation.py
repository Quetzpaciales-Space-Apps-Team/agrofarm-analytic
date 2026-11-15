"""The sensitivity experiment must be reproducible and leave serving constants intact."""

import json
import runpy
from pathlib import Path

from agrofarm_ml.rotations import rules


def test_experiment_checks_all_cases_without_changing_serving_constants() -> None:
    experiment = runpy.run_path(
        str(Path(__file__).parents[1] / "experiments/rotation_scores/evaluate.py")
    )
    original = {name: getattr(rules, name) for name in experiment["CONSTANTS"]}
    report = experiment["evaluate"]()
    assert report == experiment["evaluate"]()
    artifact = Path(__file__).parents[1] / "experiments/rotation_scores/results.json"
    assert report == json.loads(artifact.read_text())
    assert {name: getattr(rules, name) for name in original} == original
    assert report["scenario_count"] == 45
    assert len(report["profile_names"]) == 10
    assert {i["multiplier"] for i in report["iterations"]} == {0.0, 0.5, 1.0, 1.5}
    assert all(i["cases"] == 450 and i["bounded"] for i in report["iterations"])


def test_weights_only_express_relative_importance() -> None:
    scores = dict(zip(rules.PRIORITIES, [0.1, 0.8, 0.3, 0.9, 0.2, 0.5, 0.7], strict=True))
    weights = dict(zip(rules.PRIORITIES, [0.0, 1.0, 0.3, 0.8, 0.1, 0.2, 0.4], strict=True))
    scaled = {name: value * 3 for name, value in weights.items()}
    assert abs(rules.overall_score(scores, weights) - rules.overall_score(scores, scaled)) < 1e-12
