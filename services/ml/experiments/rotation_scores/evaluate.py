"""Offline sensitivity experiment. Synthetic scenarios are not agronomic validation.

Run from services/ml: uv run python experiments/rotation_scores/evaluate.py --out <file>
"""

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

from agrofarm_ml.rotations import engine, rules
from agrofarm_ml.rotations.traits import load_traits

CONSTANTS = (
    "LEGUME_AFTER_CEREAL_NITROGEN_BONUS",
    "LEGUME_AFTER_CEREAL_SOIL_BONUS",
    "BARE_FALLOW_EROSION_PENALTY",
    "DRY_WATER_FIT_SLOPE",
    "WET_EROSION_SLOPE",
    "WARMING_WATER_SLOPE",
)


def evaluate() -> dict[str, Any]:
    """Compare four rule strengths over the same independently specified input grid."""
    equal = dict.fromkeys(rules.PRIORITIES, 1.0)
    profiles = {"equal": (equal, {})}
    for priority in rules.PRIORITIES:
        profiles[priority] = (dict.fromkeys(rules.PRIORITIES, 0.0) | {priority: 1.0}, {})
    profiles["water_limited"] = (equal, {"water_limited": True})
    profiles["avoid_beans"] = (equal, {"avoid_crops": ["black_beans"]})
    scenarios = [
        {"rainfall_change_pct": rain, "temperature_change_c": temp, "enso_phase": phase}
        for rain, temp, phase in itertools.product(
            (-80.0, -30.0, 0.0, 50.0, 100.0),
            (-5.0, 0.0, 8.0),
            ("neutral", "el_nino", "la_nina"),
        )
    ]
    original = {name: getattr(rules, name) for name in CONSTANTS}
    baseline_winners: list[set[str]] = []
    iterations = []
    # Evaluate the deployed constants first so comparisons have a stable reference.
    for multiplier in (1.0, 0.0, 0.5, 1.5):
        winners: Counter[str] = Counter()
        changed = 0
        bounded = True
        overlap = 0
        example = []
        constants = {name: value * multiplier for name, value in original.items()}
        with patch.multiple(rules, **constants):
            for index, ((weights, constraints), scenario) in enumerate(
                itertools.product(profiles.values(), scenarios)
            ):
                output = engine.score(["maize", "fallow"], weights, constraints, scenario)
                best = max(s["overall_score"] for s in output)
                tied = {s["id"] for s in output if s["overall_score"] == best}
                for strategy_id in tied:
                    winners[strategy_id] += 1
                if multiplier == 1.0:
                    baseline_winners.append(tied)
                elif tied != baseline_winners[index]:
                    changed += 1
                for strategy in output:
                    low, high = strategy["uncertainty"].values()
                    bounded &= 0 <= low <= strategy["overall_score"] <= high <= 1
                    bounded &= all(0 <= v <= 1 for v in strategy["scores"].values())
                ordered = sorted(output, key=lambda s: s["overall_score"], reverse=True)
                overlap += ordered[0]["uncertainty"]["low"] <= ordered[1]["uncertainty"]["high"]
            calm = {
                "rainfall_change_pct": 0.0,
                "temperature_change_c": 0.0,
                "enso_phase": "neutral",
            }
            example = [
                {"id": s["id"], "score": s["overall_score"], "range": s["uncertainty"]}
                for s in engine.score(["maize", "fallow"], equal, {}, calm)
            ]
        iterations.append(
            {
                "multiplier": multiplier,
                "constants": constants,
                "cases": len(scenarios) * len(profiles),
                "bounded": bounded,
                "changed_winner_sets_vs_current": changed,
                "winner_counts_including_ties": dict(sorted(winners.items())),
                "top_two_ranges_overlap_cases": overlap,
                "equal_neutral_example": example,
            }
        )
    source_files = [Path(engine.__file__), Path(rules.__file__)]
    source_files += sorted((Path(engine.__file__).parent / "data").glob("*.json"))
    return {
        "generated_by": "OpenAI Codex (gpt-6.1-sol)",
        "reviewed_by": None,
        "scope": "Synthetic sensitivity and software consistency, not field calibration",
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "traits_version": load_traits().version,
        "scenario_count": len(scenarios),
        "profile_names": list(profiles),
        "iterations": iterations,
        "selected_multiplier": 1.0,
        "selection_reason": "Retain current provisional constants: no field target supports "
        "optimizing their magnitude. Boundedness and stability cannot establish "
        "agronomic accuracy.",
        "fixed_assumptions": "Crop traits, ENSO offsets, avoidance penalty, water multiplier "
        "and interval constants are held fixed; this experiment does not validate them.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(evaluate(), indent=2, allow_nan=False) + "\n")
