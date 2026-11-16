"""CI gate for the baseline model: metrics on a small fixture against a stored baseline.

Used by tests only. It trains four tiny models on a committed synthetic fixture (seconds, fixed
seed) and fails when the pinball loss on the held-out years gets worse than the stored baseline
by more than the tolerance. Full training never runs in CI.

The fixture is synthetic, so skill on it says nothing about real data.

Refresh the baseline (needs a reason in the pull request):
    uv run python -m agrofarm_ml.gate --climate C.csv --enso E.csv --update baseline.json
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from agrofarm_ml.features import build_training_table
from agrofarm_ml.model import MODEL_VERSION, evaluate, train_model
from agrofarm_ml.splits import choose_holdout_years, season_phase_by_year, split_by_years

SEED = 0
TOLERANCE = 0.05  # a loss may be up to 5% worse than the baseline (platform noise)
GATED = ("pinball_p10", "pinball_p50", "pinball_p90", "pinball_mean")
NOTE = (
    "Synthetic fixture with a built-in ENSO effect on rainfall. Skill on it says nothing about "
    "real data: the model card verdict on the real station data stays no_skill_over_climatology."
)


def fixture_metrics(climate_csv: Path, enso_csv: Path, *, seed: int = SEED) -> dict[str, Any]:
    climate = pd.read_csv(climate_csv, parse_dates=["date"])
    enso = pd.read_csv(enso_csv, parse_dates=["month"])
    table = build_training_table(climate, enso)
    holdout = choose_holdout_years(season_phase_by_year(enso), years=sorted(set(table["year"])))
    train, test = split_by_years(table, holdout)
    return evaluate(train_model(train, seed=seed), train, test)


def check_against_baseline(
    metrics: dict[str, Any], baseline: dict[str, Any], tolerance: float | None = None
) -> list[str]:
    """Return one message per gated metric that is missing, NaN or worse than allowed."""
    allowed = baseline["tolerance"] if tolerance is None else tolerance
    problems = []
    for name in GATED:
        value = metrics.get("model", {}).get(name)
        if not isinstance(value, int | float) or math.isnan(value):
            problems.append(f"{name} is missing or NaN")
            continue
        limit = baseline["metrics"]["model"][name] * (1 + allowed)
        if value > limit:
            problems.append(f"{name} {value:.3f} is worse than the baseline limit {limit:.3f}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrofarm_ml.gate")
    parser.add_argument("--climate", type=Path, required=True)
    parser.add_argument("--enso", type=Path, required=True)
    parser.add_argument("--update", type=Path, required=True, help="Write the baseline here")
    args = parser.parse_args(argv)
    metrics = fixture_metrics(args.climate, args.enso)
    baseline = {
        "synthetic": True,
        "note": NOTE,
        "model_version": MODEL_VERSION,
        "seed": SEED,
        "tolerance": TOLERANCE,
        "holdout_years": metrics["holdout_years"],
        "metrics": {"model": metrics["model"], "climatology": metrics["climatology"]},
    }
    args.update.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
