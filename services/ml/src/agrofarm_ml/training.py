"""Training CLI: python -m agrofarm_ml.training --climate <parquet> --enso <parquet> --out <dir>

Inputs are the clean tables written by data-service (`climate_insivumeh_daily` and
`enso_monthly`). Output is a model directory with a model card.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agrofarm_ml.features import FEATURES, build_training_table
from agrofarm_ml.model import MODEL_VERSION, QUANTILES, evaluate, train_model
from agrofarm_ml.splits import choose_holdout_years, season_phase_by_year, split_by_years

LIMITS = [
    "Predicts rainfall, not crop yield. No yield label exists per parcel.",
    "One station (INSIVUMEH, Guatemala City). Do not extrapolate to other regions.",
    "The source data is marked preliminary by the provider.",
    "Few independent samples (about 30 years). The uncertainty range is itself uncertain.",
    "Inputs outside the training range are flagged, and the answer there is not reliable.",
    "Decision support only. A local agronomist must review any action taken from it.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrofarm_ml.training")
    parser.add_argument("--climate", type=Path, required=True)
    parser.add_argument("--enso", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    enso = pd.read_parquet(args.enso)
    table = build_training_table(pd.read_parquet(args.climate), enso)
    phases = season_phase_by_year(enso)
    holdout = choose_holdout_years(phases, years=sorted(set(table["year"])))
    train, test = split_by_years(table, holdout)

    # Metrics come from a model that never saw the held-out years.
    metrics = evaluate(train_model(train, seed=args.seed), train, test)
    # The shipped model is refit on every year. The metrics above describe the method.
    final = train_model(table, seed=args.seed)
    final.save(args.out)

    card = {
        "model_version": MODEL_VERSION,
        "purpose": "Monthly rainfall outlook at the station under an ENSO state, with a range.",
        "target": "monthly rainfall total at the station (mm per month)",
        "algorithm": "LightGBM gradient boosting, one quantile model per quantile",
        "quantiles": list(QUANTILES),
        "features": FEATURES,
        "seed": args.seed,
        "training_period": {"start": int(table["year"].min()), "end": int(table["year"].max())},
        "validation": {
            "split": "whole years",
            "holdout_phases": {str(year): str(phases[year]) for year in holdout},
            "benchmark": "calendar-month climatology of the training years",
            "note": "Metrics are from a model fit without the held-out years. "
            "The saved model is refit on all years.",
        },
        "metrics": metrics,
        # Stated plainly so nobody presents the model as better than it is.
        "verdict": (
            "beats_climatology"
            if metrics["skill_vs_climatology"] > 0
            else "no_skill_over_climatology"
        ),
        "data_versions": {
            "climate_sha256": _sha256(args.climate),
            "enso_sha256": _sha256(args.enso),
        },
        "limits": LIMITS,
    }
    (args.out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
