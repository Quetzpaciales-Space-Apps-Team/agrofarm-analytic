"""Training CLI: python -m agrofarm_ml.ndvi.training --table <parquet> --enso <parquet> --out <dir>

`--table` is the cell and year table written by data-service (`ndvi_cell_year`), `--enso` the
clean `enso_monthly` table. Output is a model directory with a model card.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from agrofarm_ml.ndvi.model import (
    FEATURES,
    MODEL_VERSION,
    QUANTILES,
    RULE,
    leave_one_year_out,
    train_model,
    verdict,
)
from agrofarm_ml.ndvi.table import COLUMNS, with_anomalies
from agrofarm_ml.splits import season_phase_by_year

LIMITS = [
    "Predicts NDVI, not yield. NDVI says how green the cropland got, not how much it produced.",
    "Experimental. It does not replace the fixed rules of /v1/simulate.",
    "Cells of about 5 km mix many fields and crops. It says nothing about one parcel.",
    "Rainfall of the same season is an input: this is a what-if tool, not a forecast.",
    "Rows of one year share the weather. About 25 years means few independent samples.",
    "Peak NDVI of dense vegetation saturates, so wet years are told apart poorly.",
    "The cropland map is from 2021 and is used for every year.",
    "Decision support only. A local agronomist must review any action taken from it.",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agrofarm_ml.ndvi.training")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--enso", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    table = pd.read_parquet(args.table).dropna(subset=COLUMNS).reset_index(drop=True)
    phases = season_phase_by_year(pd.read_parquet(args.enso))

    # Metrics come from models that never saw the year they predict.
    metrics = leave_one_year_out(table, phases, seed=args.seed)
    # The saved model is refit on every year. The metrics above describe the method.
    final = train_model(with_anomalies(table, table), seed=args.seed)
    final.save(args.out)

    card = {
        "model_version": MODEL_VERSION,
        "status": "experimental",
        "purpose": "What-if outlook of cropland greenness under a rainfall, temperature and "
        "ENSO scenario, with a range.",
        "target": "seasonal peak NDVI on cropland, as a percentage of the cell's own normal",
        "algorithm": "LightGBM gradient boosting, one quantile model per quantile",
        "quantiles": list(QUANTILES),
        "features": FEATURES,
        "seed": args.seed,
        "training_period": {"start": int(table["year"].min()), "end": int(table["year"].max())},
        "training_rows": int(len(table)),
        "training_cells": int(table["cell_id"].nunique()),
        "validation": {
            "split": "leave one whole year out",
            "benchmark": "climatology: the quantiles of the training anomalies for every row",
            "rule": RULE,
            "note": "Cell normals are computed without the held-out year. "
            "The saved model is refit on all years.",
        },
        "metrics": metrics,
        "verdict": verdict(metrics),
        "data_versions": {
            "table_sha256": _sha256(args.table),
            "enso_sha256": _sha256(args.enso),
        },
        "limits": LIMITS,
    }
    (args.out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in metrics.items() if key != "per_year"}
    sys.stdout.write(json.dumps({**summary, "verdict": card["verdict"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
