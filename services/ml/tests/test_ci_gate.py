"""ML-020 CI gate: train on a small committed fixture, compare with a stored baseline.

The fixture is SYNTHETIC (a made-up world with a strong ENSO effect), so skill on it says
nothing about the real data. The model card verdict on real data stays no_skill_over_climatology.
Full training never runs in CI: this trains four tiny models in about a second.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agrofarm_ml import gate
from agrofarm_ml.features import FEATURES, build_training_table
from agrofarm_ml.model import MODEL_VERSION, RainfallModel
from agrofarm_ml.training import main as train_main

FIXTURE = Path(__file__).parent / "fixtures" / "ml020"
CLIMATE, ENSO, BASELINE = FIXTURE / "climate.csv", FIXTURE / "enso.csv", FIXTURE / "baseline.json"


def test_fixture_is_small_and_says_it_is_synthetic() -> None:
    assert CLIMATE.stat().st_size < 400_000
    baseline = json.loads(BASELINE.read_text("utf-8"))
    assert baseline["synthetic"] is True
    assert "says nothing about real data" in baseline["note"]
    assert baseline["model_version"] == MODEL_VERSION
    assert baseline["tolerance"] <= 0.1


def test_pinball_loss_on_the_fixture_is_not_worse_than_the_stored_baseline() -> None:
    metrics = gate.fixture_metrics(CLIMATE, ENSO, seed=gate.SEED)
    baseline = json.loads(BASELINE.read_text("utf-8"))
    assert gate.check_against_baseline(metrics, baseline) == []


def test_the_gate_fails_when_the_loss_gets_worse_than_the_tolerance() -> None:  # canary
    baseline = json.loads(BASELINE.read_text("utf-8"))
    worse = copy.deepcopy(baseline["metrics"])
    worse["model"]["pinball_mean"] = baseline["metrics"]["model"]["pinball_mean"] * (
        1 + baseline["tolerance"] + 0.05
    )
    problems = gate.check_against_baseline(worse, baseline)
    assert problems and "pinball_mean" in problems[0]


def test_the_gate_accepts_small_noise_and_improvements() -> None:
    baseline = json.loads(BASELINE.read_text("utf-8"))
    close = copy.deepcopy(baseline["metrics"])
    close["model"]["pinball_p50"] *= 1 + baseline["tolerance"] / 2
    close["model"]["pinball_p10"] *= 0.5
    assert gate.check_against_baseline(close, baseline) == []


def test_the_gate_fails_on_nan_or_missing_metrics() -> None:
    baseline = json.loads(BASELINE.read_text("utf-8"))
    broken = copy.deepcopy(baseline["metrics"])
    broken["model"]["pinball_p90"] = float("nan")
    del broken["model"]["pinball_mean"]
    problems = gate.check_against_baseline(broken, baseline)
    assert any("pinball_p90" in p for p in problems)
    assert any("pinball_mean" in p for p in problems)


def test_the_baseline_can_be_regenerated_and_then_passes(tmp_path: Path) -> None:
    out = tmp_path / "baseline.json"
    assert gate.main(["--climate", str(CLIMATE), "--enso", str(ENSO), "--update", str(out)]) == 0
    fresh: dict[str, Any] = json.loads(out.read_text("utf-8"))
    assert fresh["metrics"]["model"]["pinball_mean"] > 0
    metrics = gate.fixture_metrics(CLIMATE, ENSO, seed=gate.SEED)
    assert gate.check_against_baseline(metrics, fresh) == []


def test_same_input_and_model_version_give_the_same_output(tmp_path: Path) -> None:
    climate, enso = pd.read_csv(CLIMATE, parse_dates=["date"]), pd.read_csv(ENSO, parse_dates=["month"])
    climate.to_parquet(tmp_path / "c.parquet", index=False)
    enso.to_parquet(tmp_path / "e.parquet", index=False)
    cards, predictions = [], []
    for run in ("a", "b"):
        out = tmp_path / run
        args = ["--climate", str(tmp_path / "c.parquet"), "--enso", str(tmp_path / "e.parquet")]
        assert train_main([*args, "--out", str(out), "--seed", "0"]) == 0
        cards.append(json.loads((out / "model_card.json").read_text("utf-8")))
        features = build_training_table(climate, enso)[FEATURES]
        predictions.append(RainfallModel.load(out).predict(features))
    assert cards[0] == cards[1]
    assert cards[0]["model_version"] == MODEL_VERSION
    pd.testing.assert_frame_equal(predictions[0], predictions[1])


def test_a_random_split_is_rejected_on_the_fixture() -> None:
    from agrofarm_ml.splits import SplitError, assert_whole_year_split

    table = build_training_table(
        pd.read_csv(CLIMATE, parse_dates=["date"]), pd.read_csv(ENSO, parse_dates=["month"])
    )
    shuffled = table.sample(frac=1.0, random_state=1)
    half = len(shuffled) // 2
    with pytest.raises(SplitError):
        assert_whole_year_split(shuffled.iloc[:half], shuffled.iloc[half:])
