"""The contract (docs/contract/openapi.yaml) is the source of truth for ml-service."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate

from agrofarm_ml.contract import load_contract
from agrofarm_ml.main import create_app

CONTRACT = load_contract()
FARM_ID = "6f1c2f0a-9a0e-4d53-8a55-0d6d7a1b2c33"
SCENARIO = {"rainfall_change_pct": -20, "temperature_change_c": 1.0, "enso_phase": "el_nino"}
SIMULATE = {
    "farm_id": FARM_ID,
    "centroid_lat": 14.59,
    "centroid_lon": -90.53,
    "crop": "maize",
    "scenario": SCENARIO,
}
ROTATIONS = {
    "farm_id": FARM_ID,
    "centroid_lat": 14.59,
    "centroid_lon": -90.53,
    "current_rotation": ["maize", "fallow"],
    "priorities": {"weights": {"erosion_reduction": 0.9}, "constraints": {}},
    "scenario": SCENARIO,
}


def resolve(node: Any) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            target: Any = CONTRACT
            for part in node["$ref"].removeprefix("#/").split("/"):
                target = target[part]
            return resolve(target)
        return {key: resolve(value) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve(item) for item in node]
    return node


def schema_of(name: str) -> Any:
    return resolve(CONTRACT["components"]["schemas"][name])


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_contract_is_a_valid_openapi_document() -> None:
    validate(CONTRACT)


def test_service_publishes_the_contract_file_unchanged(client: TestClient) -> None:
    assert client.get("/openapi.json").json() == CONTRACT


def test_contract_examples_match_their_schemas() -> None:
    checked = 0
    for item in CONTRACT["paths"].values():
        for operation in item.values():
            media = resolve(operation["responses"]["200"])["content"]["application/json"]
            if "example" in media:
                Draft202012Validator(media["schema"]).validate(media["example"])
                checked += 1
    assert checked >= 4


@pytest.mark.parametrize(
    ("path", "status"), [("/healthz", "ok"), ("/readyz", "ready")], ids=["healthz", "readyz"]
)
def test_probes_answer(client: TestClient, path: str, status: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.json() == {"status": status}


def test_models_lists_the_stub_and_says_it_is_a_stub(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    Draft202012Validator(schema_of("ModelList")).validate(body)
    assert [model["status"] for model in body["items"]] == ["stub"]


def test_simulate_matches_the_contract_and_carries_provenance(client: TestClient) -> None:
    response = client.post("/v1/simulate", json=SIMULATE)
    assert response.status_code == 200
    body = response.json()
    Draft202012Validator(schema_of("SimulateResponse")).validate(body)
    assert body["crop"] == "maize"
    assert body["provenance"]["model_version"] == "stub-0.1.0"
    assert "data_versions" in body["provenance"]
    for key in ("baseline", "scenario"):
        assert body[key]["p10"] <= body[key]["p50"] <= body[key]["p90"]


def test_simulate_is_deterministic_apart_from_the_timestamp(client: TestClient) -> None:
    first = client.post("/v1/simulate", json=SIMULATE).json()
    second = client.post("/v1/simulate", json=SIMULATE).json()
    first["provenance"].pop("generated_at")
    second["provenance"].pop("generated_at")
    assert first == second


def test_simulate_answers_the_contract_example(client: TestClient) -> None:
    operation = CONTRACT["paths"]["/v1/simulate"]["post"]
    request = operation["requestBody"]["content"]["application/json"]["example"]
    expected = operation["responses"]["200"]["content"]["application/json"]["example"]
    body = client.post("/v1/simulate", json=request).json()
    body["provenance"]["generated_at"] = expected["provenance"]["generated_at"]
    assert body == expected


def test_simulate_scenario_moves_the_median_and_baseline_stays(client: TestClient) -> None:
    calm = {"rainfall_change_pct": 0, "temperature_change_c": 0, "enso_phase": "neutral"}
    neutral = {**SIMULATE, "scenario": calm}
    dry = client.post("/v1/simulate", json=SIMULATE).json()
    same = client.post("/v1/simulate", json=neutral).json()
    assert same["scenario"] == same["baseline"] == dry["baseline"]
    assert dry["scenario"]["p50"] < same["scenario"]["p50"]


def test_simulate_flags_inputs_far_from_normal(client: TestClient) -> None:
    extreme = {**SIMULATE, "scenario": {**SCENARIO, "rainfall_change_pct": -75}}
    body = client.post("/v1/simulate", json=extreme).json()
    assert body["out_of_range_inputs"] == ["rainfall_change_pct"]


@pytest.mark.parametrize(
    "broken",
    [
        {**SIMULATE, "scenario": {**SCENARIO, "enso_phase": "storm"}},
        {**SIMULATE, "scenario": {**SCENARIO, "rainfall_change_pct": -95}},
        {**SIMULATE, "centroid_lat": 120},
        {**SIMULATE, "farm_id": "not-a-uuid"},
        {key: value for key, value in SIMULATE.items() if key != "crop"},
    ],
    ids=["phase", "rainfall", "latitude", "farm_id", "missing_crop"],
)
def test_simulate_rejects_invalid_input(client: TestClient, broken: dict[str, Any]) -> None:
    response = client.post("/v1/simulate", json=broken)
    assert response.status_code == 422
    Draft202012Validator(schema_of("ValidationError")).validate(response.json())


def test_rotations_match_the_contract_with_three_strategies(client: TestClient) -> None:
    response = client.post("/v1/rotations/score", json=ROTATIONS)
    assert response.status_code == 200
    body = response.json()
    Draft202012Validator(schema_of("RotationScoreResponse")).validate(body)
    assert len(body["strategies"]) >= 3
    current = [s for s in body["strategies"] if s["is_current"]]
    assert len(current) == 1
    assert current[0]["sequence"] == ["maize", "fallow"]
    assert "does not replace" in body["disclaimer"]


def test_every_strategy_explains_itself_and_asks_for_local_review(client: TestClient) -> None:
    for strategy in client.post("/v1/rotations/score", json=ROTATIONS).json()["strategies"]:
        assert strategy["why"] and strategy["evidence"] and strategy["gaps"] and strategy["risks"]
        assert strategy["needs_local_review"] is True
        assert strategy["uncertainty"]["low"] <= strategy["overall_score"]
        assert strategy["overall_score"] <= strategy["uncertainty"]["high"]


def test_farmer_priorities_change_the_ranking(client: TestClient) -> None:
    def best(weights: dict[str, float]) -> str:
        request = {**ROTATIONS, "priorities": {"weights": weights, "constraints": {}}}
        strategies = client.post("/v1/rotations/score", json=request).json()["strategies"]
        return str(max(strategies, key=lambda s: s["overall_score"])["id"])

    zero = dict.fromkeys(
        (
            "water_use",
            "income",
            "soil_structure",
            "forage_value",
            "nitrogen_fixing",
            "erosion_reduction",
            "market_suitability",
        ),
        0.0,
    )
    assert best({**zero, "erosion_reduction": 1.0}) == "maize_cover_crop"
    assert best({**zero, "nitrogen_fixing": 1.0}) == "maize_beans"


def test_all_zero_weights_fall_back_to_equal_weights(client: TestClient) -> None:
    def scores(weights: dict[str, float]) -> list[float]:
        request = {**ROTATIONS, "priorities": {"weights": weights, "constraints": {}}}
        strategies = client.post("/v1/rotations/score", json=request).json()["strategies"]
        return [s["overall_score"] for s in strategies]

    zero = dict.fromkeys(
        (
            "water_use",
            "income",
            "soil_structure",
            "forage_value",
            "nitrogen_fixing",
            "erosion_reduction",
            "market_suitability",
        ),
        0.0,
    )
    assert scores(zero) == scores({})


def test_missing_current_rotation_is_reported_as_a_gap(client: TestClient) -> None:
    request = {**ROTATIONS, "current_rotation": []}
    body = client.post("/v1/rotations/score", json=request).json()
    Draft202012Validator(schema_of("RotationScoreResponse")).validate(body)
    current = next(s for s in body["strategies"] if s["is_current"])
    assert any("No current rotation" in gap for gap in current["gaps"])


def test_current_rotation_equal_to_an_alternative_is_not_listed_twice(client: TestClient) -> None:
    request = {**ROTATIONS, "current_rotation": ["maize", "black_beans"]}
    strategies = client.post("/v1/rotations/score", json=request).json()["strategies"]
    sequences = [tuple(s["sequence"]) for s in strategies]
    assert len(sequences) == len(set(sequences)) >= 3
