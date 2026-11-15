"""Responses must satisfy what core-api expects (vendored copy of ml-service-expected.yaml).

The copy lives in tests/fixtures with the commit it came from in its first lines. Where the two
contracts differ on purpose, the differences are listed in test_known_differences.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from agrofarm_ml.contract import load_contract
from agrofarm_ml.main import create_app

EXPECTED: dict[str, Any] = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "ml-service-expected.yaml").read_text("utf-8")
)
FARM_ID = "6f1c2f0a-9a0e-4d53-8a55-0d6d7a1b2c33"
BODY = {
    "farm_id": FARM_ID,
    "centroid_lat": 14.59,
    "centroid_lon": -90.53,
    "current_rotation": ["maize", "fallow"],
    "priorities": {"weights": {"erosion_reduction": 0.9}, "constraints": {"water_limited": True}},
    "scenario": {"rainfall_change_pct": -20, "temperature_change_c": 1.0, "enso_phase": "el_nino"},
}


def resolve(node: Any) -> Any:
    if isinstance(node, dict):
        if "$ref" in node:
            target: Any = EXPECTED
            for part in node["$ref"].removeprefix("#/").split("/"):
                target = target[part]
            return resolve(target)
        return {key: resolve(value) for key, value in node.items()}
    if isinstance(node, list):
        return [resolve(item) for item in node]
    return node


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.parametrize(
    "current",
    [["maize", "fallow"], ["maize", "black_beans"], [], ["quinoa"], ["Corn", "beans", "cover"]],
)
def test_rotation_responses_validate_against_what_core_api_expects(
    client: TestClient, current: list[str]
) -> None:
    body = client.post("/v1/rotations/score", json={**BODY, "current_rotation": current}).json()
    schema = resolve(EXPECTED["components"]["schemas"]["RotationScoreResponse"])
    Draft202012Validator(schema).validate(body)


def test_simulate_response_still_validates_against_what_core_api_expects(
    client: TestClient,
) -> None:
    request = {**BODY, "crop": "maize"}
    request.pop("current_rotation")
    request.pop("priorities")
    body = client.post("/v1/simulate", json=request).json()
    Draft202012Validator(resolve(EXPECTED["components"]["schemas"]["SimulateResponse"])).validate(
        body
    )


def test_known_differences_between_the_two_contracts() -> None:
    """Not a failure: this records where ml-service is stricter or has more than core-api."""
    ours = load_contract()["components"]["schemas"]
    theirs = EXPECTED["components"]["schemas"]
    assert "model_name" in ours["Provenance"]["required"]
    assert "model_name" not in theirs["Provenance"]["required"]
    assert ours["SimulateRequest"]["properties"]["crop"].get("maxLength") == 60
    assert "maxLength" not in theirs["SimulateRequest"]["properties"]["crop"]
    assert "/v1/models" in load_contract()["paths"]
    assert "/v1/models" not in EXPECTED["paths"]
