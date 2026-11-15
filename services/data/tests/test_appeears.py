"""AppEEARS connector (DS-030). Catalog and submit responses are recorded from the real API on
2026-11-15 (`tests/fixtures/appeears/`); nothing here touches the network."""

import json
from pathlib import Path
from typing import Any

import pytest

from agrofarm_data.connectors.appeears import (
    SPECS,
    AppEearsError,
    TaskPending,
    build_request,
    confirm_layers,
    credentials_from_env,
    login,
    submit_or_resume,
)

FIXTURES = Path(__file__).parent / "fixtures" / "appeears"
BOX = (-90.65, 14.45, -90.40, 14.75)


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeTransport:
    """Scripted answers by (method, path). Records every call."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, Any, str | None, Any]] = []

    def json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        token: str | None = None,
        basic: tuple[str, str] | None = None,
    ) -> Any:
        self.calls.append((method, path, body, token, basic))
        value = self.routes[(method, path)]
        if isinstance(value, Exception):
            raise value
        return value

    def download(self, path: str, token: str, destination: Path) -> None:
        raise AssertionError("not used here")


def catalog() -> FakeTransport:
    routes: dict[tuple[str, str], Any] = {("GET", "/product"): fixture("products.json")}
    for product in ("SPL4SMGP.008", "MOD13Q1.061", "VNP13A1.002"):
        routes[("GET", f"/product/{product}")] = fixture(f"layers_{product}.json")
    return FakeTransport(routes)


# --- products and layers are confirmed against the catalog --------------------------------------


def test_the_three_sources_name_the_products_confirmed_on_2026_11_15() -> None:
    assert {spec.product for spec in SPECS.values()} == {
        "SPL4SMGP.008",
        "MOD13Q1.061",
        "VNP13A1.002",
    }
    # SMAP L4 is version 008. Version 007, which the docs once named, is not in the catalog.
    assert SPECS["smap-l4"].product == "SPL4SMGP.008"


@pytest.mark.parametrize("key", sorted(SPECS))
def test_confirm_layers_returns_fill_scale_units_and_valid_range_of_each_layer(key: str) -> None:
    spec = SPECS[key]
    confirmed = confirm_layers(catalog(), spec)
    assert set(confirmed) == {layer.name for layer in spec.layers}
    for meta in confirmed.values():
        assert {"units", "fill_value", "scale_factor", "valid_min", "valid_max"} <= set(meta)


def test_confirmed_metadata_matches_the_catalog_values() -> None:
    ndvi = confirm_layers(catalog(), SPECS["modis-13q1"])["_250m_16_days_NDVI"]
    assert ndvi["fill_value"] == -3000
    assert ndvi["scale_factor"] == pytest.approx(0.0001)
    smap = confirm_layers(catalog(), SPECS["smap-l4"])["Geophysical_Data_sm_surface"]
    assert smap["units"] == "m^3/m^3"
    assert smap["fill_value"] == -9999.0
    assert smap["scale_factor"] is None  # the catalog has an empty string: no scaling


def test_a_product_missing_from_the_catalog_is_refused_and_named() -> None:
    transport = catalog()
    transport.routes[("GET", "/product")] = [
        p for p in fixture("products.json") if p["ProductAndVersion"] != "VNP13A1.002"
    ]
    with pytest.raises(AppEearsError, match=r"VNP13A1\.002"):
        confirm_layers(transport, SPECS["viirs-vnp13a1"])


def test_a_layer_missing_from_the_product_is_refused_and_named() -> None:
    transport = catalog()
    layers = fixture("layers_MOD13Q1.061.json")
    del layers["_250m_16_days_EVI"]
    transport.routes[("GET", "/product/MOD13Q1.061")] = layers
    with pytest.raises(AppEearsError, match="_250m_16_days_EVI"):
        confirm_layers(transport, SPECS["modis-13q1"])


# --- the request ---------------------------------------------------------------------------------


def test_the_request_is_an_area_task_with_a_closed_polygon_and_us_style_dates() -> None:
    request = build_request(SPECS["modis-13q1"], BOX, "2025-06-01", "2025-06-30", "name")
    assert request["task_type"] == "area"
    assert request["params"]["dates"] == [{"startDate": "06-01-2025", "endDate": "06-30-2025"}]
    ring = request["params"]["geo"]["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert ring[0] == [-90.65, 14.45] and [-90.4, 14.75] in ring
    assert request["params"]["output"] == {
        "format": {"type": "geotiff"},
        "projection": "geographic",
    }
    assert [(x["product"], x["layer"]) for x in request["params"]["layers"]] == [
        ("MOD13Q1.061", layer.name) for layer in SPECS["modis-13q1"].layers
    ]


@pytest.mark.parametrize(
    ("box", "start", "end"),
    [
        ((-90.4, 14.45, -90.65, 14.75), "2025-06-01", "2025-06-30"),  # west east swapped
        ((-90.65, 14.75, -90.4, 14.45), "2025-06-01", "2025-06-30"),  # south north swapped
        (BOX, "2025-06-30", "2025-06-01"),
        (BOX, "06-01-2025", "2025-06-30"),
    ],
)
def test_a_bad_box_or_date_range_is_rejected_before_any_call(
    box: tuple[float, float, float, float], start: str, end: str
) -> None:
    with pytest.raises(AppEearsError):
        build_request(SPECS["modis-13q1"], box, start, end, "name")


# --- credentials and login -----------------------------------------------------------------------


def test_credentials_come_from_the_environment_and_a_missing_one_is_named_without_values() -> None:
    assert credentials_from_env({"EARTHDATA_USERNAME": "u", "EARTHDATA_PASSWORD": "p"}) == (
        "u",
        "p",
    )
    with pytest.raises(AppEearsError, match="EARTHDATA_PASSWORD") as error:
        credentials_from_env({"EARTHDATA_USERNAME": "secret-user"})
    assert "secret-user" not in str(error.value)


def test_login_sends_basic_credentials_once_and_returns_the_session_token() -> None:
    transport = FakeTransport(
        {("POST", "/login"): {"token": "session-token", "token_type": "Bearer"}}
    )
    assert login(transport, "user", "pass") == "session-token"
    assert transport.calls == [("POST", "/login", None, None, ("user", "pass"))]


def test_a_failed_login_does_not_leak_the_password() -> None:
    transport = FakeTransport({("POST", "/login"): AppEearsError("HTTP 401 on POST /login")})
    with pytest.raises(AppEearsError) as error:
        login(transport, "user", "hunter2-password")
    assert "hunter2-password" not in str(error.value)


# --- the task is submitted once, resumed and never blocks ------------------------------------------


def make_request() -> dict[str, Any]:
    return build_request(SPECS["modis-13q1"], BOX, "2025-06-01", "2025-06-30", "agrofarm-test")


def test_the_first_call_submits_and_reports_the_task_as_pending(tmp_path: Path) -> None:
    submitted = fixture("task_submit_response.json")
    transport = FakeTransport(
        {
            ("POST", "/task"): submitted,
            ("GET", f"/task/{submitted['task_id']}"): {"status": "pending"},
        }
    )
    with pytest.raises(TaskPending) as pending:
        submit_or_resume(transport, "token", make_request(), tmp_path)
    assert pending.value.task_id == submitted["task_id"]
    assert [c[:2] for c in transport.calls].count(("POST", "/task")) == 1
    assert list(tmp_path.glob("*.json")), "the task id is stored so the job can resume"


def test_the_same_request_resumes_the_stored_task_and_never_submits_twice(tmp_path: Path) -> None:
    submitted = fixture("task_submit_response.json")
    task_path = ("GET", f"/task/{submitted['task_id']}")
    first = FakeTransport({("POST", "/task"): submitted, task_path: {"status": "processing"}})
    with pytest.raises(TaskPending):
        submit_or_resume(first, "token", make_request(), tmp_path)

    again = FakeTransport({task_path: {"status": "done"}})
    assert submit_or_resume(again, "token", make_request(), tmp_path) == submitted["task_id"]
    assert [c[:2] for c in again.calls] == [task_path]  # polled once, no POST


def test_a_different_request_is_a_different_task(tmp_path: Path) -> None:
    submitted = fixture("task_submit_response.json")
    other = {"task_id": "11111111-2222-3333-4444-555555555555", "status": "pending"}
    routes: dict[tuple[str, str], Any] = {
        ("POST", "/task"): submitted,
        ("GET", f"/task/{submitted['task_id']}"): {"status": "pending"},
        ("GET", f"/task/{other['task_id']}"): {"status": "pending"},
    }
    with pytest.raises(TaskPending):
        submit_or_resume(FakeTransport(routes), "token", make_request(), tmp_path)
    routes[("POST", "/task")] = other
    changed = build_request(SPECS["modis-13q1"], BOX, "2025-07-01", "2025-07-30", "agrofarm-test")
    with pytest.raises(TaskPending) as pending:
        submit_or_resume(FakeTransport(routes), "token", changed, tmp_path)
    assert pending.value.task_id == other["task_id"]


def test_an_errored_task_is_reported_with_its_id_and_a_new_run_submits_again(
    tmp_path: Path,
) -> None:
    submitted = fixture("task_submit_response.json")
    task_path = ("GET", f"/task/{submitted['task_id']}")
    transport = FakeTransport({("POST", "/task"): submitted, task_path: {"status": "error"}})
    with pytest.raises(AppEearsError, match=submitted["task_id"]):
        submit_or_resume(transport, "token", make_request(), tmp_path)
    assert not list(tmp_path.glob("*.json")), "a failed task must not be resumed forever"


def test_the_stored_state_never_holds_a_token_or_credentials(tmp_path: Path) -> None:
    submitted = fixture("task_submit_response.json")
    transport = FakeTransport(
        {
            ("POST", "/task"): submitted,
            ("GET", f"/task/{submitted['task_id']}"): {"status": "pending"},
        }
    )
    with pytest.raises(TaskPending):
        submit_or_resume(transport, "a-very-secret-token", make_request(), tmp_path)
    stored = "".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert "a-very-secret-token" not in stored
    assert submitted["task_id"] in stored
