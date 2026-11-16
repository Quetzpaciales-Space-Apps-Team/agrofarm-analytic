"""Cloud Storage REST upload behavior tested without Google Cloud credentials."""

import base64
import hashlib
import json
from pathlib import Path
from typing import Self
from urllib.error import URLError

import pytest

from agrofarm_data import gcs_storage
from agrofarm_data.gcs_storage import GcsStorage, GcsStorageError, HttpResponse


def test_cloud_run_metadata_token_is_requested_with_metadata_header() -> None:
    def request(method: str, url: str, headers: dict[str, str], body: bytes | None) -> HttpResponse:
        assert method == "GET"
        assert url.endswith("/default/token")
        assert headers == {"Metadata-Flavor": "Google"}
        assert body is None
        return 200, b'{"access_token":"short-lived-token"}'

    storage = GcsStorage(request=request)
    assert storage.token_provider() == "short-lived-token"


def test_cloud_run_metadata_errors_are_reported() -> None:
    with pytest.raises(GcsStorageError, match="HTTP 403"):
        GcsStorage(request=lambda *_: (403, b"{}")).token_provider()
    with pytest.raises(GcsStorageError, match="invalid token"):
        GcsStorage(request=lambda *_: (200, b"{}")).token_provider()


def test_http_request_reads_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 200

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"ok"

    monkeypatch.setattr(gcs_storage, "urlopen", lambda *_args, **_kwargs: Response())
    assert gcs_storage._http_request("GET", "https://storage.googleapis.com", {}, None) == (
        200,
        b"ok",
    )


def test_http_request_wraps_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr(gcs_storage, "urlopen", fail)
    with pytest.raises(GcsStorageError, match="HTTP request failed"):
        gcs_storage._http_request("GET", "https://storage.googleapis.com", {}, None)


def test_raw_object_upload_uses_create_only_and_skips_identical_retries(
    tmp_path: Path,
) -> None:
    file = tmp_path / "sample.tif"
    file.write_bytes(b"raw raster")
    requests: list[tuple[str, str, str]] = []
    digest = base64.b64encode(hashlib.md5(b"raw raster", usedforsecurity=False).digest()).decode()

    def request(method: str, url: str, headers: dict[str, str], body: bytes | None) -> HttpResponse:
        requests.append((method, url, headers["Authorization"]))
        if method == "POST":
            assert "ifGenerationMatch=0" in url
            assert body == b"raw raster"
            return 412, b"{}"
        return 200, json.dumps({"md5Hash": digest}).encode()

    storage = GcsStorage(token_provider=lambda: "test-token", request=request)
    storage.upload_directory("raw-bucket", "raw/climate", tmp_path, immutable=True)

    assert [entry[0] for entry in requests] == ["POST", "GET"]
    assert all(entry[2] == "Bearer test-token" for entry in requests)


def test_raw_object_change_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "raw.bin").write_bytes(b"different")

    def request(
        method: str, _url: str, _headers: dict[str, str], _body: bytes | None
    ) -> HttpResponse:
        if method == "POST":
            return 412, b"{}"
        return 200, json.dumps({"md5Hash": "not-the-same"}).encode()

    storage = GcsStorage(token_provider=lambda: "token", request=request)
    with pytest.raises(GcsStorageError, match="(?i)immutable"):
        storage.upload_directory("raw-bucket", "raw", tmp_path, immutable=True)


def test_clean_objects_are_replaceable(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    urls: list[str] = []

    def request(
        method: str, url: str, _headers: dict[str, str], _body: bytes | None
    ) -> HttpResponse:
        urls.append(url)
        assert method == "POST"
        return 200, b"{}"

    storage = GcsStorage(token_provider=lambda: "token", request=request)
    storage.upload_directory("clean-bucket", "clean/dataset", tmp_path, immutable=False)
    assert len(urls) == 1
    assert "clean%2Fdataset%2Fmanifest.json" in urls[0]
    assert "ifGenerationMatch" not in urls[0]


def test_storage_rejects_empty_bucket(tmp_path: Path) -> None:
    storage = GcsStorage(token_provider=lambda: "token", request=lambda *_: (200, b"{}"))
    with pytest.raises(GcsStorageError, match="bucket"):
        storage.upload_directory("", "clean", tmp_path, immutable=False)


def test_non_retryable_upload_failure_is_reported(tmp_path: Path) -> None:
    (tmp_path / "sample.bin").write_bytes(b"data")
    storage = GcsStorage(token_provider=lambda: "token", request=lambda *_: (400, b"{}"))
    with pytest.raises(GcsStorageError, match="HTTP 400"):
        storage.upload_directory("clean-bucket", "clean", tmp_path, immutable=False)
