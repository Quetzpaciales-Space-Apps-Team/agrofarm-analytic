"""Small Cloud Storage JSON API client using the Cloud Run identity token."""

import base64
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

HttpResponse = tuple[int, bytes]
HttpRequest = Callable[[str, str, dict[str, str], bytes | None], HttpResponse]
TokenProvider = Callable[[], str]


class GcsStorageError(RuntimeError):
    """Cloud Storage could not accept or verify an object upload."""


def _http_request(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> HttpResponse:
    request = Request(url, data=body, headers=headers, method=method)  # noqa: S310
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310
            return response.status, bytes(response.read())
    except HTTPError as exc:
        return exc.code, bytes(exc.read())
    except (URLError, TimeoutError) as exc:
        raise GcsStorageError("Cloud Storage HTTP request failed") from exc


def _metadata_token(request: HttpRequest = _http_request) -> str:
    """Get the job's short-lived identity token from the Cloud Run metadata server."""
    status, body = request(
        "GET",
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        {"Metadata-Flavor": "Google"},
        None,
    )
    if status != 200:
        raise GcsStorageError(f"Cloud Run metadata token request failed with HTTP {status}")
    try:
        token = json.loads(body)["access_token"]
    except (ValueError, KeyError, TypeError) as exc:
        raise GcsStorageError("Cloud Run metadata returned an invalid token response") from exc
    if not isinstance(token, str) or not token:
        raise GcsStorageError("Cloud Run metadata returned an empty token")
    return token


class GcsStorage:
    """Upload raw objects immutably and allow clean objects to be regenerated."""

    def __init__(
        self,
        token_provider: TokenProvider | None = None,
        request: HttpRequest = _http_request,
    ) -> None:
        self.request = request
        self.token_provider = token_provider or (lambda: _metadata_token(self.request))

    def upload_directory(
        self, bucket: str, prefix: str, directory: Path, *, immutable: bool
    ) -> None:
        if not bucket or "/" in bucket or not directory.is_dir():
            raise GcsStorageError("A valid bucket and existing local directory are required")
        token = self.token_provider()
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(directory).as_posix()
            object_name = f"{prefix.rstrip('/')}/{relative}".lstrip("/")
            self._upload_file(bucket, object_name, path, token, immutable)

    def _upload_file(self, bucket: str, name: str, path: Path, token: str, immutable: bool) -> None:
        query: dict[str, str] = {"uploadType": "media", "name": name}
        if immutable:
            query["ifGenerationMatch"] = "0"
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
        url = f"{url}?{urlencode(query, quote_via=quote, safe='')}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
        body = path.read_bytes()
        for attempt in range(3):
            try:
                status, _ = self.request("POST", url, headers, body)
            except GcsStorageError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if status in {200, 201}:
                return
            if immutable and status == 412:
                self._verify_existing(bucket, name, body, token)
                return
            if status not in {429, 500, 502, 503, 504} or attempt == 2:
                raise GcsStorageError(f"Upload failed for gs://{bucket}/{name} with HTTP {status}")
            time.sleep(2**attempt)

    def _verify_existing(self, bucket: str, name: str, body: bytes, token: str) -> None:
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/"
            f"{quote(name, safe='')}?fields=md5Hash"
        )
        status, response = self.request("GET", url, {"Authorization": f"Bearer {token}"}, None)
        if status != 200:
            raise GcsStorageError(
                f"Could not verify existing immutable object gs://{bucket}/{name}"
            )
        try:
            remote_md5 = json.loads(response)["md5Hash"]
        except (ValueError, KeyError, TypeError) as exc:
            raise GcsStorageError("Cloud Storage returned invalid object metadata") from exc
        local_md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode(
            "ascii"
        )
        if remote_md5 != local_md5:
            raise GcsStorageError(
                f"Immutable object gs://{bucket}/{name} already exists with different content"
            )
