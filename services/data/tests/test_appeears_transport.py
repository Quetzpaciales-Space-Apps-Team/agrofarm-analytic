"""The real transport, with the network replaced by scripted answers."""

import base64
import io
import json
import urllib.error
from http.client import HTTPMessage
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agrofarm_data.connectors.appeears import AppEearsError, UrllibTransport


class Answer(io.BytesIO):
    """A response body that works as a context manager, like the object urlopen returns."""

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


SESSION = "tok"  # a test value, not a credential


def http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.test", code, "x", HTTPMessage(), io.BytesIO(body.encode())
    )


def transport(sleeps: list[float] | None = None) -> UrllibTransport:
    return UrllibTransport(sleep=(sleeps if sleeps is not None else []).append)


def script(t: UrllibTransport, *items: Any) -> Any:
    return patch.object(t._opener, "open", side_effect=list(items))


def test_json_returns_the_parsed_answer_and_sends_the_token_as_a_bearer_header() -> None:
    t = transport()
    with script(t, Answer(b'{"status": "done"}')) as opened:
        assert t.json("GET", "/task/abc", token=SESSION) == {"status": "done"}
    request = opened.call_args.args[0]
    assert request.full_url.endswith("/api/task/abc")
    assert request.get_header("Authorization") == f"Bearer {SESSION}"


def test_a_call_without_a_token_sends_no_authorization_header() -> None:
    t = transport()
    with script(t, Answer(b"[]")) as opened:
        t.json("GET", "/product")
    assert opened.call_args.args[0].get_header("Authorization") is None


def test_a_body_is_sent_as_json_and_an_empty_post_has_an_empty_body() -> None:
    t = transport()
    with script(t, Answer(b"{}"), Answer(b"{}")) as opened:
        t.json("POST", "/task", {"a": 1}, token=SESSION)
        t.json("POST", "/login", basic=("user", "pass"))
    task, login = (call.args[0] for call in opened.call_args_list)
    assert json.loads(task.data) == {"a": 1}
    assert task.get_header("Content-type") == "application/json"
    assert login.data == b""


def test_login_credentials_travel_as_a_basic_header_and_nowhere_else() -> None:
    t = transport()
    with script(t, Answer(b"{}")) as opened:
        t.json("POST", "/login", basic=("user", "pass"))
    request = opened.call_args.args[0]
    header = request.get_header("Authorization")
    assert header == "Basic " + base64.b64encode(b"user:pass").decode()
    assert "pass" not in request.full_url


def test_a_client_error_is_not_retried_and_does_not_echo_the_body() -> None:
    t = transport()
    with (
        script(t, http_error(401, "the body mentions secret-account")) as opened,
        pytest.raises(AppEearsError) as error,
    ):
        t.json("POST", "/login", basic=("user", "pass"))
    assert str(error.value) == "HTTP 401 on POST /login"
    assert opened.call_count == 1


def test_a_server_error_is_retried_and_then_succeeds() -> None:
    sleeps: list[float] = []
    t = transport(sleeps)
    with script(t, http_error(503), urllib.error.URLError("down"), Answer(b'{"ok": true}')):
        assert t.json("GET", "/status") == {"ok": True}
    assert sleeps == [2.0, 4.0]


def test_the_retry_is_bounded() -> None:
    sleeps: list[float] = []
    t = transport(sleeps)
    with (
        script(t, http_error(500), http_error(502), http_error(504)) as opened,
        pytest.raises(AppEearsError, match="failed after 3 attempts"),
    ):
        t.json("GET", "/status")
    assert opened.call_count == 3 and sleeps == [2.0, 4.0]


def test_an_answer_that_is_not_json_is_an_error() -> None:
    t = transport()
    with (
        script(t, Answer(b"<html>not json</html>")),
        pytest.raises(AppEearsError, match="did not return JSON"),
    ):
        t.json("GET", "/product")


def test_download_writes_the_file_in_chunks_and_leaves_no_partial_file(tmp_path: Path) -> None:
    t = transport()
    payload = b"x" * (3 * (1 << 20) + 17)
    target = tmp_path / "deep" / "file.csv"
    with script(t, Answer(payload)):
        t.download("/bundle/a/b", SESSION, target)
    assert target.read_bytes() == payload
    assert not list(target.parent.glob("*.partial"))


def test_a_failed_download_leaves_no_file_behind(tmp_path: Path) -> None:
    t = transport()
    target = tmp_path / "file.csv"
    with script(t, http_error(404)), pytest.raises(AppEearsError):
        t.download("/bundle/a/b", SESSION, target)
    assert not target.exists()
