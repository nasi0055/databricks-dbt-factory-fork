"""Executes the packaged Power BI refresh notebook against a scripted Power BI and Entra ID."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "src" / "databricks_dbt_factory" / "notebook" / "refresh_powerbi_dataset.py"
)
WORKSPACE = "9dd59ee2-821e-45c5-8ed5-ca79ef80dd94"
DATASET = "ac27854a-9dcb-4fd3-ae69-5aaa9a48cf96"
DATASET_URL = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE}/datasets/{DATASET}/refreshes"
TOKEN_URL = "https://login.microsoftonline.com/tenant/oauth2/v2.0/token"
REQUEST_ID = "87f31ef7-1e3a-4006-9b0b-191693e79e9e"


class _Response:
    def __init__(self, status_code: int, body: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body or {}
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self.text = str(self._body)

    def json(self) -> dict:
        return self._body


class _PowerBi:
    """Serves the token, the refresh trigger and a scripted sequence of refresh-history responses."""

    def __init__(self, history: list[_Response], trigger: _Response | None = None):
        self.history = list(history)
        self.trigger = trigger or _Response(202, headers={"x-ms-request-id": REQUEST_ID})
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url: str, **kwargs) -> _Response:
        self.calls.append(("POST", url, kwargs))
        if url == TOKEN_URL:
            return _Response(200, {"access_token": "pbi-token", "expires_in": 3600})
        assert url == DATASET_URL
        return self.trigger

    def get(self, url: str, **kwargs) -> _Response:
        self.calls.append(("GET", url, kwargs))
        assert url == DATASET_URL
        return self.history.pop(0)


def _history(*statuses: str, request_id: str = REQUEST_ID) -> _Response:
    entries = [{"requestId": "older-refresh", "status": "Failed"}]
    for status in statuses:
        entries.append({"requestId": request_id, "status": status, "serviceExceptionJson": '{"errorCode":"Boom"}'})
    return _Response(200, {"value": entries})


def _dbutils(parameters: dict[str, str], secrets_read: list[tuple[str, str]]):
    values = {
        "workspace_id": WORKSPACE,
        "dataset_id": DATASET,
        "secret_scope": "powerbi",
        "poll_interval_seconds": "0",
        **parameters,
    }
    secrets = {"powerbi-tenant-id": "tenant", "powerbi-client-id": "client", "powerbi-client-secret": "secret"}

    def get_secret(scope: str, key: str) -> str:
        secrets_read.append((scope, key))
        return secrets[key]

    widgets = SimpleNamespace(text=lambda _name, _default: None, get=lambda name: values[name])
    return SimpleNamespace(widgets=widgets, secrets=SimpleNamespace(get=get_secret))


def _run(monkeypatch, power_bi: _PowerBi, **parameters: str) -> list[tuple[str, str]]:
    secrets_read: list[tuple[str, str]] = []
    with monkeypatch.context() as patched:
        patched.setattr(requests, "post", power_bi.post)
        patched.setattr(requests, "get", power_bi.get)
        runpy.run_path(str(NOTEBOOK), init_globals={"dbutils": _dbutils(parameters, secrets_read)})
    return secrets_read


def test_refresh_is_triggered_and_awaited_until_it_completes(monkeypatch):
    power_bi = _PowerBi([_history("Unknown"), _history("Completed")])

    secrets_read = _run(monkeypatch, power_bi)

    assert secrets_read == [
        ("powerbi", "powerbi-tenant-id"),
        ("powerbi", "powerbi-client-id"),
        ("powerbi", "powerbi-client-secret"),
    ]
    token_call, trigger_call, *poll_calls = power_bi.calls
    assert token_call[1] == TOKEN_URL
    assert token_call[2]["data"] == {
        "grant_type": "client_credentials",
        "client_id": "client",
        "client_secret": "secret",
        "scope": "https://analysis.windows.net/powerbi/api/.default",
    }
    assert trigger_call[:2] == ("POST", DATASET_URL)
    assert trigger_call[2]["json"] == {"notifyOption": "NoNotification"}
    assert trigger_call[2]["headers"] == {"Authorization": "Bearer pbi-token"}
    assert [call[0] for call in poll_calls] == ["GET", "GET"]
    assert not power_bi.history


@pytest.mark.parametrize("status", ["Failed", "Disabled", "Cancelled"])
def test_a_refresh_that_does_not_complete_fails_the_task(monkeypatch, status):
    power_bi = _PowerBi([_history("Unknown"), _history(status)])

    with pytest.raises(RuntimeError, match=rf"ended with status '{status}'.*Boom"):
        _run(monkeypatch, power_bi)


def test_a_refused_trigger_fails_the_task(monkeypatch):
    power_bi = _PowerBi([], trigger=_Response(400, {"error": {"code": "InvalidRequest"}}))

    with pytest.raises(RuntimeError, match="refused to refresh.*HTTP 400.*InvalidRequest"):
        _run(monkeypatch, power_bi)


def test_a_trigger_without_a_request_id_fails_the_task(monkeypatch):
    power_bi = _PowerBi([], trigger=_Response(202))

    with pytest.raises(RuntimeError, match="no request id"):
        _run(monkeypatch, power_bi)


def test_transient_poll_failures_are_retried(monkeypatch):
    power_bi = _PowerBi([_Response(503), _Response(429), _history("Completed")])

    _run(monkeypatch, power_bi)

    assert not power_bi.history


def test_persistent_poll_failures_fail_the_task(monkeypatch):
    power_bi = _PowerBi([_Response(503)] * 6)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        _run(monkeypatch, power_bi)


def test_a_refresh_missing_from_the_history_is_awaited(monkeypatch):
    power_bi = _PowerBi([_history(request_id="someone-else"), _history("Completed")])

    _run(monkeypatch, power_bi)

    assert not power_bi.history


@pytest.mark.parametrize("parameter", ["workspace_id", "dataset_id", "secret_scope"])
def test_a_missing_parameter_fails_before_calling_power_bi(monkeypatch, parameter):
    power_bi = _PowerBi([])

    with pytest.raises(ValueError, match=f"{parameter} parameter is required"):
        _run(monkeypatch, power_bi, **{parameter: ""})

    assert not power_bi.calls
