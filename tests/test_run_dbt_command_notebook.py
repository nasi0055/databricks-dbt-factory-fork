"""Executes the packaged dbt runner notebook with a stubbed Databricks SDK and dbtRunner."""

import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import databricks.sdk
import dbt.cli.main
import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "src" / "databricks_dbt_factory" / "notebook" / "run_dbt_command.py"


class _Runner:
    """Records each dbt invocation together with the environment dbt saw."""

    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.invocations: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, manifest=None):  # dbtRunner(manifest=...) returns the runner
        return self

    def invoke(self, args: list[str]):
        self.invocations.append((list(args), dict(os.environ)))
        return SimpleNamespace(success=self.succeed, exception=None, result="dbt failed")


class _WorkspaceClient:
    def __init__(self):
        self.config = SimpleNamespace(
            host="https://my-workspace.cloud.databricks.com/",
            authenticate=lambda: {"Authorization": "Bearer notebook-token"},
        )


def _dbutils(parameters: dict[str, str]):
    defaults: dict[str, str] = {}

    def text(name: str, default: str) -> None:
        defaults[name] = default

    def get(name: str) -> str:
        return parameters.get(name, defaults[name])

    return SimpleNamespace(widgets=SimpleNamespace(text=text, get=get))


def _run(monkeypatch, runner: _Runner, **parameters: str) -> None:
    parameters = {"dbt_commands": json.dumps(["dbt run --select orders"]), **parameters}
    with monkeypatch.context() as patched:
        patched.setattr(databricks.sdk, "WorkspaceClient", _WorkspaceClient)
        patched.setattr(dbt.cli.main, "dbtRunner", runner)
        runpy.run_path(str(NOTEBOOK), init_globals={"dbutils": _dbutils(parameters)})


def test_env_vars_are_set_while_dbt_runs_and_restored_afterwards(monkeypatch):
    monkeypatch.delenv("DBT_HTTP_PATH", raising=False)
    monkeypatch.setenv("DBT_WAREHOUSE_SIZE", "small")
    runner = _Runner()

    _run(
        monkeypatch,
        runner,
        env_vars=json.dumps({"DBT_HTTP_PATH": "/sql/1.0/warehouses/abc", "DBT_WAREHOUSE_SIZE": "large"}),
    )

    assert len(runner.invocations) == 1
    seen_env = runner.invocations[0][1]
    assert seen_env["DBT_HTTP_PATH"] == "/sql/1.0/warehouses/abc"
    assert seen_env["DBT_WAREHOUSE_SIZE"] == "large"
    assert seen_env["DBT_HOST"] == "my-workspace.cloud.databricks.com"
    assert "DBT_HTTP_PATH" not in os.environ
    assert os.environ["DBT_WAREHOUSE_SIZE"] == "small"


def test_without_env_vars_dbt_sees_only_the_runner_variables(monkeypatch):
    monkeypatch.delenv("DBT_HTTP_PATH", raising=False)
    runner = _Runner()

    _run(monkeypatch, runner)

    assert len(runner.invocations) == 1
    seen_env = runner.invocations[0][1]
    assert "DBT_HTTP_PATH" not in seen_env
    assert seen_env["DBT_ACCESS_TOKEN"] == "notebook-token"


def test_env_vars_are_restored_when_a_dbt_command_fails(monkeypatch):
    monkeypatch.delenv("DBT_HTTP_PATH", raising=False)

    with pytest.raises(RuntimeError, match="dbt command failed"):
        _run(monkeypatch, _Runner(succeed=False), env_vars=json.dumps({"DBT_HTTP_PATH": "/sql/1.0/warehouses/abc"}))

    assert "DBT_HTTP_PATH" not in os.environ


@pytest.mark.parametrize(
    "env_vars",
    [
        "not json",
        json.dumps(["DBT_HTTP_PATH"]),
        json.dumps({"DBT_HTTP_PATH": 42}),
        json.dumps({"": "x"}),
        json.dumps({"BAD NAME": "x"}),
    ],
)
def test_unusable_env_vars_fail_before_dbt_runs(monkeypatch, env_vars):
    runner = _Runner()

    with pytest.raises(ValueError, match="env_vars"):
        _run(monkeypatch, runner, env_vars=env_vars)

    assert not runner.invocations


@pytest.mark.parametrize("name", ["DBT_HOST", "DBT_ACCESS_TOKEN", "DBT_TARGET_PATH", "DBT_LOG_PATH"])
def test_env_vars_cannot_override_the_runner_variables(monkeypatch, name):
    runner = _Runner()

    with pytest.raises(ValueError, match=f"env_vars.*{name}"):
        _run(monkeypatch, runner, env_vars=json.dumps({name: "x"}))

    assert not runner.invocations
