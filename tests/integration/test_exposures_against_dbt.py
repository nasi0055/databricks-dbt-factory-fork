"""Power BI refresh tasks generated from exposures in manifests that dbt itself produced.

Each test writes a dbt project, has dbt parse it, and runs the factory over the real `manifest.json`, so
the exposure shape (where dbt files `meta`, how it resolves `depends_on`) is dbt's rather than ours.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from databricks_dbt_factory.dbt_task import PowerBiRefreshOptions
from databricks_dbt_factory.main import main
from tests.conftest import create_dbt_factory
from tests.integration.test_selector_against_dbt import MODEL_SQL, _parse, _write_project

DEV_WORKSPACE = "9dd59ee2-821e-45c5-8ed5-ca79ef80dd94"
DEV_DATASET = "ac27854a-9dcb-4fd3-ae69-5aaa9a48cf96"
PROD_WORKSPACE = "11111111-2222-3333-4444-555555555555"
PROD_DATASET = "66666666-7777-8888-9999-000000000000"

SCHEMA_YML = f"""\
version: 2
sources:
  - name: raw
    tables:
      - name: t
        columns:
          - name: id
            data_tests: [not_null]
models:
  - name: base
    columns:
      - name: id
        data_tests: [unique]
  - name: rpt
    columns:
      - name: id
        data_tests: [not_null]
exposures:
  - name: exp_report
    type: dashboard
    owner: {{email: owner@example.com}}
    depends_on: ["ref('rpt')", "source('raw', 't')"]
    config:
      meta:
        dataset:
          dev: {{key: {DEV_DATASET}, workspace: {DEV_WORKSPACE}}}
          prod: {{key: {PROD_DATASET}, workspace: {PROD_WORKSPACE}}}
  - name: exp_top_level_meta
    type: dashboard
    owner: {{email: owner@example.com}}
    depends_on: ["ref('base')"]
    meta:
      dataset:
        prod: {{key: {PROD_DATASET}, workspace: {PROD_WORKSPACE}}}
  - name: exp_not_powerbi
    type: application
    owner: {{email: owner@example.com}}
    depends_on: ["ref('rpt')"]
"""


@pytest.fixture(name="manifest", scope="module")
def manifest_fixture(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exposures")
    _write_project(
        root,
        {"base.sql": MODEL_SQL, "rpt.sql": "select id from {{ ref('base') }}\n"},
        schema_yml=SCHEMA_YML,
    )
    return _parse(root)


def _powerbi_options(environment: str) -> PowerBiRefreshOptions:
    return PowerBiRefreshOptions(environment=environment, secret_scope="powerbi", notebook_path="./refresh.py")


def _tasks_by_key(manifest: dict, environment: str = "prod", **factory_options) -> dict[str, dict]:
    factory = create_dbt_factory(powerbi_options=_powerbi_options(environment), **factory_options)
    return {task["task_key"]: task for task in factory.create_tasks(manifest)}


def _upstream_closure(tasks: dict[str, dict], task_key: str) -> set[str]:
    """Every task that must finish before `task_key` starts."""
    seen: set[str] = set()
    pending = [dep["task_key"] for dep in tasks[task_key]["depends_on"]]
    while pending:
        key = pending.pop()
        if key not in seen:
            seen.add(key)
            pending.extend(dep["task_key"] for dep in tasks[key]["depends_on"])
    return seen


def test_exposure_with_a_dataset_becomes_a_refresh_task_for_the_environment(manifest):
    tasks = _tasks_by_key(manifest, environment="prod")

    notebook_task = tasks["exp_report_exposure"]["notebook_task"]
    assert notebook_task == {
        "notebook_path": "./refresh.py",
        "base_parameters": {"workspace_id": PROD_WORKSPACE, "dataset_id": PROD_DATASET, "secret_scope": "powerbi"},
        "source": "WORKSPACE",
    }
    assert tasks["exp_report_exposure"]["environment_key"] == "Default"
    assert "exp_not_powerbi_exposure" not in tasks


def test_top_level_meta_is_read_as_dbt_moves_it_into_config(manifest):
    tasks = _tasks_by_key(manifest, environment="prod")

    parameters = tasks["exp_top_level_meta_exposure"]["notebook_task"]["base_parameters"]
    assert parameters["workspace_id"] == PROD_WORKSPACE
    assert parameters["dataset_id"] == PROD_DATASET


def test_per_test_refresh_waits_for_every_resource_and_test_its_data_depends_on(manifest):
    tasks = _tasks_by_key(manifest, environment="prod")

    assert [dep["task_key"] for dep in tasks["exp_report_exposure"]["depends_on"]] == [
        "rpt_model",
        "not_null_rpt_id_test",
        "source_not_null_raw_t_id_test",
    ]
    assert _upstream_closure(tasks, "exp_report_exposure") == {
        "base_model",
        "rpt_model",
        "unique_base_id_test",
        "not_null_rpt_id_test",
        "source_not_null_raw_t_id_test",
    }


def test_bundled_refresh_waits_for_the_test_tasks_of_its_resources(manifest):
    tasks = _tasks_by_key(manifest, environment="prod", bundle_tests=True)

    assert [dep["task_key"] for dep in tasks["exp_report_exposure"]["depends_on"]] == ["raw_t_test", "rpt_test"]
    assert _upstream_closure(tasks, "exp_report_exposure") == {
        "base_model",
        "base_test",
        "rpt_model",
        "rpt_test",
        "raw_t_test",
    }


def test_refresh_without_tests_waits_for_its_models(manifest):
    tasks = _tasks_by_key(manifest, environment="prod", run_tests=False)

    assert [dep["task_key"] for dep in tasks["exp_report_exposure"]["depends_on"]] == ["rpt_model"]
    assert [dep["task_key"] for dep in tasks["exp_top_level_meta_exposure"]["depends_on"]] == ["base_model"]


def test_exposure_without_an_entry_for_the_environment_is_refused(manifest):
    factory = create_dbt_factory(powerbi_options=_powerbi_options("dev"))

    with pytest.raises(ValueError, match=r"exp_top_level_meta.*no 'dev' entry"):
        factory.create_tasks(manifest)


def test_exposures_are_ignored_without_powerbi_options(manifest):
    tasks = create_dbt_factory().create_tasks(manifest)

    assert not [task for task in tasks if task["task_key"].endswith("_exposure")]


def _run_cli(tmp_path: Path, manifest: dict, *extra_args: str, monkeypatch) -> tuple[Path, dict]:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    template = tmp_path / "template.yaml"
    template.write_text(
        yaml.safe_dump({"resources": {"jobs": {"job": {"name": "job", "tasks": []}}}}), encoding="utf-8"
    )
    spec_dir = tmp_path / "resources"
    spec_dir.mkdir()
    spec = spec_dir / "job.yaml"
    monkeypatch.setattr(
        "sys.argv",
        [
            "databricks_dbt_factory",
            "--dbt-manifest-path",
            str(manifest_path),
            "--input-job-spec-path",
            str(template),
            "--target-job-spec-path",
            str(spec),
            "--target",
            "prod",
            "--powerbi-secret-scope",
            "powerbi",
            *extra_args,
        ],
    )
    main()
    return spec_dir, yaml.safe_load(spec.read_text(encoding="utf-8"))


@pytest.mark.parametrize("task_type", ["notebook", "dbt"])
def test_cli_publishes_the_refresh_notebook_next_to_the_spec(tmp_path, manifest, monkeypatch, task_type):
    spec_dir, spec = _run_cli(tmp_path, manifest, "--task-type", task_type, monkeypatch=monkeypatch)

    notebook_source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "databricks_dbt_factory"
        / "notebook"
        / "refresh_powerbi_dataset.py"
    ).read_bytes()
    published = spec_dir / f"refresh_powerbi_dataset_{hashlib.sha256(notebook_source).hexdigest()}.py"
    assert published.read_bytes() == notebook_source

    tasks = {task["task_key"]: task for task in spec["resources"]["jobs"]["job"]["tasks"]}
    notebook_task = tasks["exp_report_exposure"]["notebook_task"]
    assert notebook_task["notebook_path"] == f"./{published.name}"
    assert notebook_task["source"] == "WORKSPACE"
    assert notebook_task["base_parameters"]["secret_scope"] == "powerbi"


def test_generated_refresh_tasks_are_valid_bundle_resources(tmp_path, manifest, monkeypatch, bundle_validator):
    _, spec = _run_cli(tmp_path, manifest, "--bundle-tests", monkeypatch=monkeypatch)

    errors = sorted(bundle_validator.iter_errors(spec), key=lambda error: list(error.path))
    assert not errors, "\n".join(f"{list(error.path)}: {error.message}" for error in errors)
