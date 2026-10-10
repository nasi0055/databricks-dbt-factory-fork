import pytest

from databricks_dbt_factory.dbt_task import PowerBiRefreshOptions, PowerBiRefreshTask
from databricks_dbt_factory.main import parse_args
from databricks_dbt_factory.utils import build_task_key_maps, generate_task_key
from tests.conftest import create_dbt_factory

WORKSPACE = "9dd59ee2-821e-45c5-8ed5-ca79ef80dd94"
DATASET = "ac27854a-9dcb-4fd3-ae69-5aaa9a48cf96"


def _options(**overrides) -> PowerBiRefreshOptions:
    values = {"environment": "prod", "secret_scope": "powerbi", "notebook_path": "./refresh.py", **overrides}
    return PowerBiRefreshOptions(**values)


def test_refresh_task_runs_the_notebook_on_serverless_by_default():
    task = PowerBiRefreshTask("sales_exposure", WORKSPACE, DATASET, _options(), ["sales_model"])

    assert task.to_dict() == {
        "task_key": "sales_exposure",
        "depends_on": [{"task_key": "sales_model"}],
        "environment_key": "Default",
        "notebook_task": {
            "notebook_path": "./refresh.py",
            "base_parameters": {"workspace_id": WORKSPACE, "dataset_id": DATASET, "secret_scope": "powerbi"},
            "source": "WORKSPACE",
        },
    }


def test_refresh_task_uses_the_job_cluster_when_given():
    task = PowerBiRefreshTask("sales_exposure", WORKSPACE, DATASET, _options(job_cluster_key="dbt_cluster"))

    result = task.to_dict()

    assert result["job_cluster_key"] == "dbt_cluster"
    assert "environment_key" not in result


@pytest.mark.parametrize("field", ["workspace_id", "dataset_id"])
def test_refresh_task_refuses_an_id_that_is_not_a_guid(field):
    ids = {"workspace_id": WORKSPACE, "dataset_id": DATASET, field: "{{job.id}}"}

    with pytest.raises(ValueError, match=f"{field}.*not a GUID"):
        PowerBiRefreshTask("sales_exposure", options=_options(), **ids)


@pytest.mark.parametrize("field", ["environment", "secret_scope", "notebook_path"])
@pytest.mark.parametrize("value", ["", "{{job.parameters.scope}}"])
def test_refresh_options_refuse_empty_or_dynamic_values(field, value):
    with pytest.raises(ValueError, match=field):
        _options(**{field: value})


def test_exposure_task_key_is_suffixed_with_its_type():
    assert generate_task_key("exposure.shop.sales_report") == "sales_report_exposure"


def test_exposure_task_keys_from_two_packages_are_disambiguated():
    task_keys, _ = build_task_key_maps(["exposure.a.report", "exposure.b.report", "model.a.report"])

    assert task_keys == {
        "exposure.a.report": "a_report_exposure",
        "exposure.b.report": "b_report_exposure",
        "model.a.report": "report_model",
    }


def _manifest(exposure_meta: dict, depends_on: list[str] | None = None) -> dict:
    model = {
        "unique_id": "model.shop.sales",
        "resource_type": "model",
        "name": "sales",
        "package_name": "shop",
        "fqn": ["shop", "sales"],
        "original_file_path": "models/sales.sql",
        "depends_on": {"nodes": []},
        "config": {"enabled": True},
    }
    exposure = {
        "unique_id": "exposure.shop.report",
        "resource_type": "exposure",
        "name": "report",
        "package_name": "shop",
        "original_file_path": "models/exposures.yml",
        "depends_on": {"nodes": depends_on if depends_on is not None else ["model.shop.sales"]},
        "config": {"enabled": True, "meta": exposure_meta},
    }
    return {"nodes": {"model.shop.sales": model}, "exposures": {"exposure.shop.report": exposure}}


def _dataset(**entry) -> dict:
    return {"dataset": {"prod": {"workspace": WORKSPACE, "key": DATASET, **entry}}}


@pytest.mark.parametrize(
    "meta",
    [
        {"dataset": None},
        {"dataset": {"prod": None}},
        {"dataset": {"dev": {"workspace": WORKSPACE, "key": DATASET}}},
    ],
)
def test_dataset_without_a_usable_environment_entry_is_refused(meta):
    factory = create_dbt_factory(powerbi_options=_options())

    with pytest.raises(ValueError, match=r"'report'.*no 'prod' entry"):
        factory.create_tasks(_manifest(meta))


@pytest.mark.parametrize(
    ("entry", "field"),
    [
        ({"workspace": None}, "workspace"),
        ({"key": "My Sales Dataset"}, "key"),
        ({"key": 42}, "key"),
    ],
)
def test_dataset_ids_that_are_not_guids_are_refused(entry, field):
    factory = create_dbt_factory(powerbi_options=_options())

    with pytest.raises(ValueError, match=rf"'report'.*meta.dataset.prod.{field} .*not a Power BI GUID"):
        factory.create_tasks(_manifest(_dataset(**entry)))


def test_exposure_on_a_resource_no_task_builds_is_refused():
    factory = create_dbt_factory(powerbi_options=_options())

    with pytest.raises(ValueError, match=r"'report'.*depends on 'metric.shop.revenue', which no generated task"):
        factory.create_tasks(_manifest(_dataset(), depends_on=["model.shop.sales", "metric.shop.revenue"]))


def test_exposure_on_a_resource_type_without_a_factory_is_refused():
    manifest = _manifest(_dataset())
    seed = {**manifest["nodes"]["model.shop.sales"], "unique_id": "seed.shop.rates", "resource_type": "seed"}
    manifest["nodes"]["seed.shop.rates"] = {**seed, "name": "rates", "fqn": ["shop", "rates"]}
    manifest["exposures"]["exposure.shop.report"]["depends_on"]["nodes"] = ["seed.shop.rates"]
    factory = create_dbt_factory(powerbi_options=_options())
    del factory.task_factories["seed"]

    with pytest.raises(ValueError, match=r"depends on 'seed.shop.rates', which no generated task builds"):
        factory.create_tasks(manifest)


def test_exposure_without_a_dataset_gets_no_task():
    factory = create_dbt_factory(powerbi_options=_options())

    tasks = factory.create_tasks(_manifest({"owner_team": "growth"}))

    assert [task["task_key"] for task in tasks] == ["sales_model"]


def test_disabled_exposure_gets_no_task():
    manifest = _manifest(_dataset())
    manifest["exposures"]["exposure.shop.report"]["config"]["enabled"] = False
    factory = create_dbt_factory(powerbi_options=_options())

    assert [task["task_key"] for task in factory.create_tasks(manifest)] == ["sales_model"]


def _parse(monkeypatch, *args: str):
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--dbt-manifest-path",
            "m.json",
            "--input-job-spec-path",
            "i.yml",
            "--target-job-spec-path",
            "o.yml",
            *args,
        ],
    )
    return parse_args()


def test_powerbi_secret_scope_requires_a_target(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--powerbi-secret-scope", "powerbi")

    assert "--powerbi-secret-scope requires --target" in capsys.readouterr().err


def test_powerbi_secret_scope_must_be_nonempty(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--powerbi-secret-scope", "", "--target", "prod")

    assert "nonempty secret scope" in capsys.readouterr().err


def test_powerbi_refresh_is_off_by_default(monkeypatch):
    assert _parse(monkeypatch).powerbi_secret_scope is None
