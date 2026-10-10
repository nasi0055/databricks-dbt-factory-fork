import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from databricks_dbt_factory.utils import DYNAMIC_VALUE_REFERENCE


_MAX_NOTEBOOK_BASE_PARAMETERS_BYTES = 1_000_000

# Power BI workspace (group) and dataset ids are GUIDs.
POWERBI_ID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _compute_spec(
    task_key: str, depends_on: list[str] | None, environment_key: str, job_cluster_key: str | None
) -> dict[str, Any]:
    """The fields every generated task shares: its key, its upstream tasks and its compute."""
    spec: dict[str, Any] = {
        "task_key": task_key,
        "depends_on": [{"task_key": dep} for dep in (depends_on or [])],
    }
    if job_cluster_key:
        spec["job_cluster_key"] = job_cluster_key
    else:
        spec["environment_key"] = environment_key
    return spec


class TaskType(Enum):
    """Supported task types for generated Databricks tasks."""

    DBT = "dbt"
    NOTEBOOK = "notebook"


@dataclass(frozen=True)
class DbtTaskOptions:
    environment_key: str = "Default"  # serverless env
    """The key of an environment. It has to be unique within a job."""

    catalog: str | None = None
    """Optional name of the catalog to use. The value is the top level in the 3-level namespace of
    Unity Catalog (catalog / schema / relation). The catalog value can only be specified if a
    warehouse_id is specified. Requires dbt-databricks >= 1.1.1."""

    profiles_directory: str | None = None
    """Optional (relative) path to the profiles directory. Can only be specified if no warehouse_id is
    specified. If no warehouse_id is specified and this folder is unset, the root directory is used."""

    project_directory: str | None = None
    """Path to the project directory. Optional for Git sourced tasks, in which case if no value is
    provided, the root of the Git repository is used."""

    schema: str | None = None
    """Optional schema to write to. This parameter is only used when a warehouse_id is also provided.
    If not provided, the `default` schema is used."""

    source: str | None = None
    """Optional location type of the project directory. When set to `WORKSPACE`, the project will be
    retrieved from the local Databricks workspace. When set to `GIT`, the project will be retrieved
    from a Git repository defined in `git_source`. If the value is empty, the task will use `GIT` if
    `git_source` is defined and `WORKSPACE` otherwise.
    
    * `WORKSPACE`: Project is located in Databricks workspace. * `GIT`: Project is located in cloud
    Git provider."""

    warehouse_id: str | None = None
    """ID of the SQL warehouse to connect to. If provided, we automatically generate and provide the
    profile and connection details to dbt. It can be overridden on a per-command basis by using the
    `--profiles-dir` command line argument."""

    dbt_deps_enabled: bool = False
    """Optional flag to enable dbt deps to be run before each task. Defaults to False."""

    dbt_tasks_deps: list[str] = field(default_factory=list)
    """Optional comma separated list of tasks that requires dbt debs. Only in effect if dbt_deps_enabled is enabled."""

    task_type: TaskType = TaskType.NOTEBOOK
    """Task type to generate: `TaskType.NOTEBOOK` for a notebook_task wrapper (default),
    `TaskType.DBT` for a native dbt_task. Strings are accepted and coerced — any value outside the
    enum raises `ValueError`. Notebook mode enables base environment support and environment
    variables on serverless."""

    notebook_path: str | None = None
    """Path to the dbt runner notebook. Required when task_type is `TaskType.NOTEBOOK`."""

    job_cluster_key: str | None = None
    """Job cluster key for running tasks on job compute instead of serverless."""

    def __post_init__(self):
        if not isinstance(self.task_type, TaskType):
            object.__setattr__(self, "task_type", TaskType(self.task_type))
        if self.task_type is TaskType.NOTEBOOK:
            if not self.notebook_path:
                raise ValueError("notebook_path is required when task_type=NOTEBOOK.")
            unsupported = []
            for name, value in (
                ("warehouse_id", self.warehouse_id),
                ("schema", self.schema),
                ("catalog", self.catalog),
            ):
                if value:
                    unsupported.append(name)
            if unsupported:
                raise ValueError(
                    f"{', '.join(unsupported)} cannot be set with task_type=NOTEBOOK; "
                    "notebook tasks connect via profiles.yml."
                )


@dataclass(frozen=True)
class DbtTask:
    """Represents a dbt task in the Databricks job definition."""

    task_key: str
    commands: list[str]
    options: DbtTaskOptions
    depends_on: list[str] | None = None

    def to_dict(self) -> dict:
        """Converts the Task to a dictionary suitable for the job definition."""
        if self.options.task_type is TaskType.NOTEBOOK:
            return self._to_notebook_dict()
        return self._to_dbt_dict()

    def _base_spec(self) -> dict[str, Any]:
        return _compute_spec(self.task_key, self.depends_on, self.options.environment_key, self.options.job_cluster_key)

    def _to_dbt_dict(self) -> dict[str, Any]:
        spec = self._base_spec()
        dbt_task: dict[str, Any] = {"commands": self.commands}

        if self.options.source:
            dbt_task["source"] = self.options.source
        if self.options.project_directory:
            dbt_task["project_directory"] = self.options.project_directory
        if self.options.schema:
            dbt_task["schema"] = self.options.schema
        if self.options.warehouse_id:
            dbt_task["warehouse_id"] = self.options.warehouse_id
        if self.options.catalog:
            dbt_task["catalog"] = self.options.catalog
        if self.options.profiles_directory:
            dbt_task["profiles_directory"] = self.options.profiles_directory

        spec["dbt_task"] = dbt_task
        return spec

    def _to_notebook_dict(self) -> dict[str, Any]:
        serialized_commands = json.dumps(self.commands)
        if DYNAMIC_VALUE_REFERENCE.search(serialized_commands):
            raise ValueError(
                f"Notebook task {self.task_key!r} serialized dbt_commands forms a Databricks dynamic value "
                f"reference; rename the selected resources or change the dbt command options."
            )
        base_parameters: dict[str, str] = {
            "dbt_commands": serialized_commands,
        }
        if self.options.project_directory:
            base_parameters["project_directory"] = self.options.project_directory
        if self.options.profiles_directory:
            base_parameters["profiles_directory"] = self.options.profiles_directory

        serialized_size = len(json.dumps(base_parameters, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        if serialized_size > _MAX_NOTEBOOK_BASE_PARAMETERS_BYTES:
            raise ValueError(
                f"Notebook task {self.task_key!r} serializes base_parameters to {serialized_size:,} bytes; "
                f"Databricks allows at most {_MAX_NOTEBOOK_BASE_PARAMETERS_BYTES:,} bytes. Shorten its dbt "
                f"commands or path options, reduce the tests in this bundle, or generate native dbt tasks "
                f"with --task-type dbt."
            )

        notebook_task: dict[str, Any] = {
            "notebook_path": self.options.notebook_path,
            "base_parameters": base_parameters,
        }
        if self.options.source:
            notebook_task["source"] = self.options.source

        spec = self._base_spec()
        spec["notebook_task"] = notebook_task
        return spec


@dataclass(frozen=True)
class PowerBiRefreshOptions:
    """Settings shared by every Power BI dataset refresh task generated from a dbt exposure."""

    environment: str
    """Key under an exposure's `meta.dataset` that names the dataset to refresh — normally the dbt target."""

    secret_scope: str
    """Databricks secret scope holding the Microsoft Entra ID service principal the refresh notebook
    authenticates with, under the keys `powerbi-tenant-id`, `powerbi-client-id` and `powerbi-client-secret`."""

    notebook_path: str
    """Path to the Power BI refresh notebook."""

    source: str | None = "WORKSPACE"
    """Location type of the notebook (`WORKSPACE` or `GIT`). The packaged notebook is deployed with the
    bundle, so it is `WORKSPACE` unless the caller manages the notebook in the job's Git source."""

    environment_key: str = "Default"
    """The key of the serverless environment the task runs in, when `job_cluster_key` is unset."""

    job_cluster_key: str | None = None
    """Job cluster key for running the task on job compute instead of serverless."""

    def __post_init__(self):
        for name in ("environment", "secret_scope", "notebook_path"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"Power BI refresh {name} must be nonempty.")
            if DYNAMIC_VALUE_REFERENCE.search(value):
                raise ValueError(f"Power BI refresh {name} {value!r} contains a Databricks dynamic value reference.")


@dataclass(frozen=True)
class PowerBiRefreshTask:
    """A notebook task that refreshes one Power BI dataset and waits for the refresh to finish."""

    task_key: str
    workspace_id: str
    dataset_id: str
    options: PowerBiRefreshOptions
    depends_on: list[str] | None = None

    def __post_init__(self):
        for name in ("workspace_id", "dataset_id"):
            if not POWERBI_ID.fullmatch(getattr(self, name)):
                raise ValueError(f"Power BI {name} {getattr(self, name)!r} for task {self.task_key!r} is not a GUID.")

    def to_dict(self) -> dict[str, Any]:
        """Converts the task to a dictionary suitable for the job definition."""
        notebook_task: dict[str, Any] = {
            "notebook_path": self.options.notebook_path,
            "base_parameters": {
                "workspace_id": self.workspace_id,
                "dataset_id": self.dataset_id,
                "secret_scope": self.options.secret_scope,
            },
        }
        if self.options.source:
            notebook_task["source"] = self.options.source

        spec = _compute_spec(self.task_key, self.depends_on, self.options.environment_key, self.options.job_cluster_key)
        spec["notebook_task"] = notebook_task
        return spec
