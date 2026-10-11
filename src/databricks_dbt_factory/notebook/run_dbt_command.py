# Databricks notebook source

import json
import os
import re
import shlex
import shutil
import tempfile
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from dbt.cli.main import dbtRunner

# COMMAND ----------

dbutils.widgets.text("dbt_commands", "")
dbutils.widgets.text("project_directory", "")
dbutils.widgets.text("profiles_directory", "")
# Optional JSON object of environment variables to set while dbt runs, e.g. for `env_var()` in
# profiles.yml. Usually supplied as a job parameter, which Databricks pushes down to every task.
dbutils.widgets.text("env_vars", "")

dbt_commands = dbutils.widgets.get("dbt_commands")
project_directory = dbutils.widgets.get("project_directory")
profiles_directory = dbutils.widgets.get("profiles_directory")
env_vars_parameter = dbutils.widgets.get("env_vars")

if not dbt_commands:
    raise ValueError("dbt_commands parameter is required")

# The runner sets these itself; letting env_vars replace them would break authentication or the
# per-task artifact isolation below.
RUNNER_ENV_VARS = ("DBT_HOST", "DBT_ACCESS_TOKEN", "DBT_TARGET_PATH", "DBT_LOG_PATH")
ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

env_vars: dict[str, str] = {}
if env_vars_parameter:
    try:
        env_vars = json.loads(env_vars_parameter)
    except json.JSONDecodeError as e:
        raise ValueError(f"env_vars parameter is not valid JSON: {e}") from e
    if not isinstance(env_vars, dict):
        raise ValueError("env_vars parameter must be a JSON object of variable names to string values")
    for _name, _value in env_vars.items():
        if not ENV_VAR_NAME.fullmatch(_name):
            raise ValueError(f"env_vars parameter has an invalid variable name: {_name!r}")
        if not isinstance(_value, str):
            raise ValueError(f"env_vars parameter value for {_name} must be a string")
        if _name in RUNNER_ENV_VARS:
            raise ValueError(f"env_vars parameter cannot set {_name}; the runner sets it")

# COMMAND ----------

# Authenticate dbt with the notebook's own credentials via the Databricks SDK. The captured token
# is short-lived (about an hour) and does not self-refresh, so a single task running longer than its
# lifetime can fail mid-run.
ws = WorkspaceClient()
os.environ["DBT_ACCESS_TOKEN"] = ws.config.authenticate()["Authorization"].removeprefix("Bearer ").strip()

# dbt's host must be a bare hostname, but `config.host` is a full URL (scheme + host, and potentially
# a trailing slash or path). Parse it and keep only the netloc so a value like
# "https://my-workspace.databricks.com/" yields "my-workspace.databricks.com".
_parsed = urlparse(ws.config.host)
os.environ["DBT_HOST"] = _parsed.netloc or _parsed.path.strip("/")

# chdir to the dbt project so dbt runs from inside it. Relative `project_directory` is
# resolved against this notebook's own workspace location — the same anchor native
# `dbt_task` uses. Auto-copy mode sends `.` (resolves to the notebook's dir, which is
# project root by construction); user-pinned `--notebook-path` with relative
# `--project-directory` resolves against wherever the user placed the notebook; absolute
# `project_directory` is used as-is.
if project_directory:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_dir = os.path.dirname("/Workspace" + ctx.notebookPath().get())
    target_dir = (
        project_directory
        if os.path.isabs(project_directory)
        else os.path.normpath(os.path.join(notebook_dir, project_directory))
    )
    os.chdir(target_dir)

# Point dbt's artifact and log output at a private per-task dir so the parallel tasks of a job
# don't contend on the shared workspace `target/`/`logs/`.
local_dir = tempfile.mkdtemp(prefix="dbt_local_")
os.environ["DBT_TARGET_PATH"] = local_dir
os.environ["DBT_LOG_PATH"] = local_dir


# If a pre-built msgpack sits next to the project, deserialize it into a manifest and inject it into
# dbtRunner to skip dbt's parse phase (re-reading/hashing every file + DAG rebuild) on each task.
# Falls back to a normal parse if the msgpack is absent or unusable.
manifest = None
prebuilt_manifest_path = os.path.join("target", "partial_parse.msgpack")
if os.path.exists(prebuilt_manifest_path):
    try:
        from dbt.contracts.graph.manifest import Manifest

        with open(prebuilt_manifest_path, "rb") as f:
            manifest = Manifest.from_msgpack(f.read())
        manifest.build_flat_graph()
        print(f"[dbt-factory] injecting pre-built manifest from {prebuilt_manifest_path} (skipping dbt parse)")
    except Exception as e:
        print(f"[dbt-factory] manifest injection unavailable, falling back to dbt parse: {e}")
        manifest = None

# Set env_vars last, remembering any values they replace so the finally block can restore them.
previous_env_vars = {name: os.environ.get(name) for name in env_vars}
os.environ.update(env_vars)

try:
    runner = dbtRunner(manifest=manifest)

    for command_str in json.loads(dbt_commands):
        command_str = command_str.strip()
        if not command_str:
            continue

        if command_str.startswith("dbt "):
            command_str = command_str[4:]

        args = shlex.split(command_str)

        if profiles_directory:
            args.extend(["--profiles-dir", profiles_directory])

        print(f"Running: dbt {' '.join(args)}")
        print("-" * 60)

        result = runner.invoke(args)

        if not result.success:
            detail = result.exception or result.result or "(no further details)"
            raise RuntimeError(f"dbt command failed: dbt {' '.join(args)}\n{detail}")

        print(f"Completed successfully: dbt {' '.join(args)}")
finally:
    os.environ.pop("DBT_ACCESS_TOKEN", None)
    os.environ.pop("DBT_HOST", None)
    os.environ.pop("DBT_TARGET_PATH", None)
    os.environ.pop("DBT_LOG_PATH", None)
    for name, previous in previous_env_vars.items():
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    # Remove the private per-task target/log dir; on reused (all-purpose) clusters these
    # would otherwise accumulate under the system temp dir for the life of the cluster.
    if local_dir:
        shutil.rmtree(local_dir, ignore_errors=True)
