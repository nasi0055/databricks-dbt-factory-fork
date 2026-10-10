# Databricks notebook source

import time

import requests

# COMMAND ----------

dbutils.widgets.text("workspace_id", "")
dbutils.widgets.text("dataset_id", "")
dbutils.widgets.text("secret_scope", "")
dbutils.widgets.text("poll_interval_seconds", "30")

workspace_id = dbutils.widgets.get("workspace_id")
dataset_id = dbutils.widgets.get("dataset_id")
secret_scope = dbutils.widgets.get("secret_scope")
poll_interval_seconds = float(dbutils.widgets.get("poll_interval_seconds") or "30")

for _name, _value in (("workspace_id", workspace_id), ("dataset_id", dataset_id), ("secret_scope", secret_scope)):
    if not _value:
        raise ValueError(f"{_name} parameter is required")

# COMMAND ----------

POWERBI_API = "https://api.powerbi.com/v1.0/myorg"
POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"
ENTRA_AUTHORITY = "https://login.microsoftonline.com"
REQUEST_TIMEOUT_SECONDS = 60
# Refresh history reports a refresh in progress as `Unknown`; every other status is final.
IN_PROGRESS_STATUS = "Unknown"
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_CONSECUTIVE_POLL_FAILURES = 5
# How long a triggered refresh may stay missing from the refresh history before the task gives up.
MAX_SECONDS_UNTRACKED = 600

# The service principal must be allowed to use Power BI APIs and be a member of the workspace.
tenant_id = dbutils.secrets.get(secret_scope, "powerbi-tenant-id")
client_id = dbutils.secrets.get(secret_scope, "powerbi-client-id")
client_secret = dbutils.secrets.get(secret_scope, "powerbi-client-secret")

_token = {"value": "", "expires_at": 0.0}


def access_token() -> str:
    """Returns an Entra ID token for the Power BI API, renewing it before it expires."""
    if time.monotonic() < _token["expires_at"]:
        return _token["value"]
    response = requests.post(
        f"{ENTRA_AUTHORITY}/{tenant_id}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": POWERBI_SCOPE,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        raise RuntimeError(f"Could not authenticate to Power BI: HTTP {response.status_code}: {response.text}")
    body = response.json()
    _token["value"] = body["access_token"]
    _token["expires_at"] = time.monotonic() + int(body.get("expires_in", 3600)) - 300
    return _token["value"]


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {access_token()}"}


dataset_url = f"{POWERBI_API}/groups/{workspace_id}/datasets/{dataset_id}"

# COMMAND ----------

# Trigger the refresh. Power BI answers 202 with the refresh's id in `x-ms-request-id`, which is the
# `requestId` of its refresh-history entry.
response = requests.post(
    f"{dataset_url}/refreshes",
    headers=auth_headers(),
    json={"notifyOption": "NoNotification"},
    timeout=REQUEST_TIMEOUT_SECONDS,
)
if response.status_code != 202:
    raise RuntimeError(
        f"Power BI refused to refresh dataset {dataset_id} in workspace {workspace_id}: "
        f"HTTP {response.status_code}: {response.text}"
    )
request_id = response.headers.get("x-ms-request-id") or response.headers.get("RequestId")
if not request_id:
    raise RuntimeError("Power BI accepted the refresh but returned no request id to track it by.")
print(f"Triggered refresh {request_id} of dataset {dataset_id} in workspace {workspace_id}")

# COMMAND ----------

# Wait for the refresh to finish so the task fails when the refresh fails.
consecutive_failures = 0
triggered_at = time.monotonic()
while True:
    time.sleep(poll_interval_seconds)
    response = requests.get(
        f"{dataset_url}/refreshes",
        headers=auth_headers(),
        params={"$top": 20},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code in TRANSIENT_STATUS_CODES:
        consecutive_failures += 1
        if consecutive_failures > MAX_CONSECUTIVE_POLL_FAILURES:
            raise RuntimeError(f"Could not read the refresh history: HTTP {response.status_code}: {response.text}")
        continue
    if not response.ok:
        raise RuntimeError(f"Could not read the refresh history: HTTP {response.status_code}: {response.text}")
    consecutive_failures = 0

    refresh = next((entry for entry in response.json().get("value", []) if entry.get("requestId") == request_id), None)
    if refresh is None:
        if time.monotonic() - triggered_at > MAX_SECONDS_UNTRACKED:
            raise RuntimeError(f"Refresh {request_id} did not appear in the refresh history of dataset {dataset_id}.")
        continue

    status = refresh.get("status")
    if status == IN_PROGRESS_STATUS:
        print(f"Refresh {request_id} in progress")
        continue
    if status == "Completed":
        print(f"Refresh {request_id} completed")
        break
    raise RuntimeError(
        f"Refresh {request_id} of dataset {dataset_id} ended with status {status!r}: "
        f"{refresh.get('serviceExceptionJson') or '(no error details)'}"
    )
