"""tools.py — OData query, metadata, and sandboxed analysis tools."""

import json
import os
import tempfile

import requests

from miniagent import tool, run_python

from .schema import NORTHWIND_URL

shared_state: dict = {"entities": {}, "data_desc": ""}


@tool
def odata_query(entity: str, filter: str = None, select: str = None,
                top: int = None, skip: int = None, orderby: str = None,
                expand: str = None) -> str:
    """Query the Northwind OData service.
    Args:
        entity (str): The entity set to query (e.g., 'Orders', 'Products', 'Customers').
        filter (str): OData $filter expression (e.g., "ShipCountry eq 'Germany'").
        select (str): OData $select expression (e.g., "OrderID,OrderDate,ShippedDate").
        top (int): OData $top (limit the number of results).
        skip (int): OData $skip (offset for pagination).
        orderby (str): OData $orderby expression (e.g., "OrderDate desc").
        expand (str): OData $expand expression (e.g., "Order_Details($expand=Products)").
    """
    params = {}
    if filter:
        params["$filter"] = filter
    if select:
        params["$select"] = select
    if top is not None:
        params["$top"] = top
    if skip is not None:
        params["$skip"] = skip
    if orderby:
        params["$orderby"] = orderby
    if expand:
        params["$expand"] = expand

    url = f"{NORTHWIND_URL}/{entity}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        shared_state["entities"][entity] = data
        count = len(data.get("value", []))
        shared_state["data_desc"] = ", ".join(
            f"{k}: {len(v.get('value', []))} records" for k, v in shared_state["entities"].items()
        )
        sample = data.get("value", [])[:3]
        return json.dumps({
            "status": "ok",
            "count": count,
            "sample": sample,
            "note": f"Full result ({count} records from {entity}) stored in shared state.",
        })
    except requests.RequestException as e:
        return f"[error: {e}]"
    except json.JSONDecodeError as e:
        return f"[error: invalid JSON: {e}]"


@tool
def odata_metadata() -> str:
    """Fetch the Northwind OData service metadata (entity list and schema)."""
    try:
        resp = requests.get(NORTHWIND_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entities = [e.get("name") for e in data.get("EntitySets", [])]
        return json.dumps({"entity_sets": entities})
    except requests.RequestException as e:
        return f"[error: {e}]"


@tool
def analyze_data(code: str, timeout: int = 30) -> str:
    """Run Python code against the data stored in shared state, in a sandbox.

    The code runs in a subprocess with a timeout (reuses run_python). The
    variable `data` is pre-loaded with the full result from odata_query.
    Print your answer with print() — it will be captured and returned.

    If the code fails, the error (with traceback) is returned so the agent
    can fix the code and retry (self-healing).

    Args:
        code (str): Python code to run. Use `data` to access the fetched records.
        timeout (int): Max seconds before the subprocess is killed.
    """
    entities = shared_state["entities"]
    if not entities:
        return "[error: no data in shared state — call odata_query first]"

    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(entities, f)
        preamble = (
            "import json as _json\n"
            f"with open({path!r}) as _f:\n"
            "    _entities = _json.load(_f)\n"
        )
        # Create a variable for each entity: Orders, Products, Customers, etc.
        for name in entities:
            preamble += f"    {name} = _entities['{name}']['value']\n"
        # Also create a generic 'data' variable pointing to the last entity
        last_entity = list(entities.keys())[-1]
        preamble += f"    data = {last_entity}\n"
        full_code = preamble + code
        result = run_python({"code": full_code, "timeout": timeout})
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if result.startswith("[exit 0]"):
        body = result[len("[exit 0]"):].strip()
        return body if body else "[no output — use print() to show your answer]"
    return result
