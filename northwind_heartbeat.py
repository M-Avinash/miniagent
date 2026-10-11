"""Northwind heartbeat — a proactive monitor that watches the data and alerts.

This is the OpenClaw-style heartbeat applied to the Northwind use case: on
each beat the monitor agent runs a single query (recent orders), evaluates it
against a condition (delayed shipments / freight spikes), and either surfaces
an alert or no-ops with HEARTBEAT_OK.

Architecture:
  - One monitor Agent with the odata_query + analyze_data tools.
  - A HeartbeatAgent wakes it on an interval.
  - An on_alert callback delivers non-no-op results (print + alerts.jsonl).

Run it:
    python -m miniagent.northwind_heartbeat

By default it runs a bounded demo (a few beats). Pass --forever to run until
Ctrl-C, or --interval N to change the beat interval in seconds.
"""

import json
import sys
import time
from datetime import datetime

from . import Agent, HeartbeatAgent, verbose_log
from .northwind_qna import odata_query, analyze_data, fetch_schema, shared_state


def create_monitor_agent() -> Agent:
    """The monitor agent — runs one query per beat and decides if it's worth alerting."""
    system_prompt = """You are a Northwind sales monitor running a periodic heartbeat.

On each heartbeat, do exactly this:
1. Call odata_query on 'Orders' to fetch recent orders (use $top 200, ordered
   by OrderDate desc). Fetch the FULL entity (do not use $select).
2. Call analyze_data with Python that flags anything needing attention:
   - DELAYED: ShippedDate is more than 14 days after OrderDate, OR ShippedDate
     is null and OrderDate is older than 21 days (stuck unshipped).
   - HIGH FREIGHT: Freight greater than 300.
   Print a short list of any flagged orders (OrderID, CustomerID, ShipCountry,
   OrderDate, ShippedDate, Freight, and why it was flagged).
3. If ANY orders are flagged, reply with a brief alert naming them.
4. If NOTHING is flagged, reply with exactly: HEARTBEAT_OK

Date format in the data: ISO 8601 (e.g. 1997-04-01T00:00:00Z). Parse with
datetime.fromisoformat(s.replace('Z', '+00:00')).

Be concise. An alert is one or two lines. If nothing is wrong, ONLY reply
HEARTBEAT_OK (no other words)."""

    return Agent(
        system_prompt,
        tools=[odata_query, analyze_data],
        hooks=[verbose_log("monitor")],
        name="monitor",
        max_tool_calls=8,
    )


def on_alert(result):
    """Deliver a non-no-op heartbeat result: print it and append to alerts.jsonl."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print("=" * 60)
    print(f"[ALERT {stamp}]")
    print(result.text.strip())
    print("=" * 60)

    entry = {"ts": stamp, "beat": result.turns, "alert": result.text.strip()}
    try:
        with open("alerts.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[could not write alerts.jsonl: {e}]")


def run_demo(beats: int = 3, interval: float = 1.0, forever: bool = False):
    """Run the Northwind heartbeat.

    Args:
        beats: number of beats for a bounded demo (ignored if forever).
        interval: seconds between beats.
        forever: run until interrupted (Ctrl-C) instead of a bounded demo.
    """
    print("Fetching schema from Northwind OData service...")
    fetch_schema()
    print()

    monitor = create_monitor_agent()
    hb = HeartbeatAgent(monitor, interval=interval, on_alert=on_alert)

    print(f"Heartbeat started (interval={interval}s, "
          f"{'forever' if forever else f'{beats} beats'}). Ctrl-C to stop.")
    print("Most beats should be silent (HEARTBEAT_OK no-ops).\n")

    try:
        if forever:
            hb.run_forever()
        else:
            hb.run(n_times=beats)
    except KeyboardInterrupt:
        pass

    print(f"\nHeartbeat stopped after {hb.beat} beat(s).")


def main():
    beats = 3
    interval = 1.0
    forever = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--forever":
            forever = True
        elif args[i] == "--interval" and i + 1 < len(args):
            interval = float(args[i + 1])
            i += 1
        elif args[i] == "--beats" and i + 1 < len(args):
            beats = int(args[i + 1])
            i += 1
    run_demo(beats=beats, interval=interval, forever=forever)


if __name__ == "__main__":
    main()
