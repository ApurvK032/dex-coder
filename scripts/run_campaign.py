"""Run a list of coaching runs in parallel, each as its own scripts/run_agent.py process.

    .venv/bin/python scripts/run_campaign.py configs/campaign.json --parallel 6

The JSON is a list of {run_id, kind, model, max_turns, max_usd}. Runs that already have a
folder in outputs/runs/ are skipped, so an interrupted campaign can be restarted. Logs go to
outputs/campaign/<run_id>.log. All runs share the global cap in outputs/ledger.jsonl.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--cap", type=float, default=95.0)
    a = p.parse_args()
    runs = json.loads(Path(a.plan).read_text())
    logs = REPO / "outputs" / "campaign"
    logs.mkdir(parents=True, exist_ok=True)
    pending = [r for r in runs if not (REPO / "outputs" / "runs" / r["run_id"]).exists()]
    print(f"{len(pending)} of {len(runs)} runs to do", flush=True)
    active: dict[str, subprocess.Popen] = {}
    while pending or active:
        while pending and len(active) < a.parallel:
            r = pending.pop(0)
            cmd = [str(REPO / ".venv/bin/python"), str(REPO / "scripts/run_agent.py"),
                   "--run-id", r["run_id"], "--kind", r["kind"], "--model", r["model"],
                   "--max-turns", str(r.get("max_turns", 40)), "--max-usd", str(r["max_usd"]),
                   "--cap", str(a.cap)]
            log = (logs / f"{r['run_id']}.log").open("w")
            active[r["run_id"]] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            print(time.strftime("%H:%M:%S"), "started", r["run_id"], flush=True)
            time.sleep(5)  # stagger starts
        for run_id, proc in list(active.items()):
            if proc.poll() is not None:
                print(time.strftime("%H:%M:%S"), "finished", run_id, "exit", proc.returncode, flush=True)
                del active[run_id]
        time.sleep(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
