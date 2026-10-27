"""Held-out evaluation of finished runs and of the human reference.

    .venv/bin/python scripts/evaluate_runs.py --reference      # humanoid-vla expert, sort task
    .venv/bin/python scripts/evaluate_runs.py --runs all       # every finished run

For each run: its final controller on the 100 test layouts under all four variants
(results/eval/<run>.json), every controller version it tested on the nominal test layouts
(learning curve on the first 30 test layouts, results/eval/<run>_curve.json), plus a copy of the final controller
(results/controllers/<run>.py) and the run summary (results/runs/<run>.json).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from dexcoder.evaluate import evaluate_controller, write_json
from dexcoder.tasks import TEST_SEEDS
from dexcoder.utils import provenance

WORKERS = 8  # parallel episode workers per evaluation (--workers)
CURVE_SEEDS = list(TEST_SEEDS[:30])  # intermediate versions: first 30 test layouts, nominal only
REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "outputs" / "runs"
RESULTS = REPO / "results"
SCRATCH = REPO / "outputs" / "eval"


def evaluate_reference() -> None:
    res = evaluate_controller("builtin:expert", "sort", SCRATCH / "reference_expert_sort", sandbox=False, workers=WORKERS)
    res["provenance"] = provenance()
    write_json(RESULTS / "eval" / "reference_expert_sort.json", res)
    print("reference", {v: r["rate"] for v, r in res["variants"].items()})


def evaluate_vision_reference() -> None:
    """The scripted expert planning from cubes found in the head camera (camera-only mode)."""
    res = evaluate_controller("builtin:vision_expert", "sort", SCRATCH / "reference_vision_sort", sandbox=False, mode="vision", workers=WORKERS)
    res["provenance"] = provenance()
    write_json(RESULTS / "eval" / "reference_vision_sort.json", res)
    print("vision reference", {v: r["rate"] for v, r in res["variants"].items()})


def evaluate_run(run_id: str, curve: bool = True, final_eval: bool = True) -> None:
    run_dir = RUNS / run_id
    summary = json.loads((run_dir / "run.json").read_text())
    kind = summary["config"]["kind"]
    mode = summary["config"].get("mode", "state")
    final = run_dir / "controller.py"
    if not final.exists():
        print(run_id, "has no controller; skipped")
        return
    if final_eval:
        _evaluate_final(run_id, summary, kind, final, mode)
    if curve:
        _evaluate_curve(run_id, summary, kind, run_dir, mode)


def _evaluate_final(run_id: str, summary: dict, kind: str, final: Path, mode: str = "state") -> None:
    res = evaluate_controller(str(final), kind, SCRATCH / run_id / "final", mode=mode, workers=WORKERS)
    res["run_id"] = run_id
    res["provenance"] = provenance()
    write_json(RESULTS / "eval" / f"{run_id}.json", res)
    (RESULTS / "controllers").mkdir(parents=True, exist_ok=True)
    shutil.copy(final, RESULTS / "controllers" / f"{run_id}.py")
    write_json(RESULTS / "runs" / f"{run_id}.json", summary)
    print(run_id, {v: r["rate"] for v, r in res["variants"].items()}, "flags:", res.get("code_flags"))


MAX_CURVE_POINTS = 12


def _evaluate_curve(run_id: str, summary: dict, kind: str, run_dir: Path, mode: str = "state") -> None:
    """Score tested controller versions on CURVE_SEEDS: each distinct version once, at most
    MAX_CURVE_POINTS of them evenly spaced, always including the last one tested."""
    trials = list({t["version"]: t for t in summary["trials"]}.values())  # last trial per version
    if len(trials) > MAX_CURVE_POINTS:
        idx = sorted({round(i * (len(trials) - 1) / (MAX_CURVE_POINTS - 1)) for i in range(MAX_CURVE_POINTS)})
        trials = [trials[i] for i in idx]
    points = []
    for t in trials:
        snap = run_dir / "trials" / f"t{t['k']:03d}_controller.py"
        r = evaluate_controller(str(snap), kind, SCRATCH / run_id / f"t{t['k']:03d}", variants=("nominal",),
                                seeds=CURVE_SEEDS, mode=mode, workers=WORKERS)
        points.append({"trial": t["k"], "turn": t["turn"], "version": t["version"],
                       "cost_usd_so_far": t["cost_usd_so_far"], "practice": [t["successes"], t["n"]],
                       "test_nominal": r["variants"]["nominal"]["rate"]})
    write_json(RESULTS / "eval" / f"{run_id}_curve.json", {"run_id": run_id, "points": points})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", action="store_true")
    p.add_argument("--vision-reference", action="store_true")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--no-curve", action="store_true")
    p.add_argument("--curve-only", action="store_true", help="only the learning curves")
    p.add_argument("--workers", type=int, default=8, help="parallel episode workers per evaluation")
    a = p.parse_args()
    global WORKERS
    WORKERS = a.workers
    if a.reference:
        evaluate_reference()
    if a.vision_reference:
        evaluate_vision_reference()
    ids = sorted(d.name for d in RUNS.iterdir() if (d / "run.json").exists()) if a.runs == ["all"] else a.runs
    for run_id in ids:
        evaluate_run(run_id, curve=not a.no_curve, final_eval=not a.curve_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
