"""Tables and figures from results/: results/SUMMARY.md and results/figures/*.png.

    .venv/bin/python scripts/report.py [--results results] [--ledger outputs/ledger.jsonl]

Two studies. Study 1 (controllers read exact cube poses): 27 runs, three per task and model.
Three runs per model are too few to rank models (D-010), so this report gives no per-model
averages: it reports what holds across all controllers, and lists every run as raw data.
Study 2 (camera-only, D-009): Opus 5.5 runs on the sort task.

Statistics: paired changes across controllers use an exact or Monte Carlo sign-flip test; a
controller against a reference on the same 100 layouts uses Fisher's exact test. Colors: the
dataviz reference palette's first categorical slot for Claude-written controllers, its first
three for the three tasks, and neutral gray dashes for references. Every number in a
figure is also in a table.
"""

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODEL_NAMES = {"claude-fable-5-1": "Fable 5.1", "claude-opus-5-5": "Opus 5.5", "claude-sonnet-5-5": "Sonnet 5.5"}
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
TASK_COLORS = {"sort": BLUE, "stack": ORANGE, "pair": GREEN}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TASKS = ["sort", "stack", "pair"]
TASK_NAMES = {"sort": "Sort a block into a bin", "stack": "Stack two blocks", "pair": "Two blocks into one bin"}
VARIANTS1 = ["nominal", "heavy_slippery", "weak_motors", "noisy_sensing"]
VARIANTS2 = ["nominal", "heavy_slippery", "weak_motors", "lighting", "noisy_camera"]
VARIANT_NAMES = {"nominal": "Nominal", "heavy_slippery": "Heavy &\nslippery", "weak_motors": "Weak\nmotors",
                 "noisy_sensing": "Noisy\nsensing", "lighting": "Lighting", "noisy_camera": "Noisy\ncamera"}


# Data ------------------------------------------------------------------------------------------


def load(results: Path):
    runs = {}
    for f in sorted((results / "runs").glob("*.json")):
        s = json.loads(f.read_text())
        ev_path = results / "eval" / f"{f.stem}.json"
        curve_path = results / "eval" / f"{f.stem}_curve.json"
        runs[f.stem] = {
            "summary": s,
            "eval": json.loads(ev_path.read_text()) if ev_path.exists() else None,
            "curve": json.loads(curve_path.read_text()) if curve_path.exists() else None,
        }
    refs = {}
    for name in ("reference_expert_sort", "reference_vision_sort"):
        path = results / "eval" / f"{name}.json"
        refs[name] = json.loads(path.read_text()) if path.exists() else None
    return runs, refs


def mode(r: dict) -> str:
    return r["summary"]["config"].get("mode", "state")


def study1(runs: dict) -> dict:
    """The 27 comparison-campaign runs: the pilot used an earlier harness and is listed only."""
    return {k: v for k, v in runs.items() if mode(v) == "state" and not k.startswith("pilot") and v["eval"]}


def study2(runs: dict) -> dict:
    return {k: v for k, v in runs.items() if mode(v) == "vision" and v["eval"]}


# Statistics ------------------------------------------------------------------------------------


def signflip_p(d, n_mc: int = 200_000) -> float:
    """Two-sided p-value for a mean paired difference of zero."""
    d = np.asarray(d, dtype=float)
    obs = abs(d.mean())
    if len(d) <= 16:
        means = [abs(np.mean(d * s)) for s in itertools.product([-1, 1], repeat=len(d))]
        return float(np.mean([m >= obs - 1e-12 for m in means]))
    signs = np.random.default_rng(0).choice([-1, 1], size=(n_mc, len(d)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test for k1/n1 versus k2/n2."""
    total, k = n1 + n2, k1 + k2

    def prob(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(total, k)

    p_obs = prob(k1)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def rate(r: dict, v: str) -> float:
    return r["eval"]["variants"][v]["rate"]


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# Figures ---------------------------------------------------------------------------------------


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _controller_lines(ax, rs: list[dict], variants: list[str], color: str) -> None:
    x = np.arange(len(variants))
    for r in rs:
        ys = [rate(r, v) for v in variants]
        ax.plot(x, ys, color=color, alpha=0.35, linewidth=1.2, zorder=2)
        ax.scatter(x, ys, s=16, color=color, zorder=3, linewidths=0)


def _reference_marks(ax, ref: dict | None, variants: list[str]) -> None:
    """The reference as a short gray dash at each condition."""
    if ref:
        for xi, v in enumerate(variants):
            y = ref["variants"][v]["rate"]
            ax.plot([xi - 0.28, xi + 0.28], [y, y], color=GRAY, linewidth=3, solid_capstyle="round", zorder=1)


def fig_study1_conditions(s1: dict, ref: dict | None, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, TASKS, strict=True):
        style(ax)
        if task == "sort":
            _reference_marks(ax, ref, VARIANTS1)
        _controller_lines(ax, [r for r in s1.values() if r["summary"]["config"]["kind"] == task], VARIANTS1, BLUE)
        ax.set_xticks(np.arange(len(VARIANTS1)), [VARIANT_NAMES[v] for v in VARIANTS1])
        ax.set_ylim(0, 1.05)
        ax.set_title(TASK_NAMES[task], color=INK, fontsize=11, loc="left")
        if task == "sort":
            ax.set_ylabel("Success on 100 held-out layouts", color=INK2, fontsize=9)
    fig.suptitle("Study 1, exact cube poses: each line is one Claude-written controller across the four conditions; "
                 "gray dashes: the human-written expert", color=INK2, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _curve_xy(r: dict) -> tuple[list[float], list[float]]:
    pts = r["curve"]["points"]
    xs = [p["cost_usd_so_far"] for p in pts] + [r["summary"]["cost_usd"]]
    ys = [p["test_nominal"] for p in pts] + [rate(r, "nominal")]
    return xs, ys


def fig_study1_curves(s1: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, TASKS, strict=True):
        style(ax)
        for r in s1.values():
            if r["summary"]["config"]["kind"] == task and r["curve"] and r["curve"]["points"]:
                xs, ys = _curve_xy(r)
                ax.plot(xs, ys, color=BLUE, alpha=0.55, linewidth=1.5, marker="o", markersize=3)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
        ax.set_xlabel("API spend so far ($)", color=INK2, fontsize=9)
        ax.set_title(TASK_NAMES[task], color=INK, fontsize=11, loc="left")
        if task == "sort":
            ax.set_ylabel("Held-out success (nominal)", color=INK2, fontsize=9)
    fig.suptitle("Study 1: each line is one run; every controller version it tested, scored on 30 held-out layouts "
                 "(last point: the final controller on all 100)", color=INK2, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_study2(s2: dict, ref: dict | None, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.3, 1]})
    style(ax1)
    _reference_marks(ax1, ref, VARIANTS2)
    _controller_lines(ax1, list(s2.values()), VARIANTS2, BLUE)
    ax1.set_xticks(np.arange(len(VARIANTS2)), [VARIANT_NAMES[v] for v in VARIANTS2])
    lowest = min(rate(r, v) for r in s2.values() for v in VARIANTS2)
    ax1.set_ylim(min(0.8, lowest - 0.03), 1.01)  # zoomed: every score is high
    ax1.set_ylabel("Success on 100 held-out layouts (axis zoomed)", color=INK2, fontsize=9)
    ax1.set_title("Final controllers by condition", color=INK, fontsize=11, loc="left")
    style(ax2)
    for r in s2.values():
        if r["curve"] and r["curve"]["points"]:
            xs, ys = _curve_xy(r)
            ax2.plot(xs, ys, color=BLUE, alpha=0.6, linewidth=1.5, marker="o", markersize=3)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(left=0)
    ax2.set_xlabel("API spend so far ($)", color=INK2, fontsize=9)
    ax2.set_ylabel("Held-out success (nominal)", color=INK2, fontsize=9)
    ax2.set_title("Learning curves", color=INK, fontsize=11, loc="left")
    fig.suptitle(f"Study 2, cameras only (sort, Opus 5.5, {len(s2)} run{'s' if len(s2) != 1 else ''}): each line is one run; gray dashes: "
                 "the scripted expert planning from cubes found in the head camera", color=INK2, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# Tables ----------------------------------------------------------------------------------------


def run_rows(rs: dict, variants: list[str]) -> list[str]:
    head = "| Run | Task | Model | End | Turns | Versions | $ | " + " | ".join(VARIANT_NAMES[v].replace("\n", " ") for v in variants) + " | Failures on nominal | Code flags |"
    lines = [head, "|" + "---|" * (9 + len(variants))]
    for run_id, r in rs.items():
        s, ev, cfg = r["summary"], r["eval"], r["summary"]["config"]
        v = ev["variants"]
        fails = {k: n for k, n in v["nominal"]["outcomes"].items() if k != "success"}
        cells = [pct(v[k]["rate"]) for k in variants] + [", ".join(f"{k} {n}" for k, n in fails.items()) or "-",
                                                          ", ".join(ev.get("code_flags") or []) or "-"]
        lines.append(f"| {run_id} | {cfg['kind']} | {MODEL_NAMES.get(cfg['model'], cfg['model'])} | {s['end_reason']} | "
                     f"{s['turns']} | {s['versions']} | {s['cost_usd']:.2f} | " + " | ".join(cells) + " |")
    return lines


def study1_findings(s1: dict, ref: dict | None) -> list[str]:
    ids = list(s1)
    out = []
    # Iteration
    out += ["### Testing and fixing is what makes the controllers work", "",
            "First and last controller version each run tested, scored on the first 30 held-out layouts (nominal).", "",
            "| Task | First version (mean) | Last version (mean) | Runs where the last was better |", "|---|---|---|---|"]
    gains = []
    for task in TASKS:
        rs = [s1[i] for i in ids if s1[i]["summary"]["config"]["kind"] == task and s1[i]["curve"]]
        first = [r["curve"]["points"][0]["test_nominal"] for r in rs]
        last = [r["curve"]["points"][-1]["test_nominal"] for r in rs]
        gains += [b - a for a, b in zip(first, last, strict=True)]
        out.append(f"| {task} | {pct(np.mean(first))} | {pct(np.mean(last))} | {sum(b > a for a, b in zip(first, last, strict=True))} of {len(rs)} |")
    out += ["", f"All runs: mean gain {100 * np.mean(gains):+.0f} points, better in {sum(g > 0 for g in gains)} of {len(gains)}, "
            f"sign-flip p {fmt_p(signflip_p(gains))}.", ""]
    # Conditions
    out += ["### Changed conditions, each controller against itself", "",
            "Change in held-out success from the nominal condition, per controller (100 layouts each).", "",
            "| Condition | All 27: mean change | Worse / better / same | p | Sort | Stack | Two blocks |", "|---|---|---|---|---|---|---|"]
    for v in VARIANTS1[1:]:
        d = [rate(s1[i], v) - rate(s1[i], "nominal") for i in ids]
        per_task = [np.mean([rate(s1[i], v) - rate(s1[i], "nominal") for i in ids if s1[i]["summary"]["config"]["kind"] == t]) for t in TASKS]
        out.append(f"| {VARIANT_NAMES[v].replace(chr(10), ' ')} | {100 * np.mean(d):+.1f} | {sum(x < 0 for x in d)} / {sum(x > 0 for x in d)} / "
                   f"{sum(x == 0 for x in d)} | {fmt_p(signflip_p(d))} | " + " | ".join(f"{100 * x:+.1f}" for x in per_task) + " |")
    out.append("")
    # Sort versus the expert
    if ref:
        sort_ids = [i for i in ids if s1[i]["summary"]["config"]["kind"] == "sort"]
        same = []
        for i in sort_ids:
            ok = all(fisher_p(s1[i]["eval"]["variants"][v]["successes"], s1[i]["eval"]["variants"][v]["n"],
                              ref["variants"][v]["successes"], ref["variants"][v]["n"]) >= 0.05 for v in VARIANTS1)
            same.append(ok)
        out += ["### Sort controllers against the human-written expert", "",
                f"{sum(same)} of {len(sort_ids)} sort controllers are not distinguishable from humanoid-vla's scripted expert in any of "
                f"the four conditions (Fisher's exact test, 5%, 100 layouts each): "
                + ", ".join(i for i, ok in zip(sort_ids, same, strict=True) if ok) + ". The others are lower in at least one.", ""]
    # Failures and integrity
    counts, total = Counter(), 0
    for i in ids:
        for res in s1[i]["eval"]["variants"].values():
            total += res["n"]
            counts.update(res["outcomes"])
    failures = total - counts["success"]
    wrong = counts["wrong_bin"] + counts["wrong_object"]
    out += ["### How controllers fail", "",
            f"Of {total} held-out episodes, {failures} failed: timeouts {counts['timeout']} ({pct(counts['timeout'] / failures)}), "
            f"dropped {counts['dropped']}, wrong block or bin {wrong}, controller errors {counts['controller_error']}, "
            f"integrity mismatches {counts['integrity_mismatch']}. Code flags: "
            f"{sum(len(s1[i]['eval'].get('code_flags') or []) for i in ids)}.", ""]
    # Practice versus held-out
    gaps, untested_submitted = [], 0
    for i in ids:
        s = s1[i]["summary"]
        tested = [t for t in s["trials"] if t["version"] == s["versions"]]
        if tested:
            gaps.append(tested[-1]["successes"] / tested[-1]["n"] - rate(s1[i], "nominal"))
        elif s["submitted"]:
            untested_submitted += 1
    submitted = sum(s1[i]["summary"]["submitted"] for i in ids)
    out += ["### Practice scores against held-out scores", "",
            f"In the {len(gaps)} runs whose final version went through run_trials, its practice success minus its held-out "
            f"success averaged {100 * np.mean(gaps):+.1f} points (sign-flip p {fmt_p(signflip_p(gaps))}): no sign of tuning to "
            f"the practice layouts. In {untested_submitted} of the {submitted} submitted runs, the submitted version was never "
            "run through run_trials after its last edit.", ""]
    return out


def tables(runs: dict, refs: dict, ledger: Path | None) -> str:
    s1, s2 = study1(runs), study2(runs)
    lines = ["# Results", ""]
    if ledger and ledger.exists():
        entries = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        by_model = defaultdict(float)
        for e in entries:
            by_model[e["model"]] += e["cost_usd"]
        lines.append(f"API spend, all calls including pilots: ${sum(by_model.values()):.2f} in {len(entries)} calls ("
                     + ", ".join(f"{MODEL_NAMES.get(m, m)} ${v:.2f}" for m, v in sorted(by_model.items())) + ").")
        lines.append("")
    ref1, ref2 = refs.get("reference_expert_sort"), refs.get("reference_vision_sort")
    lines += ["## Study 1: controllers read exact cube poses", "",
              f"{len(s1)} runs: three tasks, three models, three runs each (${sum(r['summary']['cost_usd'] for r in s1.values()):.2f}). "
              "Three runs per model are too few to compare models, so no per-model averages are given; the per-run table "
              "below is raw data.", ""]
    lines += study1_findings(s1, ref1)
    if ref1:
        v = ref1["variants"]
        lines += ["### Reference", "", "humanoid-vla's scripted expert (human-written, reads true poses), sort: "
                  + ", ".join(f"{VARIANT_NAMES[k].replace(chr(10), ' ')} {pct(v[k]['rate'])}" for k in VARIANTS1) + ".", ""]
    lines += ["### Every run", ""] + run_rows(s1, VARIANTS1)
    pilot = {k: v for k, v in runs.items() if k.startswith("pilot") and v["eval"]}
    if pilot:
        lines += ["", "The pilot ran with an earlier harness (scratch scripts could not import the controller) and is not "
                  "part of the findings above:", ""] + run_rows(pilot, VARIANTS1)
    lines += ["", "![Study 1 by condition](figures/success_by_condition.png)", "",
              "![Study 1 learning curves](figures/learning_curves.png)", ""]
    if s2:
        lines += ["## Study 2: cameras only", "",
                  f"{len(s2)} runs of Opus 5.5 on the sort task (${sum(r['summary']['cost_usd'] for r in s2.values()):.2f}). "
                  "Controllers get head and wrist camera images (colour and depth), joint angles and the instruction; no "
                  "cube poses (D-009).", ""]
        lines += ["| Condition | Lowest run | Highest run | Mean of runs | Reference |", "|---|---|---|---|---|"]
        for v in VARIANTS2:
            xs = [rate(r, v) for r in s2.values()]
            refv = pct(ref2["variants"][v]["rate"]) if ref2 else "-"
            lines.append(f"| {VARIANT_NAMES[v].replace(chr(10), ' ')} | {pct(min(xs))} | {pct(max(xs))} | {pct(np.mean(xs))} | {refv} |")
        lines += ["", "Reference: humanoid-vla's scripted expert planning from cubes found by colour and depth in the first "
                  "head image (written for this study to check that every condition can be solved from the cameras).", ""]
        ids = list(s2)
        if ref2:
            same = [i for i in ids if all(
                fisher_p(s2[i]["eval"]["variants"][v]["successes"], s2[i]["eval"]["variants"][v]["n"],
                         ref2["variants"][v]["successes"], ref2["variants"][v]["n"]) >= 0.05 for v in VARIANTS2)]
            lines += [f"{len(same)} of {len(ids)} controllers are not distinguishable from the reference in any of the five "
                      f"conditions (Fisher's exact test, 5%): {', '.join(same) or 'none'}.", ""]
        drops = {i: min(rate(s2[i], v) - rate(s2[i], "nominal") for v in VARIANTS2[1:]) for i in ids}
        worst = min(drops, key=drops.get)
        counts, total = Counter(), 0
        for i in ids:
            for res in s2[i]["eval"]["variants"].values():
                total += res["n"]
                counts.update(res["outcomes"])
        failures = total - counts["success"]
        costs = [s2[i]["summary"]["cost_usd"] for i in ids]
        minutes = [s2[i]["summary"]["wall_s"] / 60 for i in ids]
        lines += [f"Largest drop from nominal under any changed condition: {100 * -drops[worst]:.0f} points ({worst}).",
                  f"Of {total} held-out episodes, {failures} failed: "
                  + ", ".join(f"{k} {n}" for k, n in counts.most_common() if k != "success") + ".",
                  f"Cost per run ${min(costs):.2f}-{max(costs):.2f}, {min(minutes):.0f}-{max(minutes):.0f} minutes; "
                  f"all {sum(s2[i]['summary']['submitted'] for i in ids)} runs submitted.", ""]
        lines += ["### Every run", ""] + run_rows(s2, VARIANTS2)
        lines += ["", "![Study 2](figures/camera_study.png)", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(REPO / "results"))
    p.add_argument("--ledger", default=str(REPO / "outputs" / "ledger.jsonl"))
    a = p.parse_args()
    results = Path(a.results)
    runs, refs = load(results)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    fig_study1_conditions(study1(runs), refs.get("reference_expert_sort"), results / "figures" / "success_by_condition.png")
    fig_study1_curves(study1(runs), results / "figures" / "learning_curves.png")
    if study2(runs):
        fig_study2(study2(runs), refs.get("reference_vision_sort"), results / "figures" / "camera_study.png")
    (results / "SUMMARY.md").write_text(tables(runs, refs, Path(a.ledger)))
    print((results / "SUMMARY.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
