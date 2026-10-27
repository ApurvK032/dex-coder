"""Run one coaching run: one model writes a controller for one task.

    .venv/bin/python scripts/run_agent.py --run-id pilot_opus_sort --kind sort --model opus \
        --max-turns 40 --max-usd 6

Spends real API money (key from ANTHROPIC_API_KEY or ~/.config/dexcoder/anthropic_api_key). The global cap in
outputs/ledger.jsonl stops every run before total spend passes --cap. Create outputs/STOP to
stop all runs before their next API call.
"""

import argparse
import json
import sys

from dexcoder.agent import MODELS, REPO, AgentRun, RunConfig, make_client
from dexcoder.ledger import Ledger
from dexcoder.tasks import TASK_KINDS

LEDGER = REPO / "outputs" / "ledger.jsonl"
# Expected cost of a full run (slightly above the most expensive finished run of the
# 2026-10-27 campaign). A run is not started unless this much budget is left, so the global
# cap does not cut runs off halfway.
FULL_RUN_USD = {
    ("pair", "fable"): 11.0, ("sort", "fable"): 5.5, ("stack", "fable"): 5.5,
    ("pair", "opus"): 4.0, ("sort", "opus"): 3.0, ("stack", "opus"): 3.0,
    ("pair", "sonnet"): 2.0, ("sort", "sonnet"): 1.6, ("stack", "sonnet"): 1.6,
}
# Camera-only runs: the pilot (vision_sort_opus_1, 2026-10-27) cost $2.41. Unknown pairs need
# their whole --max-usd.
VISION_FULL_RUN_USD = {("sort", "opus"): 4.0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--kind", choices=TASK_KINDS, required=True)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-usd", type=float, default=6.0)
    p.add_argument("--effort", default="high")
    p.add_argument("--cap", type=float, default=95.0, help="global spend cap across all runs ($)")
    p.add_argument("--mode", choices=("state", "vision"), default="state",
                   help="vision: the controller gets camera images instead of cube poses")
    a = p.parse_args()
    cfg = RunConfig(run_id=a.run_id, kind=a.kind, model=MODELS[a.model], max_turns=a.max_turns,
                    max_usd=a.max_usd, effort=a.effort, mode=a.mode)
    ledger = Ledger(LEDGER, a.cap)
    table = VISION_FULL_RUN_USD if a.mode == "vision" else FULL_RUN_USD
    need = min(a.max_usd, table.get((a.kind, a.model), a.max_usd))
    spent = ledger.total()
    if spent + need > a.cap:
        print(json.dumps({"run_id": a.run_id, "skipped": f"not started: ${spent:.2f} spent; a full run needs about ${need:.2f} under the ${a.cap:.2f} cap"}))
        return 0
    state = AgentRun(cfg, ledger, make_client(), sandbox=True).run()
    print(json.dumps({"run_id": a.run_id, "end_reason": state.end_reason, "turns": state.turn,
                      "cost_usd": round(state.cost_usd, 4), "versions": state.versions,
                      "trials": [(t["k"], t["successes"], t["n"]) for t in state.trials]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
