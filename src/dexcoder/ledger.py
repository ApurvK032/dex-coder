"""Every API call's tokens and cost, shared by parallel runs through a locked JSONL file.

Costs use list prices (2026-10-06, $ per million tokens). Cache writes cost 1.25x input (5-minute
TTL); cache reads use each model's cache-read price. The Anthropic Console spend limit on the
workspace is the hard backstop; this ledger is what stops runs before it.
"""

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

PRICES = {  # input, output, cache read
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_FACTOR = 1.25


def call_cost(model: str, usage: dict) -> float:
    p_in, p_out, p_read = PRICES[model]
    return (
        usage.get("input_tokens", 0) * p_in
        + usage.get("cache_creation_input_tokens", 0) * p_in * CACHE_WRITE_FACTOR
        + usage.get("cache_read_input_tokens", 0) * p_read
        + usage.get("output_tokens", 0) * p_out
    ) / 1e6


class Ledger:
    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.cap_usd = cap_usd

    @contextmanager
    def _locked(self):
        with self.path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield f
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _entries(self, f) -> list[dict]:
        f.seek(0)
        return [json.loads(line) for line in f if line.strip()]

    def total(self, run_id: str | None = None) -> float:
        with self._locked() as f:
            return sum(e["cost_usd"] for e in self._entries(f) if run_id in (None, e["run_id"]))

    def reserve_ok(self, reserve_usd: float) -> tuple[bool, float]:
        """Whether one more call costing up to ``reserve_usd`` stays under the global cap."""
        spent = self.total()
        return spent + reserve_usd <= self.cap_usd, spent

    def record(self, run_id: str, model: str, usage: dict, request_id: str | None, note: str = "") -> float:
        cost = call_cost(model, usage)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": run_id,
            "model": model,
            "usage": usage,
            "cost_usd": round(cost, 6),
            "request_id": request_id,
            "note": note,
        }
        with self._locked() as f:
            f.seek(0, 2)
            f.write(json.dumps(entry) + "\n")
        return cost
