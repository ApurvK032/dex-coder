"""Held-out evaluation: a controller on TEST_SEEDS under every physics/sensing variant.

Agent controllers always run sandboxed. The human reference (humanoid-vla's scripted expert)
is trusted code and runs the same pipeline with ``builtin:expert``.
"""

import json
import re
import time
from pathlib import Path

from dexcoder import trials as T
from dexcoder.tasks import TEST_SEEDS, VARIANTS, VISION_VARIANTS
from dexcoder.utils import wilson

# Code patterns that would reach the simulator around the action interface. A match is only a
# flag for a human to read the code; scoring never depends on it (replay does).
SUSPICIOUS = {
    "gc": r"\bimport\s+gc\b|\bgc\.",
    "inspect/frames": r"\bimport\s+inspect\b|\binspect\.|_getframe|f_back|f_locals|f_globals",
    "sys.modules": r"sys\.modules",
    "ctypes": r"\bctypes\b",
    # (chiron and ChironEnv are the names the package and environment had while the runs were made.)
    "simulator internals": r"\benv\.data\b|\b(ChironEnv|TaskEnv)\b|\b(chiron|dexcoder)\.(tasks|trials|worker|vision|robot_api|explore)\b|humanoid_vla",
    "files/processes": r"\bsubprocess\b|\bos\.system\b|open\(.*['\"]w",
}


def scan_code(code: str) -> list[str]:
    return [name for name, pattern in SUSPICIOUS.items() if re.search(pattern, code)]


def evaluate_controller(controller: str, kind: str, out_dir: Path, variants=None, seeds=None,
                        workers: int = 8, sandbox: bool = True, mode: str = "state") -> dict:
    seeds = list(seeds or TEST_SEEDS)
    variants = variants or (VISION_VARIANTS if mode == "vision" else VARIANTS)
    result = {"controller": controller, "kind": kind, "mode": mode, "seeds": [seeds[0], seeds[-1], len(seeds)], "variants": {}}
    for variant in variants:
        start = time.time()
        episodes = T.run_trials(controller, kind, variant, seeds, out_dir / variant, workers=workers, sandbox=sandbox, mode=mode)
        k = sum(e.outcome == "success" for e in episodes)
        lo, hi = wilson(k, len(episodes))
        result["variants"][variant] = {
            "successes": k,
            "n": len(episodes),
            "rate": k / len(episodes),
            "ci95": [round(lo, 3), round(hi, 3)],
            "outcomes": T.outcome_counts(episodes),
            "wall_s": round(time.time() - start, 1),
        }
    if not controller.startswith("builtin:"):
        result["code_flags"] = scan_code(Path(controller).read_text())
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
