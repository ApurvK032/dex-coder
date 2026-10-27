"""Small shared helpers."""

import math
import platform
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def provenance() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()

    import mujoco

    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
        "humanoid_vla_commit": subprocess.run(
            ["git", "-C", str(REPO / "third_party/humanoid-vla"), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "mujoco": mujoco.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
    }
