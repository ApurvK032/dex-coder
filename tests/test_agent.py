"""The agent loop end to end with scripted replies: every tool, the ledger and the records."""

import json

import pytest

from dexcoder import agent as A
from dexcoder.ledger import Ledger, call_cost
from dexcoder.sandbox import SandboxUnavailable, check_sandbox
from tests.fake_client import FakeClient

IDLE = """
import numpy as np
class Controller:
    def __init__(self, robot, task):
        self.home = np.asarray(robot.home_arm)
        print("instruction:", task["instruction"])
    def act(self, obs):
        return np.append(self.home, 0.0)
"""


def sandbox_available() -> bool:
    try:
        check_sandbox()
        return True
    except SandboxUnavailable:
        return False


@pytest.mark.parametrize("sandbox", [False, pytest.param(True, marks=pytest.mark.skipif(not sandbox_available(), reason="bubblewrap cannot start (AppArmor)"))])
def test_scripted_run_uses_every_tool(tmp_path, monkeypatch, sandbox):
    monkeypatch.setattr(A, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(A, "STOP_FILE", tmp_path / "STOP")
    script = [
        ("write_controller", {"code": IDLE}),
        ("edit_controller", {"old_str": "0.0)", "new_str": "0.1)"}),
        ("run_trials", {"num_episodes": 3}),
        ("inspect_episode", {"seed": 100001, "every_s": 5}),
        ("view_frames", {"seed": 100000, "times": [0, 1.0], "cameras": ["scene"]}),
        ("run_python", {"code": "from dexcoder.explore import setup\nimport controller\nenv, robot, task, obs = setup('sort', 100002)\nprint(task['instruction'], robot.grasp_site, controller.Controller.__name__)"}),
        ("run_trials", {"seeds": [999]}),
        ("submit", {"notes": "idle test controller"}),
    ]
    client = FakeClient(script)
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=50)
    cfg = A.RunConfig(run_id="fake", kind="sort", model="claude-sonnet-5-5", max_turns=20, max_usd=5)
    state = A.AgentRun(cfg, ledger, client, sandbox=sandbox).run()

    assert state.end_reason == "submitted", state.end_reason
    assert state.versions == 2
    assert state.trials[0]["n"] == 3 and state.trials[0]["outcomes"] == {"timeout": 3}
    run_dir = tmp_path / "runs" / "fake"
    lines = [json.loads(x) for x in (run_dir / "transcript.jsonl").read_text().splitlines()]
    results = {i: line["tool_results"][0] for i, line in enumerate(lines)}
    assert "timeout" in results[2]["content"][0]["text"]
    assert "grasp_site" in results[3]["content"][0]["text"]
    assert results[4]["content"][1] == {"type": "image", "note": "PNG saved under frames/"}
    out = results[5]["content"][0]["text"]
    assert "Put the" in out and "Controller" in out and "[exit code 0]" in out
    assert results[6]["is_error"] and "practice seeds" in results[6]["content"]
    # Prompt caching and effort are requested on every call; the system prompt never changes.
    assert all(r["cache_control"] == {"type": "ephemeral"} for r in client.requests)
    assert len({r["system"] for r in client.requests}) == 1
    # Costs: the fake usage is identical every turn.
    per_call = call_cost("claude-sonnet-5-5", {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 8000})
    assert ledger.total("fake") == pytest.approx(per_call * len(script), rel=1e-6)
    summary = json.loads((run_dir / "run.json").read_text())
    assert summary["submitted"] and summary["turns"] == len(script)


def test_global_cap_stops_before_calling(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(A, "STOP_FILE", tmp_path / "STOP")
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=0.4)  # below Sonnet's call reserve
    client = FakeClient([("submit", {"notes": "x"})])
    cfg = A.RunConfig(run_id="capped", kind="sort", model="claude-sonnet-5-5")
    state = A.AgentRun(cfg, ledger, client, sandbox=False).run()
    assert state.end_reason.startswith("global budget reached")
    assert client.requests == []


def test_stop_file(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(A, "STOP_FILE", tmp_path / "STOP")
    (tmp_path / "STOP").touch()
    client = FakeClient([("submit", {"notes": "x"})])
    cfg = A.RunConfig(run_id="stopped", kind="sort", model="claude-sonnet-5-5")
    state = A.AgentRun(cfg, Ledger(tmp_path / "l.jsonl", 50), client, sandbox=False).run()
    assert state.end_reason == "stopped by outputs/STOP" and client.requests == []
