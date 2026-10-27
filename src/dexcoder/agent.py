"""The coach agent: Claude, through the API, writes, tests and improves controller.py.

One run = one model on one task: a manual tool-use loop with prompt caching, a cost ledger and
hard stops (turns, per-run dollars, global dollars, wall time, and an ``outputs/STOP`` file).
Controller code and scratch scripts run only inside the bubblewrap sandbox.

No server-side refusal fallback is configured: a fallback would silently switch models mid-run
and mix the results being compared. A refusal ends the run and is recorded.
"""

import base64
import json
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dexcoder import trials as T
from dexcoder.ledger import Ledger
from dexcoder.prompts import KICKOFF, system_prompt, tools_for
from dexcoder.sandbox import check_sandbox, sandboxed_command
from dexcoder.tasks import PRACTICE_SEEDS, TIME_LIMIT_S

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "outputs" / "runs"
STOP_FILE = REPO / "outputs" / "STOP"
KEY_FILE = Path.home() / ".config" / "dexcoder" / "anthropic_api_key"

MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "fable": "claude-fable-5-1"}
MAX_TOKENS = 32000
# Worst case for one more call, kept free under the global cap before each call.
CALL_RESERVE_USD = {"claude-fable-5-1": 2.5, "claude-opus-5-5": 1.0, "claude-sonnet-5-5": 0.5}
MAX_CODE_CHARS = 80_000
PYTHON_OUTPUT_CHARS = 10_000


@dataclass
class RunConfig:
    run_id: str
    kind: str
    model: str  # full model ID
    max_turns: int = 40
    max_usd: float = 6.0
    effort: str = "high"
    max_wall_s: float = 3 * 3600
    workers: int = 4
    mode: str = "state"  # "vision": camera-only (dexcoder.vision)


@dataclass
class RunState:
    turn: int = 0
    cost_usd: float = 0.0
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    versions: int = 0
    trials: list = field(default_factory=list)
    submitted: bool = False
    notes: str | None = None
    end_reason: str | None = None
    started: float = field(default_factory=time.time)
    tool_counts: dict = field(default_factory=dict)


def make_client():
    """The API key comes from ANTHROPIC_API_KEY or, if that is unset, from KEY_FILE (one line)."""
    import os

    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        raise RuntimeError(f"set ANTHROPIC_API_KEY or put the key in {KEY_FILE}")
    return anthropic.Anthropic(api_key=key, max_retries=6, timeout=900.0)


class AgentRun:
    def __init__(self, cfg: RunConfig, ledger: Ledger, client, sandbox: bool = True) -> None:
        self.cfg, self.ledger, self.client, self.sandbox = cfg, ledger, client, sandbox
        self.dir = RUNS_DIR / cfg.run_id
        if self.dir.exists():
            raise FileExistsError(f"run {cfg.run_id} already exists: {self.dir}")
        self.dir.mkdir(parents=True)
        self.controller = self.dir / "controller.py"
        self.state = RunState()
        self.system = system_prompt(cfg.kind, cfg.max_turns, cfg.max_usd, TIME_LIMIT_S[cfg.kind], cfg.mode)
        self.tools = tools_for(cfg.mode)
        self.latest_trials: Path | None = None
        self.latest_seeds: list[int] = []
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        (self.dir / "system_prompt.txt").write_text(self.system)

    # Main loop -------------------------------------------------------------------------------

    def run(self) -> RunState:
        if self.sandbox:
            check_sandbox()
        messages: list[dict] = [{"role": "user", "content": KICKOFF}]
        idle_turns = 0
        try:
            while True:
                reason = self._should_stop()
                if reason:
                    self.state.end_reason = reason
                    break
                self.state.turn += 1
                response = self._call(messages)
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason == "refusal":
                    details = getattr(response, "stop_details", None)
                    self._log_turn(response, [], extra={"refusal": str(details)})
                    self.state.end_reason = f"refusal: {details}"
                    break
                if response.stop_reason == "max_tokens":
                    results = [self._tool_error(b.id, "Your reply hit the output limit before this tool call was complete, so nothing ran. Send it again in smaller pieces (edit_controller for changes).") for b in tool_uses]
                    note = "Your last reply hit the output limit. Continue in smaller steps."
                    messages.append({"role": "user", "content": results + [{"type": "text", "text": note}]})
                    self._log_turn(response, results)
                    continue
                if not tool_uses:
                    idle_turns += 1
                    self._log_turn(response, [])
                    if idle_turns >= 3:
                        self.state.end_reason = "stopped calling tools without submitting"
                        break
                    messages.append({"role": "user", "content": "No tool was called. Keep working with the tools, or call submit if you are finished."})
                    continue
                idle_turns = 0
                results = [self._execute(b) for b in tool_uses]
                self._log_turn(response, results)
                if self.state.submitted:
                    self.state.end_reason = "submitted"
                    break
                footer = {"type": "text", "text": self._budget_line()}
                messages.append({"role": "user", "content": [*results, footer]})
        except Exception as err:  # noqa: BLE001 - record and end the run, never crash the campaign
            self.state.end_reason = f"harness error: {type(err).__name__}: {err}"
            (self.dir / "harness_error.txt").write_text(traceback.format_exc())
        self._finish()
        return self.state

    def _should_stop(self) -> str | None:
        cfg, st = self.cfg, self.state
        if STOP_FILE.exists():
            return "stopped by outputs/STOP"
        if st.turn >= cfg.max_turns:
            return "turn limit"
        if st.cost_usd >= cfg.max_usd:
            return "run budget spent"
        ok, spent = self.ledger.reserve_ok(CALL_RESERVE_USD.get(cfg.model, 1.0))
        if not ok:
            return f"global budget reached (${spent:.2f} of ${self.ledger.cap_usd:.2f})"
        if time.time() - st.started > cfg.max_wall_s:
            return "wall-clock limit"
        return None

    def _call(self, messages: list[dict]):
        params = dict(
            model=self.cfg.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=self.tools,
            messages=messages,
            cache_control={"type": "ephemeral"},
            output_config={"effort": self.cfg.effort},
        )
        for attempt in range(5):
            try:
                with self.client.messages.stream(**params) as stream:
                    response = stream.get_final_message()
                break
            except Exception as err:  # noqa: BLE001
                name = type(err).__name__
                if name in ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError", "OverloadedError") and attempt < 4:
                    time.sleep(30 * (attempt + 1))
                    continue
                raise
        usage = {k: int(getattr(response.usage, k, 0) or 0) for k in self.state.usage}
        cost = self.ledger.record(self.cfg.run_id, self.cfg.model, usage, getattr(response, "_request_id", None), note=f"turn {self.state.turn}")
        self.state.cost_usd += cost
        for k, v in usage.items():
            self.state.usage[k] += v
        return response

    def _budget_line(self) -> str:
        st, cfg = self.state, self.cfg
        return (f"[Budget: ${st.cost_usd:.2f} of ${cfg.max_usd:.2f} spent; turn {st.turn} of {cfg.max_turns}. "
                f"Controller versions so far: {st.versions}.]")

    # Tools -----------------------------------------------------------------------------------

    def _execute(self, block) -> dict:
        name, args = block.name, dict(block.input or {})
        self.state.tool_counts[name] = self.state.tool_counts.get(name, 0) + 1
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return self._tool_error(block.id, f"Unknown tool {name}.")
            content = handler(**args)
        except TypeError as err:
            return self._tool_error(block.id, f"Bad arguments for {name}: {err}")
        except Exception as err:  # noqa: BLE001
            return self._tool_error(block.id, f"{name} failed: {type(err).__name__}: {err}")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        return {"type": "tool_result", "tool_use_id": block.id, "content": content}

    @staticmethod
    def _tool_error(tool_use_id: str, text: str) -> dict:
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": text, "is_error": True}

    def _save_controller(self, code: str) -> str:
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"controller.py is limited to {MAX_CODE_CHARS} characters")
        self.controller.write_text(code)
        self.state.versions += 1
        shutil.copy(self.controller, self.dir / f"controller_v{self.state.versions:03d}.py")
        msg = f"Saved controller.py (version {self.state.versions}, {code.count(chr(10)) + 1} lines)."
        try:
            compile(code, "controller.py", "exec")
        except SyntaxError as err:
            msg += f" Warning: it does not compile: {err}"
        return msg

    def _tool_write_controller(self, code: str) -> str:
        return self._save_controller(code)

    def _tool_edit_controller(self, old_str: str, new_str: str) -> str:
        if not self.controller.exists():
            raise FileNotFoundError("controller.py does not exist yet; use write_controller")
        code = self.controller.read_text()
        n = code.count(old_str)
        if n != 1:
            raise ValueError(f"old_str must occur exactly once in controller.py; it occurs {n} times")
        return self._save_controller(code.replace(old_str, new_str))

    def _tool_run_trials(self, num_episodes: int = 10, seeds: list[int] | None = None) -> str:
        if not self.controller.exists():
            raise FileNotFoundError("controller.py does not exist yet")
        if seeds:
            seeds = [int(s) for s in seeds][:40]
            bad = [s for s in seeds if s not in PRACTICE_SEEDS]
            if bad:
                raise ValueError(f"only practice seeds {PRACTICE_SEEDS.start}-{PRACTICE_SEEDS.stop - 1} are allowed: {bad}")
        else:
            seeds = list(PRACTICE_SEEDS[: max(1, min(40, int(num_episodes)))])
        k = len(self.state.trials) + 1
        out = self.dir / "trials" / f"t{k:03d}"
        snapshot = self.dir / "trials" / f"t{k:03d}_controller.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.controller, snapshot)
        start = time.time()
        episodes = T.run_trials(str(snapshot), self.cfg.kind, "nominal", seeds, out, workers=self.cfg.workers,
                                sandbox=self.sandbox, mode=self.cfg.mode)
        self.latest_trials, self.latest_seeds = out, seeds
        self.state.trials.append({
            "k": k, "version": self.state.versions, "turn": self.state.turn, "seeds": seeds,
            "successes": sum(e.outcome == "success" for e in episodes), "n": len(episodes),
            "outcomes": T.outcome_counts(episodes), "cost_usd_so_far": round(self.state.cost_usd, 4),
            "wall_s": round(time.time() - start, 1),
        })
        return T.summarize_for_agent(episodes)

    def _require_latest(self, seed: int) -> Path:
        if self.latest_trials is None:
            raise ValueError("run_trials first")
        if int(seed) not in self.latest_seeds:
            raise ValueError(f"seed {seed} was not in the latest run_trials (seeds {self.latest_seeds[0]}..{self.latest_seeds[-1]})")
        return self.latest_trials

    def _tool_inspect_episode(self, seed: int, start_s: float = 0, end_s: float = 60, every_s: float = 0.2) -> str:
        out = self._require_latest(seed)
        return T.episode_table(out, int(seed), self.cfg.kind, float(start_s), float(end_s), max(0.04, float(every_s)))

    def _tool_view_frames(self, seed: int, times: list[float], cameras: list[str] | None = None) -> list[dict]:
        from dexcoder.frames import CAMERAS, contact_sheet

        out = self._require_latest(seed)
        cameras = [c for c in (cameras or ["scene", "wrist"]) if c in CAMERAS][:3] or ["scene", "wrist"]
        png = contact_sheet(out / f"traj_{int(seed)}.npz", self.cfg.kind, [float(t) for t in times][:6], cameras)
        k = self.state.tool_counts["view_frames"]
        (self.dir / "frames").mkdir(exist_ok=True)
        (self.dir / "frames" / f"view_{k:03d}_seed{seed}.png").write_bytes(png)
        return [
            {"type": "text", "text": f"Seed {seed}, rows = times {times[:6]}, columns = {cameras}."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode()}},
        ]

    def _tool_get_observations(self, seed: int, times: list[float] | None = None) -> str:
        if self.cfg.mode != "vision":
            raise ValueError("get_observations exists only in camera-only runs")
        from dexcoder.vision import save_observations

        seed = int(seed)
        if seed not in PRACTICE_SEEDS:
            raise ValueError(f"only practice seeds {PRACTICE_SEEDS.start}-{PRACTICE_SEEDS.stop - 1} are allowed")
        times = [max(0.0, float(t)) for t in (times or [0.0])][:6]
        traj = None
        if self.latest_trials is not None and seed in self.latest_seeds:
            traj = self.latest_trials / f"traj_{seed}.npz"
        scratch = self.dir / "scratch"
        scratch.mkdir(exist_ok=True)
        names = save_observations(self.cfg.kind, seed, times, traj, scratch)
        return (f"Saved in run_python's folder: {', '.join(names)}. Keys: head_rgb, head_depth, head_pos, "
                "head_mat, wrist_rgb, wrist_depth, wrist_pos, wrist_mat (as in obs['cameras']), qpos, closure, "
                "t, and the true cube poses true_red_pos, true_red_yaw, true_green_pos, true_green_yaw, "
                "true_blue_pos, true_blue_yaw (practice only; controllers never get these).")

    def _tool_run_python(self, code: str) -> str:
        scratch = self.dir / "scratch"
        scratch.mkdir(exist_ok=True)
        if self.controller.exists():  # so scripts can `import controller`
            shutil.copy(self.controller, scratch / "controller.py")
        k = self.state.tool_counts["run_python"]
        script = scratch / f"s{k:03d}.py"
        script.write_text(code)
        # -E -s rather than -I: the script's folder must stay importable (import controller);
        # the sandbox is what confines the script.
        cmd = [str(REPO / ".venv/bin/python"), "-E", "-s", str(script)]
        if self.sandbox:
            cmd = sandboxed_command(cmd, writable=[scratch])
        env = {"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1", "HOME": "/tmp", "MUJOCO_GL": "disable"}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env, cwd=scratch)
            out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr.strip() else "")
            status = f"[exit code {proc.returncode}]"
        except subprocess.TimeoutExpired as err:
            out = (err.stdout or b"").decode(errors="replace") if isinstance(err.stdout, bytes) else (err.stdout or "")
            status = "[stopped after 120 s]"
        if len(out) > PYTHON_OUTPUT_CHARS:
            out = out[:PYTHON_OUTPUT_CHARS] + f"\n[... output truncated, {len(out) - PYTHON_OUTPUT_CHARS} more characters]"
        return f"{status}\n{out}"

    def _tool_submit(self, notes: str) -> str:
        if not self.controller.exists():
            raise FileNotFoundError("write a controller before submitting")
        self.state.submitted = True
        self.state.notes = notes
        return "Submitted."

    # Records ---------------------------------------------------------------------------------

    def _log_turn(self, response, results: list[dict], extra: dict | None = None) -> None:
        blocks = []
        for b in response.content:
            d = b.model_dump() if hasattr(b, "model_dump") else dict(b.__dict__)
            d.pop("signature", None)
            blocks.append(d)
        logged = []
        for r in results:
            content = r["content"]
            if isinstance(content, list):
                content = [c if c.get("type") != "image" else {"type": "image", "note": "PNG saved under frames/"} for c in content]
            logged.append({"tool_use_id": r["tool_use_id"], "is_error": r.get("is_error", False), "content": content})
        entry = {
            "turn": self.state.turn,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stop_reason": response.stop_reason,
            "usage": {k: int(getattr(response.usage, k, 0) or 0) for k in self.state.usage},
            "run_cost_usd": round(self.state.cost_usd, 4),
            "assistant": blocks,
            "tool_results": logged,
            **(extra or {}),
        }
        with (self.dir / "transcript.jsonl").open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _finish(self) -> None:
        st = self.state
        summary = {
            "config": asdict(self.cfg),
            "end_reason": st.end_reason,
            "submitted": st.submitted,
            "notes": st.notes,
            "turns": st.turn,
            "cost_usd": round(st.cost_usd, 4),
            "usage": st.usage,
            "versions": st.versions,
            "trials": st.trials,
            "tool_counts": st.tool_counts,
            "wall_s": round(time.time() - st.started, 1),
            "final_controller": "controller.py" if self.controller.exists() else None,
        }
        (self.dir / "run.json").write_text(json.dumps(summary, indent=1))
