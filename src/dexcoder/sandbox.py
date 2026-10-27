"""bubblewrap sandbox for code the coach agent writes.

Inside the sandbox a program has no network, no access to the home folder (no keys, tokens or
other projects), a private /tmp, read-only views of the system, the project's Python
environment and the simulator, and write access only to the folders it is given. It cannot
create further user namespaces. Time and memory are limited by the caller and the worker.

On Ubuntu 24.04 bubblewrap needs Ubuntu's bwrap-userns-restrict AppArmor profile (README, Setup).
"""

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
READ_ONLY = [REPO / ".venv", REPO / "src", REPO / "third_party" / "humanoid-vla"]


class SandboxUnavailable(RuntimeError):
    pass


def sandboxed_command(cmd: list[str], writable: list[Path], readable: list[Path] = (),
                      base_readonly: list[Path] | None = None) -> list[str]:
    """``base_readonly`` replaces the default read-only paths (the project's Python environment,
    src/ and the simulator); camera-only workers get only the Python environment and one file."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")
    args = [
        bwrap,
        "--unshare-all",
        "--unshare-user",  # explicit: --disable-userns requires it even with --unshare-all
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--hostname", "sandbox",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin",
        "--symlink", "usr/sbin", "/sbin",
        "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    for path in [*(READ_ONLY if base_readonly is None else base_readonly), *readable]:
        args += ["--ro-bind", str(path), str(path)]
    for path in writable:
        args += ["--bind", str(path), str(path)]
    # The sandbox's own root (a throwaway tmpfs holding the mount points) becomes read-only;
    # /tmp and the writable folders keep their own mounts.
    args += ["--remount-ro", "/"]
    args += ["--chdir", str(writable[0]) if writable else "/tmp", "--"]
    return args + cmd


def check_sandbox() -> None:
    """Raise SandboxUnavailable unless bubblewrap can start a sandbox here."""
    try:
        cmd = sandboxed_command(["/usr/bin/true"], writable=[])
    except SandboxUnavailable:
        raise
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise SandboxUnavailable(
            "bubblewrap cannot start a sandbox: "
            + (proc.stderr.strip() or f"exit code {proc.returncode}")
            + ". On Ubuntu 24.04 load the bwrap-userns-restrict AppArmor profile (README, Setup)."
        )
