"""dexcoder: Claude, through the API, writes, tests and improves controllers for a simulated humanoid
hand (Unitree G1 + Dex3), built on humanoid-vla."""

import os

# MuJoCo picks its OpenGL backend at import time; headless EGL unless the caller chose one.
os.environ.setdefault("MUJOCO_GL", "egl")
