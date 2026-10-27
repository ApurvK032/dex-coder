# Design decisions

One entry per non-obvious choice: what was decided, why, and what it costs.

## D-1 Build on humanoid-vla

The robot, scene, tasks, judge and human reference come from
[humanoid-vla](https://github.com/ApurvK032/humanoid-vla), pinned as a git submodule
(`third_party/humanoid-vla`, commit `f5df502`). humanoid-vla provides a tested MuJoCo scene with a
fixed-base Unitree G1 (right arm and Dex3 hand driven by one closure value), seeded layouts, an
outcome judge (success, wrong bin, wrong object, dropped, timeout, unstable) and a scripted expert
that a person tuned by hand over several days.

Why: the study measures the agent, not a new simulator. Reusing humanoid-vla's judge and expert
gives a direct comparison with a human-written controller on identical terms.

## D-2 What the agent does and what it is told

Claude, called through the API, gets the robot's documented interface and the rules and writes
`controller.py`: a class with `__init__(robot, task)` and `act(obs)` returning the 8-value action
(7 arm joint targets and the hand closure, at 25 Hz). It tests on practice layouts (seeds
100000-100999), reads outcomes, numbers and camera frames, edits, and submits. The final
controller is scored on test layouts (seeds 300000-300099) that the agent never sees.

The prompt ([prompts.py](../src/dexcoder/prompts.py)) documents the interface, not the expert's
design knowledge (reachable regions, grasp headings, waypoint heights, timings). Effort is `high`,
prompt caching is on, and there is no server-side refusal fallback: a fallback would switch
models mid-run and mix results, so a refusal ends the run and is recorded.

## D-3 How scores are protected

- **Study 1 (exact poses):** controller code runs in a bubblewrap sandbox and records the actions
  it executed. A separate trusted process replays them in a fresh simulator and judges the
  replay. Physics is deterministic, so an honest controller always matches; a mismatch is reported
  as `integrity_mismatch`. A text scan also flags code that reaches around the interface.
- **Study 2 (cameras):** physics and rendering run in a trusted process
  ([vision.py](../src/dexcoder/vision.py)). The controller runs in its own sandboxed process
  ([vworker.py](../src/dexcoder/vworker.py)) that can import only the standard library, numpy and
  mujoco, whose robot model has no cubes, and that receives only the observation each step. The
  simulator, this package, humanoid-vla and the home folder are not visible to it (tested).
  Episodes are judged on the simulation that ran.
- **The sandbox** has no network, no home folder, a private /tmp, read-only system and Python
  environment, no new user namespaces, and write access only to its run folder. On Ubuntu 24.04,
  bubblewrap needs Ubuntu's `bwrap-userns-restrict` AppArmor profile (README, Setup).

## D-4 Held-out conditions

Every final controller is scored on 100 test layouts under each condition. The conditions were
tuned so that the reference controllers still succeed (99-100%):

- **Heavy and slippery:** blocks three times heavier (set before compiling the model, since
  changing masses in a compiled model left stale constants and unstable contacts) and half the
  sliding friction everywhere, fingers included.
- **Weak motors:** arm and hand servo stiffness halved, damping scaled to match, and gravity
  compensation off, so the arm sags.
- **Noisy sensing (study 1):** block poses with a per-episode 4 mm bias, 1.5 mm jitter, a 4 degree
  yaw bias and one control step (40 ms) of delay.
- **Lighting (study 2):** a dimmer, warmer headlight and a strong side light that casts shadows,
  its direction varying between episodes.
- **Noisy camera (study 2):** pixel noise, depth noise that grows with distance, 2% of depth
  pixels missing, the head camera knocked about 3 mm and 0.5 degrees off its calibrated pose
  (not reported to the controller), and images one control step late.

## D-5 Study 1: three models, three tasks, three runs each

Tasks: sort a named cube into a named bin (humanoid-vla's task; its expert is the reference),
stack one cube on another, and put two named cubes into one bin. Models: Claude Fable 5.1,
Opus 5.5 and Sonnet 5.5, each with 40 turns. 27 runs and a pilot, $80.45 of API spend.

Changes made during the campaign:

- After the pilot, scratch scripts got a copy of the current `controller.py` so they can
  `import controller`; the pilot's agent tried and could not. The pilot is therefore reported
  but kept out of the findings.
- A run starts only if the budget left covers a full run of its kind. It never skipped a run.
- Learning curves score each distinct controller version a run tested (at most 12, evenly
  spaced, always the last) on the first 30 test layouts, nominal condition only. Scoring every
  trial on all 100 layouts would have taken hours, mostly on failing two-block controllers that
  run to the 40 s limit.

## D-6 No model ranking

Three runs per task and model cannot rank models. The same model on the same task ranged from
19% to 87% (Opus, two blocks), and with three runs against three even a perfect separation
reaches only p = 0.05. A simulation from the observed spread put a clear answer at about 10 runs
per model for Opus against Sonnet and 20 for Opus against Fable. Study 1 therefore reports only
results that use every controller or thousands of episodes, and lists each run as raw data
without per-model averages.

## D-7 Study 2: cameras only

The biggest limitation of study 1 was that controllers read exact cube poses. Study 2 removes
them: Opus 5.5, five runs, the sort task, 50 turns and an $8 cap per run (perception needs more
work than reading poses). The controller gets the head and wrist camera images (320 x 240, colour
and depth along the viewing axis), each camera's pose from the robot's calibration and measured
joints, the joint angles and the instruction. One more tool, `get_observations`, saves what a
controller would observe at chosen times of a practice episode, with the true cube poses, for
developing perception offline. The reference for this study is humanoid-vla's scripted expert
planning from cubes found by colour and depth in the first head image; it was written for this
study to check that every condition can be solved from the cameras. The five runs cost $13.22.

## D-8 Statistics

A controller against a reference on the same 100 layouts: Fisher's exact test. A change across
controllers (each controller against itself under another condition, or its first version
against its last): a sign-flip test on the paired differences, exact for up to 16 pairs and
Monte Carlo (200,000 draws) above. Rates carry 95% Wilson intervals in the result files.

## D-9 Repository history

The study was developed in a private repository between 2026-10-19 and 2026-10-27 and moved
here on 2026-10-27. On the move the package was renamed from `chiron` to `dexcoder` and its
environment class from `ChironEnv` to `TaskEnv`; nothing else in the code changed. The runs were
made under the old names, so the run transcripts and the system prompts in `results/transcripts/`
say `chiron` (for example `from chiron.explore import setup`). Every final score, every learning
curve and both references were then re-scored from this repository's code; each result file
records the commit it came from.
