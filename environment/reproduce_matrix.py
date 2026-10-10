#!/usr/bin/env python3
"""Reproduce the whole experiment matrix from this repository.

Usage (from artifact/ or anywhere):
  python3 ../environment/reproduce_matrix.py --dry-run
  python3 ../environment/reproduce_matrix.py --host http://127.0.0.1:1234 [--models-dir <LM Studio models>] [--resume]

Reads ../protocol/protocol.yaml and enumerates the cells: kind `full` = every pre-registered policy x primary dataset x seed x model minus `matrix.exclusions`
(model-major order), then kind `sensitivity` = every `sensitivity_cells` group with its own scope. Within each kind the
`unguarded` cell of every (dataset, model, seed) runs before the other policies, so the shared first-pass cache is
written once and every policy scores the same answer. Groups marked `requires_full` run only when every full cell is ok
under the current runner (otherwise they are reported as deferred).

Each cell runs the protocol's `execution.command` in `execution.workdir` (artifact/) with the environment of
manuscript/COMMANDS.md (PAPER_LMSTUDIO_HOST, PAPER_LIMIT=1500, PAPER_NCE_LIMIT=450, policy types fixed in the command;
TELEQNA_PATH / NETCONFEVAL_PATH pass through). Outputs go to experiments/runs/<run_id>/ for a cell's first execution and
experiments/runs/<run_id>/r<N>/ for re-executions (N = one more than the largest r<N> suffix found on disk or in the registry for that run_id; a base directory that already
holds outputs counts as 1, so a directory is never reused). A record
with the fields of the README's registry schema (plus provenance fields) is appended to
experiments/registry.json under a file lock and copied to <out_dir>/run.json. Failures are recorded too. Afterwards
analyze.py and audit_scorer.py are run.

After every full and sensitivity cell is ok under the current runner, the driver runs the POST-HOC option-aware verifier
control (kind `posthoc_control`, environment/control_option_aware.py, which registers its own record) for the datasets,
seeds and models listed in protocol `posthoc_controls.verifier_gated_opt.scope` (read from protocol.yaml, not hard-coded);
with --resume a model that already has an ok posthoc_control record under the current runner AND the current control script hash (control_script_sha256) or one listed in protocol posthoc_controls.verifier_gated_opt.script_sha256_history is skipped; any other or a missing script hash means not done. If any
full or sensitivity cell is not ok, the controls are reported as deferred. The control scores the cached first-pass answers of
the full cells and makes no first-pass call. control_option_aware.py registers its records through the standalone
registry code of this file (or an optional external executor package when one is importable). A record whose
control_script_sha256 is listed in protocol
`posthoc_controls.verifier_gated_opt.script_sha256_history` counts as done like one under the current script hash.

Before any cell, check_benchmarks.py verifies the benchmark checkouts (skip with --skip-checks) and, when --models-dir is
given, verify_models.py verifies the GGUF files.

--dry-run makes no model call and writes environment/reproduce_dry_run.json (every cell's run_id, kind, command, env,
plus counts; the control runs are listed with kind posthoc_control; protocol_sha256 is the hash of the current protocol.yaml and
protocol_frozen_sha256 the content of protocol/FROZEN.sha256). --only <substring> restricts to run_ids containing it. --resume
skips cells (and controls) already ok under the current runner hash (controls: and the current control script hash or one in its script_sha256_history). In a dry run the posthoc_control rows mirror the real
precondition (every full and sensitivity cell ok under the current runner after the matrix phase, assuming each executed cell
succeeds): they are would_run only when the precondition would be satisfied, otherwise deferred, and `condition` says which.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
LANE = ENV_DIR.parent
ARTIFACT = LANE / "artifact"
PROTOCOL = LANE / "protocol" / "protocol.yaml"
FROZEN = LANE / "protocol" / "FROZEN.sha256"
CONTROL_SCRIPT = ENV_DIR / "control_option_aware.py"
CONTROL_METHOD = "verifier_gated_opt"
REGISTRY = LANE / "experiments" / "registry.json"
DRY_RUN_OUT = ENV_DIR / "reproduce_dry_run.json"
ANALYSIS_ONLY = {"analyze.py", "README.md", "requirements.txt"}  # files that never influence a run's outputs
DEFAULT_HOST = "http://127.0.0.1:1234"
# the environment of manuscript/COMMANDS.md (the command template also sets these as shell defaults)
CELL_ENV = {"PAPER_LIMIT": "1500", "PAPER_NCE_LIMIT": "450", "PAPER_NCE_POLICIES": "reachability,waypoint,loadbalancing"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_sha256(runner_only: bool = False) -> str:
    """The README recipe: SHA-256 over every file under artifact/ (no __pycache__/.pyc) in sorted relative-path order,
    feeding each relative path string, then the file bytes. runner_only drops analyze.py, README.md, requirements.txt."""
    h = hashlib.sha256()
    for p in sorted(x for x in ARTIFACT.rglob("*") if x.is_file() and "__pycache__" not in x.parts and not x.name.endswith(".pyc")):
        if runner_only and p.name in ANALYSIS_ONLY:
            continue
        h.update(str(p.relative_to(ARTIFACT)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def load_protocol() -> dict:
    import yaml
    return yaml.safe_load(PROTOCOL.read_text(encoding="utf-8")) or {}


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {"runs": []}


# ----------------------------------------------------------------------------- enumeration
def is_excluded(protocol: dict, method: str, dataset: str, model: str) -> bool:
    for e in (protocol.get("matrix") or {}).get("exclusions") or []:
        if isinstance(e, dict) and all(str(e[k]) == v for k, v in (("method", method), ("dataset", dataset), ("model", model)) if e.get(k) not in (None, "")):
            return True
    return False


def cells_for(protocol: dict, kind: str, excluded: list | None = None) -> list[dict]:
    """kind full or sensitivity; cells removed by matrix.exclusions are appended to `excluded` when given."""
    datasets = [str(d.get("id")) for d in protocol.get("datasets") or [] if str(d.get("role") or "primary") != "sensitivity"]
    seeds = list(protocol.get("seeds") or [1])
    models = [str(m.get("id") if isinstance(m, dict) else m) for m in protocol.get("models_or_configs") or []] or ["default"]
    cells: list[dict] = []
    if kind == "sensitivity":
        sens = protocol.get("sensitivity_cells") or {}
        for g in (sens if isinstance(sens, list) else [sens]):
            scope = g.get("scope") or {}
            for m in [str(x) for x in g.get("methods") or []]:
                for d in [str(x) for x in (scope.get("datasets") or datasets)]:
                    for s in list(scope.get("seeds") or seeds):
                        for mo in [str(x) for x in (scope.get("models") or models)]:
                            cells.append({"run_id": f"{m}-{d}-s{s}-{mo}".replace("/", "_"), "kind": kind, "group": g.get("id"), "method": m, "dataset": d, "seed": s, "model": mo, "requires_full": bool(g.get("requires_full"))})
        return cells
    methods = [str(b.get("id")) for b in protocol.get("baselines") or []] + [str((protocol.get("method") or {}).get("id") or "method")]
    for mo in models:  # model-major: one weight swap per model
        for d in datasets:
            for s in seeds:
                for m in methods:
                    cell = {"run_id": f"{m}-{d}-s{s}-{mo}".replace("/", "_"), "kind": kind, "method": m, "dataset": d, "seed": s, "model": mo}
                    if is_excluded(protocol, m, d, mo):
                        if excluded is not None:
                            excluded.append(cell)
                        continue
                    cells.append(cell)
    return cells


def control_cells(protocol: dict) -> list[dict]:
    """Post-hoc controls registered in protocol `posthoc_controls.verifier_gated_opt.scope` (datasets x seeds x models, as listed there)."""
    spec = (protocol.get("posthoc_controls") or {}).get(CONTROL_METHOD)
    if not spec:
        return []
    scope = spec.get("scope") or {}
    datasets = [str(x) for x in scope.get("datasets") or []]
    if datasets != ["teleqna"]:
        raise SystemExit(f"{CONTROL_SCRIPT.name} scores TeleQnA only; the protocol scope lists datasets {datasets}")
    return [{"run_id": f"{CONTROL_METHOD}-{d}-s{s}-{mo}".replace("/", "_"), "kind": "posthoc_control", "method": CONTROL_METHOD, "dataset": d, "seed": s, "model": mo}
            for d in datasets for s in scope.get("seeds") or [] for mo in [str(x) for x in scope.get("models") or []]]


def control_command(cell: dict, host: str, timeout: int) -> list[str]:
    """The argv executed for a control; --dry-run shows exactly this."""
    return ["python3", str(CONTROL_SCRIPT.relative_to(LANE)), "--model", cell["model"], "--seed", str(cell["seed"]), "--host", host, "--timeout", str(timeout)]


def control_script_history(protocol: dict) -> set:
    """Earlier control-script hashes that the protocol accepts as completing a control: posthoc_controls.verifier_gated_opt.
    script_sha256_history (a list; empty when absent). A script revision that changes only where the script's registry code
    comes from lists the hash under which the control records were executed here."""
    spec = (protocol.get("posthoc_controls") or {}).get(CONTROL_METHOD) or {}
    return {str(h) for h in spec.get("script_sha256_history") or [] if h}


def control_done(record: dict, runner: str, script_sha: str, history: set | frozenset = frozenset()) -> bool:
    """The single predicate deciding whether a registry record completes a post-hoc control, used by the dry run and by real
    execution alike: kind posthoc_control, status ok, the current runner hash, AND control_script_sha256 equal to the sha256 of the
    current environment/control_option_aware.py or listed in the protocol's script_sha256_history (control_script_history; the
    control's behaviour lives in that file, outside the runner hash; a record with any other or a missing script hash is stale).
    Deliberately NOT required: record config_sha256 == current protocol hash. The protocol may be re-frozen after text-only
    edits, which must not invalidate completed controls."""
    return (record.get("kind") == "posthoc_control" and record.get("status") == "ok" and record.get("runner_sha256") == runner
            and (record.get("control_script_sha256") == script_sha or (bool(record.get("control_script_sha256")) and record.get("control_script_sha256") in history)))


def stale_control_ids(registry: dict, runner: str, script_sha: str, history: set | frozenset = frozenset()) -> set:
    """Controls with an ok record under the current runner whose control script hash is neither current nor in the history (not done)."""
    return {r.get("run_id") for r in registry.get("runs") or []
            if r.get("kind") == "posthoc_control" and r.get("status") == "ok" and r.get("runner_sha256") == runner and not control_done(r, runner, script_sha, history)}


def control_condition(pending: bool, gate_now: bool, gate_after_matrix: bool, matrix_will_run: bool, stale: bool = False) -> str:
    if not pending:
        return "skipped: already ok under the current runner and the current control script or a hash in its script_sha256_history (--resume)"
    if stale:
        prefix = "stale control script: the registry record's control_script_sha256 differs from (or is missing against) the current control_option_aware.py and is not in script_sha256_history, so the control is not done; "
    else:
        prefix = ""
    return prefix + _control_gate_condition(gate_now, gate_after_matrix, matrix_will_run)


def _control_gate_condition(gate_now: bool, gate_after_matrix: bool, matrix_will_run: bool) -> str:
    if not gate_after_matrix:
        return "deferred: some full or sensitivity cell is not ok under the current runner and would not be (re)executed in this invocation, so the real run defers every control"
    if gate_now and not matrix_will_run:
        return "ready now: every full and sensitivity cell is ok under the current runner"
    return "conditional on matrix completion: runs only if every executed full and sensitivity cell ends ok under the current runner (otherwise deferred)"


def ordered(protocol: dict, cells: list[dict]) -> list[dict]:
    """Execution order: cells of the first-listed baseline (unguarded: the shared first pass) before every other cell."""
    first = str((protocol.get("baselines") or [{}])[0].get("id") or "")
    return [c for c in cells if c["method"] == first] + [c for c in cells if c["method"] != first]


# ----------------------------------------------------------------------------- execution
def out_dir_for(run_id: str, registry: dict) -> Path:
    """Never reuse a directory. The first execution of a cell uses runs/<run_id>/ when that directory
    holds no outputs and no registry record exists; otherwise runs/<run_id>/r<N>/ with N one more than the largest suffix
    found on disk (the base directory counts as 1) and in the registry records of this run_id."""
    base = LANE / "experiments" / "runs" / run_id
    suffixes = [0]
    if base.is_dir():
        for p in base.iterdir():
            m = re.fullmatch(r"r(\d+)", p.name)
            if p.is_dir() and m:
                suffixes.append(int(m.group(1)))
            elif p.is_file():
                suffixes.append(1)  # the base directory already holds outputs
    pat = re.compile(r"/" + re.escape(run_id) + r"/r(\d+)/")
    for r in registry.get("runs") or []:
        if r.get("run_id") != run_id:
            continue
        found = [int(m.group(1)) for o in r.get("outputs") or [] for m in [pat.search(str(o))] if m]
        suffixes.append(max(found) if found else 1)
    top = max(suffixes)
    return base if top == 0 else base / f"r{top + 1}"


def render_command(template: str, cell: dict, out_dir: Path, workdir: Path) -> str:
    """Substitute only the placeholders so shell syntax such as ${VAR:-default} survives; paths are relative to workdir."""
    values = {"method": cell["method"], "dataset": cell["dataset"], "seed": cell["seed"], "model": cell["model"], "run_id": cell["run_id"],
              "out_dir": os.path.relpath(out_dir, workdir), "lane_dir": os.path.relpath(LANE, workdir)}
    return re.sub(r"(?<!\$)\{(method|dataset|seed|model|out_dir|lane_dir|run_id)\}", lambda m: str(values[m.group(1)]), template)


def cell_env(host: str, cell: dict, out_dir: Path) -> dict[str, str]:
    env = {"PAPER_LMSTUDIO_HOST": host, **CELL_ENV, "PAPER_RUN_ID": cell["run_id"], "PAPER_OUT_DIR": str(out_dir.relative_to(LANE))}
    for key in ("TELEQNA_PATH", "NETCONFEVAL_PATH"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def append_run(run: dict, out_dir: Path) -> None:
    lock_path = REGISTRY.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = load_registry()
        # append-only: every run, including failures and earlier records of the same run_id, stays in the registry
        registry.setdefault("runs", []).append(run)
        tmp = REGISTRY.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, REGISTRY)
        fcntl.flock(lock, fcntl.LOCK_UN)
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def run_cell(protocol: dict, cell: dict, host: str, timeout: int, retries: int) -> dict:
    execution = protocol.get("execution") or {}
    workdir = LANE / str(execution.get("workdir") or "artifact")
    out_dir = out_dir_for(cell["run_id"], load_registry())
    out_dir.mkdir(parents=True, exist_ok=True)
    command = render_command(str(execution["command"]), cell, out_dir, workdir)
    env = {k: v for k, v in os.environ.items() if k != "PAPER_CATEGORY"}  # PAPER_CATEGORY would filter the TeleQnA frame
    env.update(cell_env(host, cell, out_dir))
    started, attempts, status, error, result = utc_now(), [], "failed", "", {}
    for attempt in range(retries + 1):
        log_path = out_dir / f"attempt-{attempt + 1}.log"
        t0 = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(["/bin/sh", "-c", command], cwd=workdir, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, env=env)
            attempts.append({"attempt": attempt + 1, "exit_code": proc.returncode, "seconds": round(time.time() - t0, 1), "log": str(log_path.relative_to(LANE))})
            result_path = out_dir / "result.json"
            if proc.returncode == 0 and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(result.get("metric_values"), dict) and result["metric_values"]:
                    status = "ok"
                    break
                error = "result.json has no metric_values"
            else:
                error = f"exit {proc.returncode}" if proc.returncode != 0 else "no result.json"
        except subprocess.TimeoutExpired:
            attempts.append({"attempt": attempt + 1, "exit_code": None, "timeout": timeout, "log": str(log_path.relative_to(LANE))})
            error = f"timeout after {timeout}s"
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))
    run = {
        "run_id": cell["run_id"], "kind": cell["kind"], "method": cell["method"], "dataset": cell["dataset"], "model": cell["model"], "seed": cell["seed"],
        "status": status, "evaluator_id": result.get("evaluator_id") or ((protocol.get("metrics") or [{}])[0].get("evaluator_id")), "metric_values": result.get("metric_values") or {}, "n": result.get("n"),
        "outputs": sorted(str(p.relative_to(LANE)) for p in out_dir.rglob("*") if p.is_file() and p.name != "run.json"), "command": command,
        "config_sha256": sha256_file(PROTOCOL), "artifact_sha256": artifact_sha256(), "runner_sha256": artifact_sha256(runner_only=True),
        "started_at": started, "finished_at": utc_now(), "hardware": os.uname().nodename, "attempts": attempts, "error": error if status != "ok" else "",
        "extra": {k: v for k, v in result.items() if k not in {"metric_values", "evaluator_id", "n"}}, "driver": "environment/reproduce_matrix.py",
    }
    append_run(run, out_dir)
    return run


def run_checks(args: argparse.Namespace) -> bool:
    ok = True
    if not args.skip_checks:
        ok &= subprocess.run([sys.executable, str(ENV_DIR / "check_benchmarks.py")]).returncode == 0
    if args.models_dir:
        ok &= subprocess.run([sys.executable, str(ENV_DIR / "verify_models.py"), "--models-dir", str(args.models_dir)]).returncode == 0
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="no model calls; write environment/reproduce_dry_run.json")
    parser.add_argument("--only", default="", help="run only cells whose run_id contains this substring")
    parser.add_argument("--resume", action="store_true", help="skip cells already ok under the current runner hash")
    parser.add_argument("--host", default=os.environ.get("PAPER_LMSTUDIO_HOST", DEFAULT_HOST), help="LM Studio server URL (sets PAPER_LMSTUDIO_HOST)")
    parser.add_argument("--models-dir", type=Path, help="LM Studio models directory: also run verify_models.py")
    parser.add_argument("--skip-checks", action="store_true", help="do not run check_benchmarks.py")
    parser.add_argument("--no-analysis", action="store_true", help="do not run analyze.py and audit_scorer.py afterwards")
    parser.add_argument("--timeout", type=int, default=14400, help="seconds per attempt (the reported execution used 14400)")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args(argv)
    args.host = args.host.rstrip("/")

    protocol = load_protocol()
    excluded: list[dict] = []
    full = ordered(protocol, cells_for(protocol, "full", excluded))
    sensitivity = ordered(protocol, cells_for(protocol, "sensitivity"))
    cells = full + sensitivity
    controls = control_cells(protocol)
    runner = artifact_sha256(runner_only=True)
    done = {r.get("run_id") for r in load_registry().get("runs") or [] if r.get("status") == "ok" and r.get("runner_sha256") == runner}
    selected = [c for c in cells if args.only in c["run_id"]]
    pending = [c for c in selected if not (args.resume and c["run_id"] in done)]
    script_sha = sha256_file(CONTROL_SCRIPT)
    script_history = control_script_history(protocol)
    registry_now = load_registry()
    done_controls = {r.get("run_id") for r in registry_now.get("runs") or [] if control_done(r, runner, script_sha, script_history)}
    stale_controls = stale_control_ids(registry_now, runner, script_sha, script_history) - done_controls
    sel_controls = [c for c in controls if args.only in c["run_id"]]
    pending_controls = [c for c in sel_controls if not (args.resume and c["run_id"] in done_controls)]
    # the real precondition: every full and sensitivity cell (not only the selected ones) ok under the current runner after the matrix phase;
    # a dry run assumes each cell the matrix phase would execute succeeds, and a requires_full cell is skipped unless every full cell is ok
    full_ids = {c["run_id"] for c in full}
    pending_ids = {c["run_id"] for c in pending}
    full_ok_after = full_ids <= (done | pending_ids)
    runs_ids = {c["run_id"] for c in pending if full_ok_after or not c.get("requires_full")}
    gate_now = {c["run_id"] for c in cells} <= done
    gate_after_matrix = {c["run_id"] for c in cells} <= (done | runs_ids)
    matrix_will_run = bool(runs_ids)

    checks_ok = run_checks(args) if (not args.skip_checks or args.models_dir) else None
    if args.dry_run:
        workdir = LANE / str((protocol.get("execution") or {}).get("workdir") or "artifact")
        registry = load_registry()
        rows = []
        for i, c in enumerate(selected):
            out_dir = out_dir_for(c["run_id"], registry)
            rows.append({"order": i + 1, **{k: c[k] for k in ("run_id", "kind", "method", "dataset", "seed", "model")}, "group": c.get("group"),
                         "requires_full": c.get("requires_full", False), "done_under_current_runner": c["run_id"] in done,
                         "would_run": c in pending, "workdir": str(workdir.relative_to(LANE)), "out_dir": str(out_dir.relative_to(LANE)),
                         "command": render_command(str(protocol["execution"]["command"]), c, out_dir, workdir), "env": cell_env(args.host, c, out_dir)})
        for c in sel_controls:
            out_dir = out_dir_for(c["run_id"], registry)
            rows.append({"order": len(rows) + 1, **{k: c[k] for k in ("run_id", "kind", "method", "dataset", "seed", "model")}, "group": "posthoc_controls.verifier_gated_opt",
                         "requires_full": True, "done_under_current_runner": c["run_id"] in done_controls,
                         "would_run": c in pending_controls and gate_after_matrix, "deferred": c in pending_controls and not gate_after_matrix,
                         "stale_control_script": c["run_id"] in stale_controls,
                         "condition": control_condition(c in pending_controls, gate_now, gate_after_matrix, matrix_will_run, c["run_id"] in stale_controls),
                         "workdir": ".", "out_dir": str(out_dir.relative_to(LANE)), "command": shlex.join(control_command(c, args.host, args.timeout)),
                         "env": {"PAPER_LMSTUDIO_HOST": args.host}})
        groups: dict[str, int] = {}
        for c in sensitivity:
            groups[str(c["group"])] = groups.get(str(c["group"]), 0) + 1
        report = {
            "generated_at": utc_now(), "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": runner,
            "protocol_frozen_sha256": FROZEN.read_text().strip() if FROZEN.exists() else None,
            "flags": {"only": args.only, "resume": args.resume, "host": args.host},
            "ordering": "per kind (full, then sensitivity): unguarded cells first (shared first pass), then the other policies; full cells model-major; then the post-hoc controls (kind posthoc_control) once every full and sensitivity cell is ok",
            "counts": {"full": len(full), "sensitivity": len(sensitivity), "total": len(cells), "sensitivity_by_group": groups,
                       "matrix_cells_expected": (protocol.get("matrix") or {}).get("cells_expected"), "excluded_by_matrix_exclusions": len(excluded),
                       "selected": len(selected), "done_under_current_runner": sum(1 for c in selected if c["run_id"] in done), "would_run": len(pending),
                       "posthoc_control": len(controls), "posthoc_control_selected": len(sel_controls), "posthoc_control_done_under_current_runner": sum(1 for c in sel_controls if c["run_id"] in done_controls),
                       "posthoc_control_would_run": sum(1 for c in pending_controls if gate_after_matrix),
                       "posthoc_control_deferred": sum(1 for c in pending_controls if not gate_after_matrix)},
            "posthoc_control_gate": {"all_full_and_sensitivity_ok_now": gate_now, "all_ok_after_matrix_phase_if_every_executed_cell_succeeds": gate_after_matrix,
                                     "matrix_cells_that_would_execute": len(runs_ids)},
            "excluded": [c["run_id"] for c in excluded],
            "benchmark_and_model_checks": {None: "skipped", True: "PASS", False: "FAIL"}[checks_ok],
            "cells": rows,
        }
        DRY_RUN_OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(report["counts"]))
        print(f"wrote {DRY_RUN_OUT}")
        return 0
    if checks_ok is False:
        print("checks failed: fix the benchmark checkouts / model files or pass --skip-checks", file=sys.stderr)
        return 1

    results, deferred = [], []
    for c in pending:
        if c.get("requires_full"):
            current = artifact_sha256(runner_only=True)
            done_now = {r.get("run_id") for r in load_registry().get("runs") or [] if r.get("status") == "ok" and r.get("runner_sha256") == current}
            if not full_ids <= done_now:
                deferred.append(c["run_id"])
                continue
        print(f"[{len(results) + 1}/{len(pending)}] {c['run_id']}", flush=True)
        run = run_cell(protocol, c, args.host, args.timeout, args.retries)
        print(f"    {run['status']} {run['error']}", flush=True)
        results.append(run)
    ok = sum(r["status"] == "ok" for r in results)
    # final phase: the post-hoc option-aware verifier control, only when every full and sensitivity cell is ok under the current runner
    control_results, controls_deferred = [], []
    if pending_controls:
        current = artifact_sha256(runner_only=True)
        done_now = {r.get("run_id") for r in load_registry().get("runs") or [] if r.get("status") == "ok" and r.get("runner_sha256") == current}
        if not {c["run_id"] for c in cells} <= done_now:
            controls_deferred = [c["run_id"] for c in pending_controls]
        else:
            for c in pending_controls:
                print(f"[control {len(control_results) + 1}/{len(pending_controls)}] {c['run_id']}", flush=True)
                rc = subprocess.run(control_command(c, args.host, args.timeout), cwd=LANE).returncode
                print(f"    {'ok' if rc == 0 else 'failed'}", flush=True)
                control_results.append(rc == 0)
    summary = {"finished_at": utc_now(), "selected": len(selected), "executed": len(results), "ok": ok, "failed": len(results) - ok, "deferred_until_full": deferred,
               "controls_selected": len(sel_controls), "controls_executed": len(control_results), "controls_ok": sum(control_results), "controls_deferred": controls_deferred}
    print(json.dumps(summary))
    status = 0 if ok == len(results) and not deferred and all(control_results) and not controls_deferred else 1
    if not args.no_analysis:
        teleqna = os.environ.get("TELEQNA_PATH")
        for cmd in ([sys.executable, "analyze.py"], [sys.executable, "audit_scorer.py"] + ([teleqna] if teleqna else [])):
            print("$", " ".join(Path(x).name if x == sys.executable else x for x in cmd), flush=True)
            status |= subprocess.run(cmd, cwd=ARTIFACT).returncode != 0
    return status


if __name__ == "__main__":
    sys.exit(main())
