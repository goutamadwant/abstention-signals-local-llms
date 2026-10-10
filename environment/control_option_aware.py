#!/usr/bin/env python3
"""POST-HOC option-aware verifier control on TeleQnA.

The pre-registered `verifier_gated` policy shows the verifier the question and the proposed answer ("option k: text")
but NOT the other answer options. This control, `verifier_gated_opt`, is identical in every other respect: same system
message and YES/NO wording, temperature 0 (run_cell.VERIFIER_TEMPERATURE), max 3 tokens, top-20 alternatives, the same
score (run_cell.p_yes_from: sigmoid(log P(YES) - log P(NO)) with tokenization variants aggregated), the same threshold
(0.5), the SAME request seeds (run_cell.request_seed with the role "verifier" that verifier_gated passes, so item by item
the seed is identical to the pre-registered verifier's), and it scores the SAME cached v2
first-pass answers of the given (model, seed). The only change: the question's answer options are listed (in the
first-pass prompt's format, "option N: text") between the question and the proposed answer.

The cell logic is NOT copied: run_cell is imported from ../artifact and its `run()` executes the cell (frame loading,
cache naming, row schema, metrics, token and latency accounting, result.json / predictions.jsonl) with two hooks:
  - `signal` gains the method `verifier_gated_opt` (every other method is delegated unchanged);
  - `first_pass` is replaced by a cache-only lookup: a missing cache file or a missing item stops the cell before any
    model call. A first pass is never regenerated.
Nothing under artifact/ is modified, so the runner hash of the 140 registered runs is unchanged; every record of this
script carries that runner hash (runner_sha256) and the hash of this file (control_script_sha256).

Registration (append-only, failures included): through an optional external executor package when it is importable,
otherwise through the standalone path (hash recipes from environment/reproduce_matrix.py; monotonic directory reservation
with a pending "started" run.json, base out_dir counted as attempt 1, a locked append that replaces the attempt's own
rebuild_registry placeholder, atomic registry.json and run.json writes). Both paths write the same record fields and
hashes; prompts, scoring, request seeds and outputs do not depend on the path.
  run_id  verifier_gated_opt-teleqna-s<seed>-<model>          kind posthoc_control   (full 1,500-item frame)
  run_id  smoke-verifier_gated_opt-teleqna-s<seed>-<model>    kind smoke             (--smoke --limit N)
Outputs: experiments/runs/<run_id>/ (re-executions: experiments/runs/<run_id>/r<N>/), with attempt-1.log, result.json,
predictions.jsonl and run.json. The registry kinds keep the records out of every matrix consumer: analyze.py excludes
kind smoke and kind posthoc_control from the pre-registered cells and reads posthoc_control only in its option_aware_control
block; the matrix driver (reproduce_matrix.cells_for) enumerates only protocol-matrix run_ids, which never start with
verifier_gated_opt.

A non-smoke (posthoc_control) execution requires the full frame (no --limit other than 1500), a frozen protocol
(protocol/FROZEN.sha256 equal to the current protocol.yaml) and a protocol that registers the control (the text
`verifier_gated_opt` appears in protocol.yaml). Smoke runs need neither.

Usage (from anywhere):
  python3 environment/control_option_aware.py --smoke --limit 20 --model qwen2.5-coder-7b-instruct --seed 1 --host http://127.0.0.1:11435
  python3 environment/control_option_aware.py --model qwen3-coder-30b-a3b-instruct --seed 1 --host http://127.0.0.1:1234
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ENV_DIR = Path(__file__).resolve().parent
LANE = ENV_DIR.parent
REPO = LANE.parents[1]
ARTIFACT = LANE / "artifact"
PROTOCOL = LANE / "protocol" / "protocol.yaml"
METHOD = "verifier_gated_opt"
BASE_METHOD = "verifier_gated"
ROLE = "verifier"  # the request-seed role run_cell.signal passes for verifier_gated (vrole = "verifier"): listing the options is the only difference
DATASET = "teleqna"
FRAME_SIZE = 1500  # PAPER_LIMIT default of protocol.execution.command: the primary TeleQnA frame
DEFAULT_HOST = os.environ.get("PAPER_LMSTUDIO_HOST", "http://127.0.0.1:1234")
DEFAULT_DATA = os.environ.get("TELEQNA_PATH", str(REPO / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt"))


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


RECORD_FILES = {"run.json", "run.json.tmp"}  # registry record files: never one of a run's outputs


def engine() -> SimpleNamespace:
    """Registry code (hash recipes, attempt reservation, append-only registry writes), built on reproduce_matrix.py."""
    return standalone()


def standalone() -> SimpleNamespace:
    """Standalone registry path. The hash recipes come from reproduce_matrix.py; reservation and final append work as follows:
      - reservation: runs/<run_id>/ only when neither a registry record nor a directory exists for run_id, else r<max+1>;
        a registry record whose out_dir is runs/<run_id> counts as attempt 1 and runs/<run_id>/rN as N, independent of whether
        its outputs exist on disk (a registry-only or redacted copy reserves r2, never the base directory again);
      - append: under an flock, one FINAL record per attempt, append-only except that the same attempt's own rebuild_registry
        placeholder (same record_key, or an "interrupted before start" one in the same directory) is replaced in place;
        registry.json and the attempt's run.json are written atomically (temp file + os.replace)."""
    import fcntl
    if str(ENV_DIR) not in sys.path:
        sys.path.insert(0, str(ENV_DIR))
    import reproduce_matrix as rm
    if rm.LANE != LANE:
        raise SystemExit(f"reproduce_matrix.py resolves the project root to {rm.LANE}, expected {LANE}")
    interrupted, not_started = "interrupted before completion", "interrupted before start"  # error texts of rebuild_registry placeholder records

    def write_json(path: Path, payload: dict) -> None:  # sorted, indented JSON written atomically (temp file + os.replace)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def registry_path(lane: Path) -> Path:
        return lane / "experiments" / "registry.json"

    def load_registry(lane: Path) -> dict:
        path = registry_path(lane)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}

    def record_dir(run: dict) -> str:
        if run.get("out_dir"):
            return str(run["out_dir"])
        out_rel = next((str(o) for o in run.get("outputs") or [] if str(o).endswith(("result.json", "attempt-1.log"))), None)
        return str(Path(out_rel).parent) if out_rel else f"experiments/runs/{run.get('run_id')}"

    def record_key(run: dict) -> tuple[str, str, str]:
        return (str(run.get("run_id") or ""), str(run.get("started_at") or ""), record_dir(run))

    def is_placeholder(run: dict) -> bool:
        return run.get("recovered_by") == "rebuild_registry" and run.get("error") in (interrupted, not_started)

    def same_attempt(a: dict, b: dict) -> bool:
        return a.get("run_id") == b.get("run_id") and record_dir(a) == record_dir(b)

    def reserve_attempt(lane: Path, run_id: str, build_pending):
        base = lane / "experiments" / "runs" / run_id
        used = 0
        pat = re.compile(re.escape(f"experiments/runs/{run_id}") + r"/r(\d+)")
        for r in load_registry(lane).get("runs") or []:  # base out_dir counts 1, rN counts N, whatever is on disk
            if r.get("run_id") != run_id:
                continue
            m = pat.fullmatch(record_dir(r))
            used = max(used, int(m.group(1)) if m else 1)
        if base.exists():
            used = max(used, 1)
            for p in base.iterdir():
                m = re.fullmatch(r"r(\d+)", p.name)
                if m and p.is_dir():
                    used = max(used, int(m.group(1)))
        n = used + 1
        while True:  # reserve atomically: a concurrent executor that picked the same number moves on to the next one
            out_dir = base if n == 1 else base / f"r{n}"
            try:
                out_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                n += 1
                continue
            pending = build_pending(out_dir)
            write_json(out_dir / "run.json", pending)
            return out_dir, pending

    def append_run(lane: Path, run: dict) -> None:
        key = record_key(run)
        lock_path = registry_path(lane).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                registry = load_registry(lane)
                runs = list(registry.get("runs") or [])
                index = next((i for i, r in enumerate(runs) if is_placeholder(r) and (record_key(r) == key or (r.get("error") == not_started and same_attempt(r, run)))), None)
                if index is None:
                    runs.append(run)
                else:
                    runs[index] = run
                registry["runs"] = runs
                write_json(registry_path(lane), registry)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        write_json(lane / record_dir(run) / "run.json", run)  # atomically replaces the attempt's pending ("started") run.json

    return SimpleNamespace(source="standalone (environment/reproduce_matrix.py)", artifact_sha256=lambda lane: rm.artifact_sha256(),
                           runner_sha256=lambda lane: rm.artifact_sha256(runner_only=True), reserve_attempt=reserve_attempt,
                           append_run=append_run,
                           list_outputs=lambda lane, out_dir: sorted(str(p.relative_to(lane)) for p in out_dir.rglob("*") if p.is_file() and p.name not in RECORD_FILES),
                           utc_now=rm.utc_now)


def runner():
    if str(ARTIFACT) not in sys.path:
        sys.path.insert(0, str(ARTIFACT))
    import run_cell
    return run_cell


def cache_path(rc, model: str, seed: int) -> Path:
    # exactly run_cell.run()'s cache name for dataset teleqna (no NetConfEval variant tag)
    return LANE / "experiments" / "cache" / f"firstpass-{DATASET}-{rc.CACHE_SCHEME}-{model}-s{seed}.jsonl"


# ----------------------------------------------------------------------------- inner: one cell (no registry access)
def option_aware_question(item: dict, fp: dict) -> str:
    """verifier_gated's TeleQnA user message with the answer options inserted after the question, in the first-pass
    prompt's format (run_cell.prompt_for)."""
    opts = "\n".join(f"{k}: {v}" for k, v in sorted(item["options"].items(), key=lambda kv: int(kv[0].split()[1])))
    chosen = item["options"].get(f"option {fp['pred']}", "") if fp.get("pred") else "(no answer)"
    return f"Question: {item['question']}\n{opts}\nProposed answer: option {fp.get('pred')}: {chosen}\nIs the proposed answer correct? Reply YES or NO only."


def frame_problem(rc, args: argparse.Namespace) -> str:
    """'' when the cached first pass of (model, seed) covers every item of the frame; otherwise the reason it does not."""
    limit = args.limit or FRAME_SIZE
    cpath = cache_path(rc, args.model, args.seed)
    if not cpath.exists():
        return f"first-pass cache missing: {cpath} (the control never regenerates a first pass)"
    try:
        items = rc.load_teleqna(Path(args.data), limit, None)
    except (OSError, ValueError, KeyError, SystemExit) as exc:
        return f"cannot load the TeleQnA frame from {args.data}: {type(exc).__name__}: {exc}"[:300]
    cache: dict = {}
    rc.refresh_cache(cache, cpath, {"offset": 0})
    missing = [it["id"] for it in items if it["id"] not in cache]
    if missing:
        return f"{len(missing)} of {len(items)} frame items have no cached first pass in {cpath.name} (first: {missing[0]}); the control never regenerates a first pass"
    return ""


def inner(args: argparse.Namespace) -> int:
    rc = runner()
    rc.HOST = args.host.rstrip("/")  # chat() and ensure_model() read the module global at call time
    data = Path(args.data)
    limit = args.limit or FRAME_SIZE
    cpath = cache_path(rc, args.model, args.seed)
    problem = frame_problem(rc, args)
    if problem:
        raise SystemExit(problem)
    cache: dict = {}
    rc.refresh_cache(cache, cpath, {"offset": 0})
    orig_signal = rc.signal

    def signal(method, item, fp, model, seed, cost, a):
        if method != METHOD:
            return orig_signal(method, item, fp, model, seed, cost, a)
        sysmsg = "You are a strict telecommunications standards reviewer. Reply with exactly YES or NO."  # verifier_gated's
        q = option_aware_question(item, fp)
        text, u = rc.chat(model, [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], temperature=rc.VERIFIER_TEMPERATURE,
                          seed=rc.request_seed(seed, item["id"], ROLE), max_tokens=3, logprobs=True)
        cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
        p_yes, p_no, norm = rc.p_yes_from(u)
        return norm, {"verifier_text": text.strip()[:8], "p_yes_raw": p_yes, "p_no_raw": p_no, "p_no_listed": p_no > 0, "verifier_model": model}

    def cached_first_pass(item, model, seed, cache_, cache_path_, cost, state=None):
        if item["id"] not in cache_:
            rc.refresh_cache(cache_, cache_path_, state if state is not None else {})
        if item["id"] not in cache_:
            raise SystemExit(f"first pass for {item['id']} missing from {Path(cache_path_).name}; the control never regenerates a first pass")
        return cache_[item["id"]]

    rc.signal = signal
    rc.first_pass = cached_first_pass
    rc.THRESHOLDS[METHOD] = rc.THRESHOLDS[BASE_METHOD]
    out = Path(args.out)
    rc.run(argparse.Namespace(method=METHOD, dataset=DATASET, seed=args.seed, model=args.model, out=str(out), data=str(data), limit=limit, category=None))
    result_path = out / "result.json"
    result = json.loads(result_path.read_text())
    if Path(result.get("cache") or "").name != cpath.name:
        raise SystemExit(f"run_cell used cache {result.get('cache')}, expected {cpath.name}")
    result["control"] = {"post_hoc": True, "label": "POST-HOC", "base_method": BASE_METHOD, "verifier_role": ROLE,
                         "change": "verifier prompt lists every answer option of the question before the proposed answer; wording, temperature, top-20 scoring, threshold, request seeds (same role, so the same seed per item) and cached first pass as verifier_gated",
                         "first_pass": "cached v2 first pass only (never regenerated)", "runner_sha256": engine().runner_sha256(LANE),
                         "control_script_sha256": sha256_path(Path(__file__).resolve()), "frame_limit": limit}
    result_path.write_text(json.dumps(result, indent=1))
    return 0


# ----------------------------------------------------------------------------- outer: registered execution
def protocol_ready() -> str:
    """'' when a posthoc_control execution may run; otherwise the reason it may not."""
    frozen = LANE / "protocol" / "FROZEN.sha256"
    if not frozen.exists() or frozen.read_text().strip() != sha256_path(PROTOCOL):
        return "protocol/protocol.yaml is not frozen (FROZEN.sha256 differs): freeze the protocol before running the control"
    if METHOD not in PROTOCOL.read_text(encoding="utf-8"):
        return f"protocol/protocol.yaml does not register {METHOD}: add the control to the protocol and freeze it first"
    return ""


def refusal_reason(args: argparse.Namespace) -> str:
    """'' when the execution may start; otherwise why it is refused (protocol not frozen, method not in the protocol, bad
    --limit, missing first-pass cache or frame mismatch). A refusal is registered as a failed record, never silent."""
    if args.smoke:
        if not args.limit or not 1 <= args.limit <= FRAME_SIZE:
            return f"--smoke needs --limit N with 1 <= N <= {FRAME_SIZE} (got {args.limit})"
    else:
        if args.limit not in (None, FRAME_SIZE):
            return f"a posthoc_control run scores the full {FRAME_SIZE}-item frame; --limit is for --smoke only (got {args.limit})"
        reason = protocol_ready()
        if reason:
            return reason
    return frame_problem(runner(), args)


def execute(args: argparse.Namespace) -> int:
    ex = engine()
    utc_now = ex.utc_now
    print(f"registry code: {ex.source}", file=sys.stderr)
    kind = "smoke" if args.smoke else "posthoc_control"
    run_id = (("smoke-" if args.smoke else "") + f"{METHOD}-{DATASET}-s{args.seed}-{args.model}").replace("/", "_")
    script = Path(__file__).resolve()
    hashes = {"config_sha256": sha256_path(PROTOCOL), "artifact_sha256": ex.artifact_sha256(LANE), "runner_sha256": ex.runner_sha256(LANE),
              "control_script_sha256": sha256_path(script)}
    evaluator = "e_teleqna"
    started = utc_now()

    def command_for(out_dir: Path) -> list[str]:
        cmd = ["python3", str(script.relative_to(LANE)), "--inner", "--model", args.model, "--seed", str(args.seed), "--host", args.host,
               "--out", str(out_dir.relative_to(LANE))]
        if args.data != DEFAULT_DATA:  # the default (TELEQNA_PATH or the repo's runtime checkout) is resolved again by the inner run
            cmd += ["--data", args.data]
        return cmd + (["--limit", str(args.limit)] if args.limit else [])

    def build_pending(out_dir: Path) -> dict:
        return {"run_id": run_id, "kind": kind, "method": METHOD, "dataset": DATASET, "model": args.model, "seed": args.seed, "status": "started",
                "evaluator_id": evaluator, "metric_values": {}, "outputs": [], "command": shlex.join(command_for(out_dir)), **hashes,
                "started_at": started, "hardware": os.uname().nodename, "out_dir": str(out_dir.relative_to(LANE)), "attempts": [], "error": "", "extra": {},
                "driver": "environment/control_option_aware.py", "post_hoc": True}

    out_dir, pending = ex.reserve_attempt(LANE, run_id, build_pending)
    refusal = refusal_reason(args)
    if refusal:  # every attempt is registered: a refusal after argument parsing leaves a failed record that states why (exit code stays non-zero)
        run = {**pending, "status": "failed", "finished_at": utc_now(), "attempts": [], "outputs": ex.list_outputs(LANE, out_dir), "error": f"refused: {refusal}"[:600]}
        ex.append_run(LANE, run)
        print(f"refused: {refusal}", file=sys.stderr)
        print(json.dumps({"run_id": run_id, "kind": kind, "status": "failed", "error": run["error"], "out_dir": run["out_dir"]}))
        return 1
    log_path = out_dir / "attempt-1.log"
    status, error, result, attempts = "failed", "", {}, []
    t0 = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable] + command_for(out_dir)[1:], cwd=LANE, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                  env={**os.environ, "PAPER_LMSTUDIO_HOST": args.host, "PAPER_RUN_ID": run_id, "PAPER_OUT_DIR": str(out_dir.relative_to(LANE))})
        attempts.append({"attempt": 1, "exit_code": proc.returncode, "seconds": round(time.time() - t0, 1), "log": str(log_path.relative_to(LANE))})
        result_path, pred_path = out_dir / "result.json", out_dir / "predictions.jsonl"
        if proc.returncode != 0 or not result_path.exists():
            tail = log_path.read_text(errors="replace").strip().splitlines()[-1:] if log_path.exists() else []
            error = (f"exit {proc.returncode}" if proc.returncode != 0 else "no result.json") + (f": {tail[0][:200]}" if tail else "")
        else:
            result = json.loads(result_path.read_text())
            rows = [json.loads(line) for line in pred_path.read_text().splitlines()] if pred_path.exists() else []
            if not (isinstance(result.get("metric_values"), dict) and result["metric_values"]):
                error = "result.json has no metric_values"
            elif len(rows) != result.get("n") or any(r.get("score") is None or "correct" not in r for r in rows):
                error = f"predictions.jsonl has {len(rows)} parsable rows, result.json n={result.get('n')}"
            elif ex.runner_sha256(LANE) != hashes["runner_sha256"] or sha256_path(script) != hashes["control_script_sha256"]:
                error = "runner or control script changed during execution"
            else:
                status = "ok"
    except subprocess.TimeoutExpired:
        attempts.append({"attempt": 1, "exit_code": None, "timeout": args.timeout, "log": str(log_path.relative_to(LANE))})
        error = f"timeout after {args.timeout}s"
    except Exception as exc:  # recorded, never swallowed: the record is failed and the error says why
        error = f"{type(exc).__name__}: {exc}"[:300]
    run = {**pending, "status": status, "evaluator_id": result.get("evaluator_id") or evaluator, "metric_values": result.get("metric_values") or {}, "n": result.get("n"),
           "outputs": ex.list_outputs(LANE, out_dir), "config_sha256": sha256_path(PROTOCOL), "artifact_sha256": ex.artifact_sha256(LANE),
           "runner_sha256": hashes["runner_sha256"], "finished_at": utc_now(), "attempts": attempts, "error": error if status != "ok" else "",
           "extra": {k: v for k, v in result.items() if k not in {"metric_values", "evaluator_id", "n"}}}
    ex.append_run(LANE, run)
    print(json.dumps({"run_id": run_id, "kind": kind, "status": status, "error": run["error"], "out_dir": run["out_dir"], "n": run["n"], "metric_values": run["metric_values"]}))
    return 0 if status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--host", default=DEFAULT_HOST, help="LM Studio server URL")
    ap.add_argument("--limit", type=int, default=None, help="smoke only: stratified frame of N items (run_cell.load_teleqna)")
    ap.add_argument("--smoke", action="store_true", help="registered smoke run (kind smoke, run_id prefix smoke-)")
    ap.add_argument("--timeout", type=int, default=14400, help="seconds for the cell")
    ap.add_argument("--data", default=DEFAULT_DATA, help="TeleQnA.txt (default: TELEQNA_PATH, else the repo's runtime checkout, as run_cell.py)")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)  # the registered command executed by the outer process
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.host = args.host.rstrip("/")
    if args.inner:
        if not args.out:
            raise SystemExit("--inner needs --out")
        args.out = str((LANE / args.out) if not Path(args.out).is_absolute() else Path(args.out))
        return inner(args)
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
