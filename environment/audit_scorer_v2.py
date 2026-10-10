#!/usr/bin/env python3
"""Scorer audit v2: the TeleQnA option-id scorer against an independent parser, on EVERY raw output of the REPORTED execution.

Why a v2: artifact/audit_scorer.py (part of the hashed artifact, kept unchanged) predates the reported execution. It globs every
firstpass-teleqna-*.jsonl cache (the 10 v2 caches of the reported execution AND 13 stale first-execution caches), ignores TELEQNA_PATH unless
given as argv, and writes a different file than the stored experiments/scorer_audit.json. This script fixes the scope:

 * audited pool = the TeleQnA first-pass caches referenced (`extra.cache` basename) by the registry's CURRENT-runner `ok` records of the
   analysis's own selection rule (analyze.py: status ok, kind not smoke, not posthoc_control, runner_sha256 == the current runner hash,
   dataset teleqna or teleqna_full); stale first-execution caches are never read;
 * the benchmark file is resolved exactly as run_cell.py does: $TELEQNA_PATH (the path of the TeleQnA.txt FILE, not a directory), else
   <repo>/runtime/ground_truth/TeleQnA/TeleQnA.txt; a directory is rejected with an error, as run_cell.py would fail on it;
 * every cached row is audited (no sampling), in file order, so two runs give byte-identical output except the date;
 * production verdict = artifact/run_cell.parse_option applied to the raw text (imported, not copied) AND the `pred` stored in the cache;
   independent verdict = independent_parse from artifact/audit_scorer.py (its source is loaded by AST, not copied and not executed, because
   that module runs its audit at import time). Checks: A gold id, B independent parse vs production parse (recomputed) and vs the stored
   pred, C official string equality ("option {pred}: {text}" == gold) vs the stored `correct` flag. Any failed check on a row counts that
   row once as a disagreement. Kill criterion as audit_scorer.py: disagreement rate > 0.5%.

Pool modes (--pool; default `registry` when experiments/registry.json exists, else `caches`, and the JSON says which and why):
 * registry: the pool above, selected through the registry (the author's published run; 23,500 rows over 11 caches);
 * caches: every firstpass-teleqna*-<scheme>-*.jsonl cache present under experiments/cache/ (<scheme> = run_cell.CACHE_SCHEME; stale
   pre-scheme caches never match). It needs no registry, so a reproducer who re-ran cells, or a clean copy of
   artifact/ (which holds only the cache that the printed cell command wrote), can run it. The audited pool is then whatever the
   reproducer produced, and the JSON lists each cache's basename, sha256 and row count.

Id modes (the ids of the audited questions in the output):
 * the published environment/scorer_audit_v2.json stores keyed pseudonyms (tq-<16 hex>, HMAC-SHA256 with a private key);
   real ids appear nowhere in it;
 * --id-mode real: the real benchmark ids, for the reproducer's own local use (the file then contains benchmark ids and is flagged
   "publishable": false);
 * --id-mode none (default): no per-row ids; aggregates, per-cache hashes and counts of failed checks by parser category only.
No private material is needed in the last two modes.

Output: experiments/scorer_audit_v2_public.json, or --out. In caches mode the published path is
never written: the default is experiments/scorer_audit_v2_public.json, and an --out that resolves to environment/scorer_audit_v2.json
exits 2. Raw model outputs and anything derived from them are not stored per row: a disagreement record keeps only the cache, the id
(per id mode) and the failed check letters; the parser category of each failed check appears only as a count in the aggregates
(`disagreement_categories` per report and per cache); no raw shape or length, no option id, no predicted id and no gold- or
pred-derived per-row flag. It needs the raw model outputs and the benchmark's real question ids, so it runs after a fresh
re-run (which writes unredacted caches), not on the redacted public caches (their ids are pseudonyms; the script then stops with an error).

Usage: TELEQNA_PATH=<path of the TeleQnA.txt file> python3 audit_scorer_v2.py [--pool registry|caches] [--id-mode real|none] [--out F]
Exit code: 0 PASS (disagreement rate <= 0.5%), 1 FAIL, 2 usage or input error.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never leave __pycache__ under artifact/ (the runner hash ignores it, but keep the tree clean)

import argparse, ast, datetime, hashlib, importlib.util, json, os, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Runs from <root>/environment/ (the author's layout) or from a copy placed in <root>/artifact/; everything is located relative to the project root.
if (HERE / "run_cell.py").is_file() and (HERE / "audit_scorer.py").is_file():
    ART = HERE
    LANE = HERE.parent
else:
    LANE = HERE.parent
    ART = LANE / "artifact"
REPO = LANE.parents[1]
KILL = 0.005


def die(msg: str) -> "None":
    print(f"audit_scorer_v2: {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path, argv=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    if argv is not None:
        sys.argv = argv
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def load_independent_parser():
    """Extract independent_parse from artifact/audit_scorer.py by AST (the module itself runs an audit at import)."""
    src = (ART / "audit_scorer.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "independent_parse":
            ns = {"re": re}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ART / "audit_scorer.py"), "exec"), ns)
            return ns["independent_parse"], ast.get_docstring(node) or "", ast.get_source_segment(src, node)
    die("independent_parse not found in artifact/audit_scorer.py")


def resolve_teleqna() -> Path:
    # same default resolution as artifact/run_cell.py --data
    p = Path(os.environ.get("TELEQNA_PATH", str(REPO / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt")))
    if p.is_dir():
        die(f"TELEQNA_PATH must be the path of the TeleQnA.txt file, as artifact/run_cell.py takes it, not a directory: {p}")
    if not p.is_file():
        die(f"TeleQnA file not found: {p} (set TELEQNA_PATH to the TeleQnA.txt file)")
    return p


def reported_caches(analyze) -> tuple[list[str], str]:
    """Cache basenames of the current-runner ok records, by analyze.py's own selection rule."""
    registry = json.load(open(analyze.REG))
    current = analyze.runner_sha256()
    names = set()
    for r in registry["runs"]:
        if r.get("status") != "ok" or r.get("kind") in ("smoke", "posthoc_control") or r.get("runner_sha256") != current:
            continue
        if r.get("dataset") not in ("teleqna", "teleqna_full"):
            continue
        c = Path(str((r.get("extra") or {}).get("cache") or "")).name
        if c:
            names.add(c)
    return sorted(names), current


def parser_category(problem: str, raw: str) -> str:
    """Coarse, non-identifying category of a failed check; depends on the raw text's form only, never on a gold or predicted option id."""
    if problem == "gold_id":
        return "gold_id_unextractable"
    if problem == "correctness":
        return "official_string_mismatch"
    return "decimal_reply" if re.search(r"\d[^\w\s]\d", raw) else "parser_disagreement"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", choices=("registry", "caches"), default=None,
                    help="registry (default when experiments/registry.json exists): caches referenced by current-runner ok registry records; "
                         "caches (default otherwise): every first-pass TeleQnA cache of the current scheme present under experiments/cache/")
    ap.add_argument("--id-mode", choices=("real", "none"), default=None,
                    help="none (default) = no per-row ids, aggregates only; real = real benchmark ids (local use, not publishable)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="directory of the first-pass caches (default <root>/experiments/cache)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # ---- id mode
    id_mode = a.id_mode or "none"
    ps = None

    # ---- pool mode
    cache_dir = a.cache_dir or (LANE / "experiments" / "cache")
    registry_path = LANE / "experiments" / "registry.json"
    if a.pool is not None:
        pool, pool_selection = a.pool, "explicit --pool"
    elif registry_path.is_file():
        pool, pool_selection = "registry", "default: experiments/registry.json exists"
    else:
        pool, pool_selection = "caches", "auto: experiments/registry.json absent"
    if pool == "registry" and not registry_path.is_file():
        die(f"--pool registry needs {registry_path}; use --pool caches without a registry")

    if pool == "caches" and a.out is not None and a.out.resolve() == (LANE / "environment" / "scorer_audit_v2.json").resolve():
        die("--pool caches must not write the published environment/scorer_audit_v2.json (that file is the registry-pool run); "
            "omit --out (default experiments/scorer_audit_v2_public.json) or give another path")

    run_cell = load_module("run_cell", ART / "run_cell.py", argv=["run_cell.py"])
    independent_parse, ind_doc, ind_src = load_independent_parser()

    data_path = resolve_teleqna()
    data = json.load(open(data_path))

    runner = None
    cache_glob = None
    if pool == "registry":
        analyze = load_module("analyze", ART / "analyze.py", argv=["analyze.py"])
        caches, runner = reported_caches(analyze)
        if not caches:
            die("no TeleQnA cache referenced by current-runner ok registry records")
        missing = [c for c in caches if not (cache_dir / c).is_file()]
        if missing:
            die(f"cache files referenced by the registry are missing: {missing}")
    else:
        cache_glob = f"firstpass-teleqna*-{run_cell.CACHE_SCHEME}-*.jsonl"
        caches = sorted(p.name for p in cache_dir.glob(cache_glob) if p.is_file())
        if not caches:
            die(f"no first-pass TeleQnA cache matching {cache_glob} under {cache_dir} (run a teleqna cell first)")

    total = dict(rows=0, disagreements=0, non_bare=0, ambiguous=0, ind_none=0, prod_none=0, stored_pred_ne_recomputed=0, gold=0, parse=0, correctness=0)
    per_cache, details, distinct = [], [], set()
    categories: dict = {}

    def shown(rid):
        return ps.item(rid) if id_mode == "pseudonym" else str(rid)

    for name in caches:
        path = cache_dir / name
        c = dict(rows=0, disagreements=0, non_bare=0, ambiguous=0)
        cat_c: dict = {}
        ids = []
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["id"] not in data:
                die(f"{name}: row id not in the benchmark file (redacted public cache, or a different TeleQnA file?); the audit needs the unredacted caches of a fresh re-run")
            q = data[r["id"]]
            options = {k: v for k, v in q.items() if k.startswith("option ")}
            gold_str = str(q["answer"]).strip()
            gm = re.match(r"option\s*(\d+)", gold_str.lower())
            gold_id = int(gm.group(1)) if gm else None
            raw = str(r.get("text", ""))
            bare = bool(re.fullmatch(r"\s*\d+\s*", raw))
            ind, amb = independent_parse(raw, len(options))
            prod = run_cell.parse_option(raw, len(options))
            pred = r.get("pred")
            official_correct = pred is not None and f"option {pred}: {options.get(f'option {pred}', '')}".strip() == gold_str
            problems = []
            if gold_id is None or (r.get("gold") is not None and gold_id != r.get("gold")):
                problems.append("gold_id"); total["gold"] += 1
            if ind != prod or ind != pred:
                problems.append("parse"); total["parse"] += 1
            if prod != pred:
                total["stored_pred_ne_recomputed"] += 1
            if official_correct != bool(r.get("correct")):
                problems.append("correctness"); total["correctness"] += 1
            c["rows"] += 1
            c["non_bare"] += not bare
            c["ambiguous"] += bool(amb)
            total["ind_none"] += ind is None
            total["prod_none"] += prod is None
            distinct.add(str(r["id"]))
            if id_mode != "none":
                ids.append(shown(r["id"]))
            if problems:
                c["disagreements"] += 1
                # only non-identifying categories: no option id, no predicted id, no gold- or pred-derived boolean
                letter = {"gold_id": "A", "parse": "B", "correctness": "C"}
                d = {"cache": name}
                if id_mode != "none":
                    d["id"] = shown(r["id"])
                d["failed_checks"] = [letter[x] for x in problems]
                details.append(d)
                # raw-text-derived categories are aggregated only, never kept per row
                for x in problems:
                    cat = parser_category(x, raw)
                    cat_c[cat] = cat_c.get(cat, 0) + 1
                    categories[cat] = categories.get(cat, 0) + 1
        for k in ("rows", "disagreements", "non_bare", "ambiguous"):
            total[k] += c[k]
        entry = {"cache": name, "sha256": sha256_file(path), "rows": c["rows"], "disagreements": c["disagreements"],
                 "non_bare_digit_outputs": c["non_bare"], "ambiguous_multi_number_outputs": c["ambiguous"],
                 "disagreement_categories": dict(sorted(cat_c.items()))}
        if id_mode == "pseudonym":
            entry["audited_ids_pseudonymous"] = ids
        elif id_mode == "real":
            entry["audited_ids_real"] = ids
        per_cache.append(entry)

    n = total["rows"]
    if n == 0:
        die("the audited caches hold no rows")
    rate = total["disagreements"] / n
    id_text = {"pseudonym": "tq-<16 hex> keyed pseudonyms (HMAC-SHA256 with a private key); real ids are not stored",
               "real": "real benchmark ids (--id-mode real, for the reproducer's own local use; this file is not publishable)",
               "none": "no per-row ids are stored (--id-mode none, the default); aggregates, per-cache sha256 and counts of failed checks by parser category only"}[id_mode]
    scope = {"registry": "every cached first-pass row of the TeleQnA caches referenced by current-runner ok registry records (analyze.py selection rule: status ok, kind not smoke/posthoc_control, runner_sha256 == current); no sampling",
             "caches": f"every cached first-pass row of every {cache_glob} cache present under experiments/cache/ (no registry needed); no sampling"}[pool]
    report = {
        "date": datetime.date.today().isoformat(),
        "script": "environment/audit_scorer_v2.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "pool_mode": pool,
        "pool_mode_selection": pool_selection,
        "cache_glob": cache_glob,
        "id_mode": id_mode,
        "publishable": id_mode != "real",
        "supersedes_scope_of": "artifact/audit_scorer.py (pools stale first-execution caches; writes experiments/scorer_audit.json)",
        "scope": scope,
        "current_runner_sha256": runner,
        "teleqna_txt_sha256": sha256_file(data_path),
        "audit_scorer_py_sha256": sha256_file(ART / "audit_scorer.py"),
        "run_cell_py_sha256": sha256_file(ART / "run_cell.py"),
        "n_caches": len(per_cache),
        "audited_rows": n,
        "distinct_questions": len(distinct),
        "disagreements": total["disagreements"],
        "disagreement_rate": round(rate, 6),
        "kill_threshold": KILL,
        "pass": rate <= KILL,
        "non_bare_digit_outputs": total["non_bare"],
        "ambiguous_multi_number_outputs": total["ambiguous"],
        "declined_by_production_parse": total["prod_none"],
        "declined_by_independent_parse": total["ind_none"],
        "stored_pred_differs_from_recomputed_production_parse": total["stored_pred_ne_recomputed"],
        "failed_checks": {"A_gold_id": total["gold"], "B_parse": total["parse"], "C_correctness": total["correctness"]},
        "disagreement_categories": dict(sorted(categories.items())),
        "checks": ["A gold id extraction from the official answer string",
                   "B independent parser vs production parse (run_cell.parse_option, recomputed on the raw text) vs the pred stored in the cache",
                   "C official string equality ('option {pred}: {text}' == gold) vs the stored correct flag"],
        "independent_parser": {"source": "artifact/audit_scorer.py::independent_parse (loaded by AST, not copied)",
                               "docstring": ind_doc,
                               "source_sha256": hashlib.sha256(ind_src.encode()).hexdigest(),
                               "description": "On the lower-cased, stripped raw text: if 'option N' occurs, N is the answer; otherwise the first standalone integer (not adjacent to digits or a dot). The answer is valid only if 1 <= N <= number of options, else None. A text with several distinct standalone integers and no 'option N' is counted as ambiguous."},
        "production_parser": "artifact/run_cell.py::parse_option (imported)",
        "id_pseudonymisation": id_text,
        "caches": per_cache,
        "disagreement_details": details,
    }
    published = HERE / "scorer_audit_v2.json"
    if a.out is not None:
        out = a.out
    elif pool == "registry" and id_mode == "pseudonym" and HERE.name == "environment":
        out = published  # the published file: registry pool + pseudonymised ids + run from environment/
    else:
        out = LANE / "experiments" / "scorer_audit_v2_public.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"scorer audit v2 [{pool} pool, ids {id_mode}]: {n} rows over {len(per_cache)} caches ({len(distinct)} questions), {total['disagreements']} disagreements "
          f"(rate {rate:.4%}, kill threshold {KILL:.1%}, {'PASS' if rate <= KILL else 'FAIL'}), non-bare-digit {total['non_bare']}, ambiguous {total['ambiguous']} -> {out}")
    return 0 if rate <= KILL else 1


if __name__ == "__main__":
    sys.exit(main())
