#!/usr/bin/env python3
"""Check a fresh re-run's analysis against the paper's headline (H1) results, with declared tolerances.

A re-run replicates within tolerance, not digit for digit (the server's reported probabilities differ slightly between a
first and a repeated identical request; see README.md). This script turns "within tolerance" into a pass/fail
check. It reads the results manifest that `analyze.py` writes after a re-run and compares it with the stored headline
reference (environment/headline_expected.json, written from the published analysis).

Decisive rows: H1 on TeleQnA (the pre-registered primary test) for the three pre-registered models. For each row:
  1. the H1 outcome label is the same (free_signal_better / verifier_better);
  2. the pooled AURC difference (verifier minus the re-run's best free signal) has the same sign and its 95% interval excludes zero
     on the same side;
  3. the pooled AURC difference is within ABS_TOL (0.02 AURC) of the reference;
  4. every per-seed difference has the reference's sign (same number of seeds, none missing);
  5. for every seed, the verifier cell and the best-free-signal cell exist and cover the same number of items as in the
     reference (all policies score the same fixed frame, so equal counts mean a complete, paired comparison); a partial
     re-run cannot pass.
Which free signal is best (logprob_gate or margin_gate) is reported, not decisive: for the 7B the two differ by about
0.001 AURC, so a faithful re-run may pick either. Global checks: the re-run's manifest must name the same runner and
analysis-script hashes as the reference (otherwise it is not a re-run of this study: exit 2), and its source registry
must differ from the published one (a manifest built from the published registry is the published analysis, not a
re-run: exit 3, so skipping the documented step of moving the shipped experiments/ aside cannot report PASS). The
published registries are the author's registry and the redacted copy shipped in this release; the reference records
both hashes.
A manifest that lacks primary_h1 or a decisive row, whose decisive row lacks a required field, or that lacks a decisive
row's per-seed verifier or best-free cell in per_cell, is incomplete input (exit 2), not a failed headline.
NetConfEval rows and the single-seed sensitivity rows (further model, full benchmark) are reported, not decisive: the
paper states that no NetConfEval verdict is robust.

Usage (from artifact/, after a re-run has regenerated ../analysis/results_manifest.json):
  python3 ../environment/verify_headline_reproduction.py [--manifest ../analysis/results_manifest.json]
                                                        [--expected ../environment/headline_expected.json] [--out report.json]
Exit 0: every decisive check passes. Exit 1: a decisive check fails. Exit 2: unreadable or incomplete input, or a
different runner or analysis script. Exit 3: the manifest is the published analysis, not a re-run."""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABS_TOL = 0.02  # AURC; the smallest reported TeleQnA H1 gap is about 0.048, so a re-run inside this band keeps every verdict
DECISIVE_DATASET = "teleqna"
PREREGISTERED_MODELS = ("qwen2.5-coder-7b-instruct", "qwen3-coder-30b-a3b-instruct", "llama-3.3-70b-instruct")


VERIFIER = "verifier_gated"


def h1_rows(manifest: dict) -> dict:
    """(dataset, model) -> the H1 summary: outcome, best free signal, delta vs the best free signal, interval, per-seed
    deltas, and the per-seed item counts of the verifier cell and the best-free-signal cell (from per_cell)."""
    cell_n = {(c.get("dataset"), c.get("model"), c.get("seed"), c.get("method")): c.get("n") for c in manifest.get("per_cell", [])}
    out = {}
    for row in manifest.get("primary_h1", []):
        best = row.get("best_free_signal")
        cmp = next((c for c in row.get("comparisons", []) if c.get("baseline") == best), None)
        if cmp is None:
            continue
        ds, m = row["dataset"], row["model"]
        seeds = row.get("seeds_pooled") or []
        out[(ds, m)] = {
            "outcome": row.get("outcome"), "best_free_signal": best, "delta": cmp.get("aurc_delta"),
            "ci": cmp.get("ci"), "seeds_agree_in_sign": cmp.get("seeds_agree_in_sign"),
            "per_seed_delta": cmp.get("per_seed_delta"),
            "cell_n": {str(s): [cell_n.get((ds, m, s, VERIFIER)), cell_n.get((ds, m, s, best))] for s in seeds}}
    return out


def side(ci) -> str:
    lo, hi = ci
    return "below_zero" if hi < 0 else "above_zero" if lo > 0 else "includes_zero"


def check(expected: dict, observed: dict) -> list[dict]:
    results = []
    for key, ref in sorted(expected.items()):
        ds, model = key
        decisive = ds == DECISIVE_DATASET and model in PREREGISTERED_MODELS
        obs = observed.get(key)
        rec = {"dataset": ds, "model": model, "decisive": decisive, "checks": {}}
        if obs is None:
            rec["checks"]["present"] = False
        else:
            rec["checks"] = {
                "present": True,
                "outcome_same": obs["outcome"] == ref["outcome"],
                "sign_and_interval_side_same": (obs["delta"] > 0) == (ref["delta"] > 0) and side(obs["ci"]) == side(ref["ci"]),
                "delta_within_tolerance": abs(obs["delta"] - ref["delta"]) <= ABS_TOL,
                "per_seed_signs_match_reference": len(obs["per_seed_delta"]) == len(ref["per_seed_delta"])
                    and all(x is not None and (x > 0) == (ref["delta"] > 0) for x in obs["per_seed_delta"]),
                "same_cells_and_item_counts": obs["cell_n"] == ref["cell_n"] and (not decisive or all(None not in v for v in obs["cell_n"].values())),
            }
            rec["notes"] = {"best_free_signal_same": obs["best_free_signal"] == ref["best_free_signal"]}
            rec["reference"] = {k: ref.get(k) for k in ("outcome", "best_free_signal", "delta", "ci", "per_seed_delta", "cell_n")}
            rec["observed"] = {k: obs.get(k) for k in ("outcome", "best_free_signal", "delta", "ci", "per_seed_delta", "cell_n")}
        rec["pass"] = all(rec["checks"].values())
        results.append(rec)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(HERE.parent / "analysis" / "results_manifest.json"))
    ap.add_argument("--expected", default=str(HERE / "headline_expected.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        manifest = json.load(open(a.manifest))
    except (OSError, ValueError) as e:
        print(f"verify_headline_reproduction: cannot read manifest {a.manifest}: {e}", file=sys.stderr); return 2
    if not isinstance(manifest, dict) or not isinstance(manifest.get("primary_h1"), list):
        print(f"verify_headline_reproduction: {a.manifest} has no primary_h1 list (not an analyze.py results manifest)", file=sys.stderr); return 2
    observed = h1_rows(manifest)
    try:
        exp = json.load(open(a.expected))
        expected = {(r["dataset"], r["model"]): r for r in exp["rows"]}
    except (OSError, ValueError, KeyError) as e:
        print(f"verify_headline_reproduction: cannot read reference {a.expected}: {e}", file=sys.stderr); return 2
    published = set(exp.get("published_registry_sha256s") or [])
    if not exp.get("source_manifest_registry_sha256") or not manifest.get("source_registry_sha256") or len(published) < 2:
        print("verify_headline_reproduction: registry hash missing from the manifest or the reference", file=sys.stderr); return 2
    if manifest.get("source_registry_sha256") in published | {exp.get("source_manifest_registry_sha256")}:
        print("verify_headline_reproduction: this manifest was built from the published registry: it is the published"
              " analysis, not a re-run (move the shipped experiments/ aside before re-running; see README.md)", file=sys.stderr); return 3
    for field, key in (("runner_sha256", "source_manifest_runner_sha256"), ("analysis_script_sha256", "source_manifest_analysis_script_sha256")):
        if not exp.get(key) or manifest.get(field) != exp.get(key):
            print(f"verify_headline_reproduction: the manifest's {field} ({str(manifest.get(field))[:12]}) differs from the reference"
                  f" ({str(exp.get(key))[:12]}): not a re-run of this study", file=sys.stderr); return 2
    decisive_keys = [(DECISIVE_DATASET, m) for m in PREREGISTERED_MODELS]
    if any(k not in expected for k in decisive_keys):
        print("verify_headline_reproduction: reference lacks a decisive TeleQnA row", file=sys.stderr); return 2
    REQUIRED = ("outcome", "best_free_signal", "delta", "ci", "per_seed_delta", "cell_n")
    for k in decisive_keys:
        row = observed.get(k)
        if row is None or any(row.get(f) is None for f in REQUIRED) or not isinstance(row["ci"], list) or len(row["ci"]) != 2 \
                or not isinstance(row["per_seed_delta"], list) or not row["cell_n"] \
                or any(None in v for v in row["cell_n"].values()):
            print(f"verify_headline_reproduction: incomplete input: decisive row {k} missing, lacking a required field, or lacking"
                  " a per-seed verifier/best-free cell in per_cell", file=sys.stderr); return 2
    results = check(expected, observed)
    decisive = [r for r in results if r["decisive"]]
    ok = all(r["pass"] for r in decisive)
    report = {"decision": "PASS" if ok else "FAIL", "abs_tol_aurc": ABS_TOL, "decisive_rows": len(decisive), "rows": results}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    for r in results:
        tag = "decisive" if r["decisive"] else "reported"
        failed = [k for k, v in r["checks"].items() if not v]
        note = "" if r.get("notes", {}).get("best_free_signal_same", True) else " (note: best free signal differs; reported, not decisive)"
        print(f"{'PASS' if r['pass'] else 'FAIL'} [{tag}] {r['dataset']} {r['model']}" + (f" failed: {', '.join(failed)}" if failed else "") + note)
    print(f"headline reproduction: {report['decision']} ({sum(r['pass'] for r in decisive)}/{len(decisive)} decisive rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
