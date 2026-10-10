#!/usr/bin/env python3
"""Analysis: experiments/registry.json + per-run predictions.jsonl -> analysis/results_manifest.json,
analysis/generated_numbers.tex, analysis/tables/*.tex, analysis/figures/*.pdf. No number is typed by hand anywhere else.

Statistics follow the frozen protocol:
- AURC and selective accuracy at matched coverage are TIE-AWARE: items with equal scores are ranked in expectation
  (uniform random tie order), so a policy with a constant score (unguarded) or coarse scores (self-consistency k=1,
  verbalized confidence) is not rewarded or punished by an arbitrary sort order.
- Primary H1 rule, pre-registered per model: pool the items of all seeds (item = seed:id), compare verifier_gated to the
  best free signal (the one of {logprob_gate, margin_gate} with the lower pooled AURC) by paired bootstrap over items
  (10,000 resamples), two-sided; Holm across the six policy-vs-verifier comparisons within each (dataset, model).
- Per (dataset, model, seed) comparisons with paired bootstrap on AURC and Sel@0.8 differences and McNemar at 0.8.
- H2 (exploratory): Kendall tau between the per-policy AURC vectors on TeleQnA and NetConfEval per (model, seed),
  excluding the constant-score unguarded policy, with a bootstrap CI over items (resampled within each dataset).
- Wrong-answer control (verifier AUROC right vs deliberately wrong), combined-signal control (logistic fit with item-level
  5-fold cross-validation, folds by question id), ECE/Brier on ALL items, per-category tables, format-failure and
  cost tables (tokens and seconds per item, cost-matched k=1 control on the same row as the verifier).
Post hoc analyses (flagged post_hoc in the manifest; additional outputs that never alter the numbers above):
  decline_sensitivity, tie_sensitivity, extra_latency, tau_range, first_pass_nonargmax_share, sample_diversity,
  nce_lone_lb_sensitivity, nce_official_accuracy_sensitivity, decline_sensitivity_h4, h1_sel80, compute_hours (registry
  wall clock), option_aware_control (registry records of kind posthoc_control written by environment/control_option_aware.py;
  such records never enter the pre-registered cells), nce_iteration_cluster, combined_vs_best, and option_aware_sensitivity
  (option-aware control at 1e-6 tie resolution, matched-coverage Sel@0.8 differences, extra tokens and seconds per item).
Additional pre-registered reporting: nce_batch_size (NetConfEval per batch size), mcnemar_h1 (table only), aurc_by_policy.
Tables come in print-size form (main/h3/h1 compact, floor) with full versions kept as main_full/h3_full. Figures use one
fixed colour and line style per policy, keyed by the pre-registered policy list, in every panel and legend (risk-coverage,
cost, reliability), and a fixed colour per bar model in h3_gain (H3 gains over the unguarded floor).
Usage: python3 analyze.py [bootstrap_resamples]
"""
from __future__ import annotations

import hashlib, json, math, os, random, re, sys
from collections import defaultdict
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
ART = Path(__file__).resolve().parent
REG = LANE / "experiments" / "registry.json"
OUT = LANE / "analysis"
METHOD = "verifier_gated"
FREE = ("logprob_gate", "margin_gate")
COVERAGES = (0.9, 0.8, 0.7)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
B_TAU = min(B, 1000)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_preds(run: dict) -> list[dict]:
    for o in run.get("outputs", []):
        if o.endswith("predictions.jsonl"):
            return [json.loads(l) for l in (LANE / o).read_text().splitlines()]
    return []


def load_result(run: dict) -> dict:
    for o in run.get("outputs", []):
        if o.endswith("result.json"):
            return json.load(open(LANE / o))
    return {}


# ---------------------------------------------------------------- tie-aware selective metrics

def _groups(rows: list[dict]):
    """Yield (group_size, wrong_in_group) for score groups in descending score order. Ties are EXACT float equality, the
    pre-registered rule; near-ties (scores within 1e-6 of 0 or 1) are disclosed through saturation_rate, not merged."""
    ranked = sorted(rows, key=lambda r: -float(r["score"]))
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j]["score"] == ranked[i]["score"]:
            j += 1
        yield (j - i, sum(0 if r["correct"] else 1 for r in ranked[i:j]))
        i = j


def aurc(rows: list[dict]) -> float:
    """Tie-aware area under the risk-coverage curve: expected risk at each coverage under random tie order."""
    n = len(rows)
    if n == 0:
        return float("nan")
    acc = 0.0; W = 0; N = 0
    for g, w in _groups(rows):
        for j in range(1, g + 1):
            acc += (W + w * j / g) / (N + j)
        W += w; N += g
    return acc / n


def sel_at(rows: list[dict], cov: float) -> float:
    """Tie-aware selective accuracy at coverage: expected accuracy among the top-k scored items."""
    n = len(rows)
    k = max(1, int(round(cov * n)))
    C = 0.0; N = 0
    for g, w in _groups(rows):
        take = min(g, k - N)
        C += (g - w) * take / g
        N += take
        if N >= k:
            break
    return C / k


def risk_coverage_curve(rows: list[dict], points: int = 50) -> list[tuple[float, float]]:
    n = len(rows); out = []
    cum = []; W = 0; N = 0
    for g, w in _groups(rows):
        for j in range(1, g + 1):
            cum.append((W + w * j / g) / (N + j))
        W += w; N += g
    for t in range(1, points + 1):
        k = max(1, int(round(t / points * n)))
        out.append((k / n, cum[k - 1]))
    return out


def ece_brier(rows: list[dict], bins: int = 15) -> tuple[float, float]:
    brier = sum((r["score"] - (1.0 if r["correct"] else 0.0)) ** 2 for r in rows) / len(rows)
    buckets = defaultdict(list)
    for r in rows:
        buckets[min(bins - 1, int(max(0.0, min(1.0, r["score"])) * bins))].append(r)
    ece = sum(len(b) / len(rows) * abs(sum(x["score"] for x in b) / len(b) - sum(x["correct"] for x in b) / len(b)) for b in buckets.values())
    return ece, brier


# ---------------------------------------------------------------- inference

def paired_bootstrap(a: list[dict], b: list[dict], stat, B: int, seed: int = 0) -> tuple[float, float, float, float]:
    """stat(rows)->float; returns (delta, lo, hi, p_two_sided) for stat(a)-stat(b) over item resamples (paired on id)."""
    ida = {r["id"]: r for r in a}; idb = {r["id"]: r for r in b}
    ids = sorted(set(ida) & set(idb))
    rng = random.Random(seed)
    obs = stat([ida[i] for i in ids]) - stat([idb[i] for i in ids])
    deltas = []
    for _ in range(B):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        deltas.append(stat([ida[i] for i in s]) - stat([idb[i] for i in s]))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B) - 1]
    p = 2 * min(sum(d <= 0 for d in deltas), sum(d >= 0 for d in deltas)) / B
    return obs, lo, hi, min(1.0, p)


def clustered_bootstrap(a: list[dict], b: list[dict], stat, B: int, seed: int = 0) -> tuple[float, float, float, float]:
    """Pooled-seed test that respects repeated observations of the same item: rows carry id 'seed:item'; the resampling
    unit is the ITEM (cluster) and every seed row of a drawn item is taken together. Returns (delta, lo, hi, p)."""
    ida = {r["id"]: r for r in a}; idb = {r["id"]: r for r in b}
    ids = sorted(set(ida) & set(idb))
    clusters = defaultdict(list)
    for i in ids:
        clusters[i.split(":", 1)[1]].append(i)
    keys = sorted(clusters)
    rng = random.Random(seed)
    obs = stat([ida[i] for i in ids]) - stat([idb[i] for i in ids])
    deltas = []
    for _ in range(B):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        sel = [i for k in drawn for i in clusters[k]]
        deltas.append(stat([ida[i] for i in sel]) - stat([idb[i] for i in sel]))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B) - 1]
    p = 2 * min(sum(d <= 0 for d in deltas), sum(d >= 0 for d in deltas)) / B
    return obs, lo, hi, min(1.0, p)


def mcnemar_at(a: list[dict], b: list[dict], cov: float) -> tuple[int, int, float]:
    """Discordant pairs at matched coverage (deterministic tie-break by id for the answered set): answered-correct under a
    but wrong/abstained under b, and vice versa; exact binomial two-sided p."""
    def answered_correct(rows):
        ranked = sorted(rows, key=lambda r: (-r["score"], r["id"]))
        k = max(1, int(round(cov * len(rows))))
        return {r["id"]: r["correct"] for r in ranked[:k]}
    ca, cb = answered_correct(a), answered_correct(b)
    n01 = sum(1 for i in ca if ca[i] and not cb.get(i, False))
    n10 = sum(1 for i in cb if cb[i] and not ca.get(i, False))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    p = sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) * 2 / (2 ** n)
    return n01, n10, min(1.0, p)


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals); m = len(pvals); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def auroc(pos: list[float], neg: list[float]) -> float:
    """AUROC of a score separating positives (should be high) from negatives; ties count half (rank-based, O(n log n))."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks with ties
    ranks = [0.0] * len(allv); i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = r
        i = j
    rpos = sum(r for r, (v, y) in zip(ranks, allv) if y == 1)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def logit_feat(p: float) -> float:
    """Scores live near 1.0 for most items; the log-odds transform spreads them so a linear combination can use them."""
    p = min(1 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1 - p))


def standardise(X: list[list[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    d = len(X[0]); mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = [max(1e-9, (sum((r[j] - mu[j]) ** 2 for r in X) / len(X)) ** 0.5) for j in range(d)]
    return [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in X], mu, sd


def logistic_fit(X: list[list[float]], y: list[int], iters: int = 400, lr: float = 0.5) -> list[float]:
    """Tiny L2-regularised logistic regression (no dependencies) for the combined-signal control."""
    d = len(X[0]); w = [0.0] * (d + 1)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        for xi, yi in zip(X, y):
            z = w[0] + sum(wj * xj for wj, xj in zip(w[1:], xi))
            pz = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = pz - yi
            grad[0] += e
            for j in range(d):
                grad[j + 1] += e * xi[j]
        n = len(X)
        for j in range(d + 1):
            w[j] -= lr * (grad[j] / n + (0.01 * w[j] if j else 0.0))
    return w


def kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x); conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            conc += s > 0; disc += s < 0
    return (conc - disc) / max(1, (conc + disc))


def pooled(cells: dict, ds: str, m: str, seeds: list[int], pol: str) -> list[dict]:
    """Pool items over seeds with item id = seed:id (paired across policies because every policy scores the same first pass)."""
    out = []
    for s in seeds:
        for r in cells.get((ds, m, s, pol), []):
            out.append({"id": f"{s}:{r['id']}", "score": r["score"], "correct": r["correct"], "category": r.get("category")})
    return out


# ---------------------------------------------------------------- post hoc reporting helpers (not pre-registered)
# These only build modified COPIES of rows and then call the unchanged pre-registered functions (aurc, clustered_bootstrap).

def rows_at_resolution(rows: list[dict], ndigits: int = 6) -> list[dict]:
    """POST-HOC tie-resolution sensitivity: copy rows with scores rounded to `ndigits` decimals, so scores that differ by
    less than 10^-ndigits become exact ties and are ranked in expectation by the unchanged `_groups`/`aurc`."""
    return [dict(r, score=round(float(r["score"]), ndigits)) for r in rows]


def rows_declines_last(rows: list[dict], declined_ids: set) -> list[dict]:
    """POST-HOC decline sensitivity: copy rows and give every declined row (id in declined_ids) the same score, strictly
    below every other score of the policy, so declines are abstained on first and are tied among themselves."""
    if not rows:
        return []
    bottom = min(float(r["score"]) for r in rows) - 1.0
    return [dict(r, score=bottom) if r["id"] in declined_ids else dict(r) for r in rows]


# ---- NetConfEval item metadata (read-only helpers, never used by the pre-registered outputs above)
NCE_BATCH_WORD = {1: "One", 2: "Two", 5: "Five", 10: "Ten"}


def nce_batch_size(item_id: str) -> int | None:
    """NetConfEval item ids are 'nce-it{iteration}-b{batch_size}-c{chunk}' (run_cell.load_netconfeval_t1)."""
    m = re.fullmatch(r"nce-it\d+-b(\d+)-c\d+", str(item_id))
    return int(m.group(1)) if m else None


def nce_frame(policy_types: list[str], batch_sizes=(1, 2, 5, 10), iterations: int = 25) -> dict:
    """Offline reconstruction of the fixed NetConfEval frame with the OFFICIAL sampler, same calls in the same order as
    run_cell.load_netconfeval_t1: item id -> {'types': policy type of each requirement, 'expected': official expected
    specification}. The phrasing step is not needed (pick_sample re-seeds per iteration). The global RNG state is restored."""
    nce = Path(os.environ.get("NETCONFEVAL_PATH", str(LANE.parents[1] / "runtime" / "ground_truth" / "NetConfEval")))
    if str(nce) not in sys.path:
        sys.path.insert(0, str(nce))
    from sortedcontainers import SortedSet
    from netconfeval.common.utils import load_csv, pick_sample, transform_sample_to_expected, chunk_list
    pts = SortedSet(policy_types)
    state = random.getstate()
    try:
        dataset = load_csv(str(nce / "assets" / "step_1_policies.csv"), pts)
        out = {}
        for it in range(iterations):
            samples = pick_sample(max(batch_sizes), dataset, it, pts)
            for b in batch_sizes:
                for ci, chunk in enumerate(chunk_list(samples, b)):
                    out[f"nce-it{it}-b{b}-c{ci}"] = {"types": [x["type"] if x else None for x in chunk], "expected": transform_sample_to_expected(chunk)}
    finally:
        random.setstate(state)
    return out


def nce_official_row(expected: dict, text: str) -> dict | None:
    """Re-score a cached first-pass text with the official compare_result exactly as run_cell.netconfeval_score parses it
    (None when the text is a decline, unparsable or carries no policy key); used only to VALIDATE the frame reconstruction."""
    from netconfeval.common.utils import compare_result
    import logging
    raw = (text or "").strip(); start, end = raw.find("{"), raw.rfind("}")
    try:
        out = json.loads(raw[start:end + 1]) if start != -1 else {}
    except json.JSONDecodeError:
        return None
    if not isinstance(out, dict) or str(out.get("status", "")).lower() == "error":
        return None
    result = out.get("result", out)
    if not isinstance(result, dict) or not any(k in result for k in expected.keys()):
        return None
    row = {"total": 0, "success": 0, "fail": 0, "wrong": 0, "accuracy": 0}
    logging.disable(logging.CRITICAL)
    try:
        compare_result(expected, result, row)
    except Exception:
        return None
    finally:
        logging.disable(logging.NOTSET)
    return row


# ---- post hoc helpers (read-only, never used by the pre-registered outputs above)
def clustered_bootstrap_by(a: list[dict], b: list[dict], stat, B: int, seed: int, cluster_of) -> tuple[float, float, float, float, int]:
    """clustered_bootstrap with a caller-defined cluster: cluster_of(row id) -> cluster key; every row of a drawn cluster is
    taken together (all seeds, all items of the cluster). Same percentile CI and two-sided p as clustered_bootstrap.
    Returns (delta, lo, hi, p, number of clusters); the observed delta is computed on the same paired rows as clustered_bootstrap."""
    ida = {r["id"]: r for r in a}; idb = {r["id"]: r for r in b}
    ids = sorted(set(ida) & set(idb))
    clusters = defaultdict(list)
    for i in ids:
        clusters[cluster_of(i)].append(i)
    keys = sorted(clusters)
    rng = random.Random(seed)
    obs = stat([ida[i] for i in ids]) - stat([idb[i] for i in ids])
    deltas = []
    for _ in range(B):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        sel = [i for k in drawn for i in clusters[k]]
        deltas.append(stat([ida[i] for i in sel]) - stat([idb[i] for i in sel]))
    deltas.sort()
    lo, hi = deltas[int(0.025 * B)], deltas[int(0.975 * B) - 1]
    p = 2 * min(sum(d <= 0 for d in deltas), sum(d >= 0 for d in deltas)) / B
    return obs, lo, hi, min(1.0, p), len(keys)


def fmt_boot_p(p: float, B: int) -> str:
    """Bootstrap p for the post hoc macros: a p of exactly 0 (no resample on the other side of zero) prints as '$<$1/B'."""
    return f"$<${1 / B:.2g}" if p == 0 else f"{p:.3g}"


def nce_iteration(pooled_id: str) -> str:
    """Sampler iteration of a pooled NetConfEval row id 'seed:nce-it{i}-b{size}-c{chunk}' (all batch sizes and seeds of one
    iteration share the sampled requirements)."""
    m = re.fullmatch(r"nce-it(\d+)-b\d+-c\d+", pooled_id.split(":", 1)[-1])
    return f"it{int(m.group(1))}" if m else pooled_id.split(":", 1)[-1]


POLICY_LABEL = {"unguarded": "unguarded", "logprob_gate": "max-logprob", "margin_gate": "margin", "self_consistency": "self-consistency k=5",
                "self_consistency_k1": "self-consistency k=1", "confidence_gate": "verbalised confidence", "verifier_gated": "verifier gate (P(True))",
                "verifier_gated_pB": "verifier, neutral wording", "verifier_gated_fs": "verifier, few-shot", "verifier_wrong_control": "verifier on wrong option",
                "verifier_gated_x": "verifier, independent model"}


MODEL_LABEL = {"qwen2.5-coder-7b-instruct": "Qwen2.5-Coder-7B", "qwen3-coder-30b-a3b-instruct": "Qwen3-Coder-30B-A3B", "llama-3.3-70b-instruct": "Llama-3.3-70B",
               "gemma-4-31b-it-qat": "Gemma-4-31B (dense)"}
DATASET_LABEL = {"teleqna": "TeleQnA", "netconfeval_t1": "NetConfEval", "teleqna_full": "TeleQnA (all)"}
# short labels for the print-size (compact) tables and figures; the full tables keep the long labels above
SHORT_MODEL = {"qwen2.5-coder-7b-instruct": "7B", "qwen3-coder-30b-a3b-instruct": "30B", "llama-3.3-70b-instruct": "70B", "gemma-4-31b-it-qat": "Gemma-31B"}
SHORT_DATASET = {"teleqna": "TeleQnA", "netconfeval_t1": "NetConfEval", "teleqna_full": "TeleQnA (all)"}
SHORT_POLICY = {"unguarded": "unguarded", "logprob_gate": "max-logprob", "margin_gate": "margin", "self_consistency": "SC k=5", "self_consistency_k1": "SC k=1",
                "confidence_gate": "verbalised", "verifier_gated": "verifier", "error_abstain_floor": "error-abstain floor"}


def mlabel(m: str) -> str:
    return texsafe(MODEL_LABEL.get(m, m))


def dlabel(d: str) -> str:
    return texsafe(DATASET_LABEL.get(d, d))


def plabel(p: str) -> str:
    return texsafe(POLICY_LABEL.get(p, p))


def smlabel(m: str) -> str:
    return texsafe(SHORT_MODEL.get(m, MODEL_LABEL.get(m, m)))


def sdlabel(d: str) -> str:
    return texsafe(SHORT_DATASET.get(d, d))


def splabel(p: str) -> str:
    return texsafe(SHORT_POLICY.get(p, POLICY_LABEL.get(p, p)))


def texsafe(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%")


SEED_WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}


def skey(seed) -> str:
    """LaTeX control sequences may contain letters only: seed 1 -> 'SOne'."""
    return "S" + SEED_WORD.get(int(seed), "N" + "".join(SEED_WORD.get(int(c), "") for c in str(seed)))


def mkey(*parts: str) -> str:
    return "".join(ch for ch in "".join(parts) if ch.isalpha())


# ---------------------------------------------------------------- main

ANALYSIS_ONLY = {"analyze.py", "README.md", "requirements.txt"}


def runner_sha256() -> str:
    """Runner hash recorded with every run: SHA-256 over every artifact file except the analysis-only files."""
    h = hashlib.sha256()
    for p in sorted(x for x in ART.rglob("*") if x.is_file() and "__pycache__" not in x.parts and not x.name.endswith(".pyc") and x.name not in ANALYSIS_ONLY):
        h.update(str(p.relative_to(ART)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def load_protocol() -> dict:
    import yaml
    return yaml.safe_load((LANE / "protocol" / "protocol.yaml").read_text()) or {}


def main() -> None:
    protocol = load_protocol()
    # binding: the pre-registered analysis script is the one executed (protocol.analysis_script_sha256 set at freeze)
    bound = str(protocol.get("analysis_script_sha256") or "")
    if bound and bound != sha(Path(__file__)) and os.environ.get("PAPER_ALLOW_UNBOUND_ANALYSIS") != "1":
        raise SystemExit(f"analyze.py sha256 {sha(Path(__file__))[:12]} differs from the frozen protocol's analysis_script_sha256 {bound[:12]}; set PAPER_ALLOW_UNBOUND_ANALYSIS=1 to run it anyway")
    registry = json.load(open(REG))
    current = runner_sha256()
    runs = [r for r in registry["runs"] if r.get("status") == "ok" and r.get("kind") != "smoke" and r.get("runner_sha256") == current]
    stale = [r["run_id"] for r in registry["runs"] if r.get("status") == "ok" and r.get("kind") != "smoke" and r.get("runner_sha256") != current]
    # post hoc control records (kind posthoc_control, environment/control_option_aware.py) never enter the
    # pre-registered cells; only the option_aware_control block reads them (from the registry)
    runs = [r for r in runs if r.get("kind") != "posthoc_control"]
    bar_models = [str(m.get("id") if isinstance(m, dict) else m) for m in protocol.get("models_or_configs") or []]
    primary_datasets = [str(d.get("id")) for d in protocol.get("datasets") or [] if str(d.get("role") or "primary") != "sensitivity"]
    evaluator_of = {}
    for m in protocol.get("metrics") or []:
        for ds, ev in (m.get("evaluator_by_dataset") or {}).items():
            evaluator_of[ds] = ev
    format_limit = 0.5
    protocol_seeds = {int(x) for x in (protocol.get("seeds") or [])}
    cells, results = {}, {}
    for r in runs:
        if protocol_seeds and int(r.get("seed", -1)) not in protocol_seeds:
            continue  # records of a replaced seed stay in the registry as history but never enter the analysis
        rows = load_preds(r)
        if rows:
            key = (r["dataset"], r["model"], int(r["seed"]), r["method"])
            cells[key] = rows
            results[key] = load_result(r)
    datasets = sorted({k[0] for k in cells}); models = sorted({k[1] for k in cells}); seeds = sorted({k[2] for k in cells}); methods = sorted({k[3] for k in cells})
    # pre-registered policy set (protocol.yaml matrix.policies_pre_registered); sensitivity/control cells are reported separately
    try:
        import yaml
        matrix = (yaml.safe_load((LANE / "protocol" / "protocol.yaml").read_text()) or {}).get("matrix") or {}
        prereg = [p for p in (matrix.get("policies_pre_registered") or []) if p in methods]
    except Exception:
        prereg = []
    policies = prereg or [p for p in methods if p not in ("verifier_wrong_control", "verifier_gated_pB", "verifier_gated_fs")]
    sensitivity = [p for p in methods if p not in policies and p != "verifier_wrong_control"]
    comparisons, per_cell = [], []
    numbers = {}
    # ---- per-cell metrics and per-seed comparisons vs the verifier
    for ds in datasets:
        for m in models:
            for s in seeds:
                ref = cells.get((ds, m, s, METHOD))
                for pol in methods:
                    rows = cells.get((ds, m, s, pol))
                    if not rows:
                        continue
                    e, b = ece_brier(rows)
                    mv = (results.get((ds, m, s, pol)) or {}).get("metric_values") or {}
                    # saturation: scores within 1e-6 of 0 or 1 (exact ties and near-ties alike are ranked by noise; both are disclosed)
                    sat = sum(1 for r in rows if r["score"] <= 1e-6 or r["score"] >= 1 - 1e-6) / len(rows)
                    own = auroc([r["score"] for r in rows if r["correct"]], [r["score"] for r in rows if not r["correct"]])
                    per_cell.append({"dataset": ds, "model": m, "seed": s, "method": pol, "n": len(rows), "aurc": round(aurc(rows), 4),
                                     "sel90": round(sel_at(rows, 0.9), 4), "sel80": round(sel_at(rows, 0.8), 4), "sel70": round(sel_at(rows, 0.7), 4),
                                     "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4), "coverage_at_threshold": round(sum(1 for r in rows if r.get("answered")) / len(rows), 4),
                                     "ece": round(e, 4), "brier": round(b, 4), "saturation_rate": round(sat, 4), "own_answer_auroc": (round(own, 4) if own == own else None), "format_failure_rate": mv.get("format_failure_rate"), "declined_rate": mv.get("declined_rate"),
                                     "tokens_per_item": mv.get("tokens_per_item"), "extra_tokens_per_item": mv.get("extra_tokens_per_item"), "latency_s_per_item": mv.get("latency_s_per_item")})
                    if ref and pol not in (METHOD, "verifier_wrong_control"):
                        d, lo, hi, p = paired_bootstrap(ref, rows, aurc, B, seed=s)
                        d2, lo2, hi2, p2 = paired_bootstrap(ref, rows, lambda rr: sel_at(rr, 0.8), B, seed=s)
                        n01, n10, pm = mcnemar_at(ref, rows, 0.8)
                        comparisons.append({"id": f"cmp-{ds}-{m}-s{s}-{pol}", "method": METHOD, "baseline": pol, "dataset": ds, "model": m, "seed": s, "metric_id": "aurc",
                                            "evaluator_id": evaluator_of.get(ds, "e_teleqna"), "n": len(rows), "test": "paired bootstrap (tie-aware AURC difference) + McNemar at coverage 0.8",
                                            "statistic": round(d, 5), "p_value": round(p, 5), "effect_size": round(d, 5),
                                            "effect_measure": "AURC difference (verifier minus baseline; negative favours verifier)", "ci_low": round(lo, 5), "ci_high": round(hi, 5),
                                            "sel80_delta": round(d2, 4), "sel80_ci": [round(lo2, 4), round(hi2, 4)], "sel80_p": round(p2, 5), "mcnemar": {"n01": n01, "n10": n10, "p": round(pm, 5)}})
    groups = defaultdict(list)
    for i, c in enumerate(comparisons):
        c["family"] = "pre_registered" if c["baseline"] in policies else "sensitivity"
        groups[(c["dataset"], c["model"], c["seed"], c["family"])].append(i)
    for key, idxs in groups.items():
        if key[3] != "pre_registered":  # sensitivity cells are reported uncorrected and marked as such (protocol statistics.assumptions.holm)
            for i in idxs:
                comparisons[i]["corrected_p"] = None; comparisons[i]["correction"] = "none (sensitivity cell, uncorrected)"
            continue
        adj = holm([comparisons[i]["p_value"] for i in idxs])
        for i, a in zip(idxs, adj):
            comparisons[i]["corrected_p"] = round(a, 5); comparisons[i]["correction"] = "Holm"
    # ---- primary H1 rule per (dataset, model): pooled items across seeds, verifier vs best free signal
    h1 = []
    for ds in datasets:
        for m in models:
            ref = pooled(cells, ds, m, seeds, METHOD)
            if not ref:
                continue
            free = {f: pooled(cells, ds, m, seeds, f) for f in FREE if pooled(cells, ds, m, seeds, f)}
            if not free:
                continue
            best = min(free, key=lambda f: aurc(free[f]))
            pooled_cmps = []
            for pol in policies:
                rows = pooled(cells, ds, m, seeds, pol)
                if not rows or pol == METHOD:
                    continue
                d, lo, hi, p = clustered_bootstrap(ref, rows, aurc, B, seed=101)
                # seed-level robustness: sign of the per-seed AURC difference for each seed (primary claim requires all seeds to agree in sign)
                per_seed = []
                for sd in seeds:
                    ra = cells.get((ds, m, sd, METHOD)); rb = cells.get((ds, m, sd, pol))
                    if ra and rb:
                        per_seed.append(round(aurc(ra) - aurc(rb), 5))
                pooled_cmps.append({"baseline": pol, "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5), "test": "clustered bootstrap over items (all seeds of a drawn item together)",
                                    "per_seed_delta": per_seed, "seeds_agree_in_sign": bool(per_seed) and (all(x > 0 for x in per_seed) or all(x < 0 for x in per_seed))})
            adj = holm([c["p_value"] for c in pooled_cmps])
            for c, a in zip(pooled_cmps, adj):
                c["corrected_p"] = round(a, 5)
            bc = next(c for c in pooled_cmps if c["baseline"] == best)
            sig = bc["corrected_p"] < 0.05 and bc["seeds_agree_in_sign"]
            outcome = "verifier_better" if (sig and bc["aurc_delta"] < 0) else "free_signal_better" if (sig and bc["aurc_delta"] > 0) else "no_significant_difference"
            h1.append({"dataset": ds, "model": m, "seeds_pooled": seeds, "n_items": len(ref), "best_free_signal": best, "aurc_verifier": round(aurc(ref), 4), "aurc_best_free": round(aurc(free[best]), 4),
                       "sel80_verifier": round(sel_at(ref, 0.8), 4), "sel80_best_free": round(sel_at(free[best], 0.8), 4), "outcome": outcome, "comparisons": pooled_cmps})
    # ---- H3 per-policy report: selective-accuracy gain over the unguarded floor at coverage 0.8, pooled seeds, clustered bootstrap
    h3 = []
    for ds in datasets:
        for m in models:
            floor = pooled(cells, ds, m, seeds, "unguarded")
            if not floor:
                continue
            for pol in policies:
                if pol == "unguarded":
                    continue
                rows = pooled(cells, ds, m, seeds, pol)
                if not rows:
                    continue
                d, lo, hi, p = clustered_bootstrap(rows, floor, lambda rr: sel_at(rr, 0.8), B, seed=303)
                h3.append({"dataset": ds, "model": m, "policy": pol, "n_items": len(rows), "sel80_gain_points": round(100 * d, 2), "ci": [round(100 * lo, 2), round(100 * hi, 2)], "p_value": round(p, 5),
                           "reference_line_3_points": bool(lo > 3.0)})
    # ---- seed agreement: share of items whose first-pass prediction is identical between seed pairs (replication check)
    seed_agreement = []

    def first_pass_key(ds_, m_, s_, basis):
        """Per-item first-pass identity under ONE basis for all seeds of a (dataset, model): 'pred' = the option id (None
        when the model gave no parsable option, which is itself comparable); 'spec' = the canonical specification from the
        run's cache file (matched by basename under experiments/cache/), falling back to the raw text."""
        rows = cells.get((ds_, m_, s_, "unguarded"), [])
        if basis == "pred":
            return {r["id"]: r.get("pred") for r in rows}
        run = next((rr for rr in runs if rr["dataset"] == ds_ and rr["model"] == m_ and int(rr["seed"]) == s_ and rr["method"] == "unguarded"), None)
        cache_name = Path(str(((run or {}).get("extra") or {}).get("cache") or "")).name
        cache_path = LANE / "experiments" / "cache" / cache_name if cache_name else None
        out = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[rec["id"]] = rec.get("spec_key") if rec.get("spec_key") is not None else (rec.get("text") or "")
            return {r["id"]: out.get(r["id"]) for r in rows}
        return {r["id"]: (bool(r.get("correct")), bool(r.get("declined")), bool(r.get("format_error"))) for r in rows}

    def basis_for(ds_, m_):
        """'pred' when at least half of the rows of every seed carry an option id; otherwise 'spec' (structured tasks)."""
        for s_ in seeds:
            rows = cells.get((ds_, m_, s_, "unguarded"), [])
            if rows and sum(r.get("pred") is not None for r in rows) < 0.5 * len(rows):
                return "spec"
        return "pred"

    for ds in datasets:
        for m in models:
            basis = basis_for(ds, m)
            ref = {s: first_pass_key(ds, m, s, basis) for s in seeds}
            pairs = {}
            for i, a_ in enumerate(seeds):
                for b_ in seeds[i + 1:]:
                    ids = set(ref.get(a_, {})) & set(ref.get(b_, {}))
                    if ids:
                        pairs[f"{skey(a_)}{skey(b_)}"] = round(sum(ref[a_][x] == ref[b_][x] for x in ids) / len(ids), 4)
            if pairs:
                seed_agreement.append({"dataset": ds, "model": m, "identical_prediction_share": pairs, "basis": "option id" if basis == "pred" else "canonical specification"})
    # ---- wrong-answer control
    wrong_control = []
    for m in models:
        for s in seeds:
            v = cells.get(("teleqna", m, s, METHOD)); w = cells.get(("teleqna", m, s, "verifier_wrong_control"))
            if v and w:
                right = [r["score"] for r in v if r["correct"]]
                wrong = [r["score"] for r in w]
                wrong_control.append({"model": m, "seed": s, "auroc_right_vs_wrong": round(auroc(right, wrong), 4), "n_right": len(right), "n_wrong": len(wrong)})
    # ---- combined signal, held-out seed
    combined = []
    combined_oof = {}  # the out-of-fold combined scores below, kept per (dataset, model) for combined_vs_best (read only)
    for ds in datasets:
        for m in models:
            feats = {}
            for s in seeds:
                trio = [cells.get((ds, m, s, k)) for k in ("logprob_gate", "margin_gate", METHOD)]
                if all(trio):
                    by_id = {}
                    for k, rows in zip(("lp", "mg", "vf"), trio):
                        for r in rows:
                            by_id.setdefault(r["id"], {})[k] = r["score"]; by_id[r["id"]]["y"] = int(r["correct"])
                    feats[s] = [(v["lp"], v["mg"], v["vf"], v["y"], i) for i, v in by_id.items() if len(v) == 4]
            if feats:
                # item-level 5-fold cross-validation: folds by question id, every seed row of an id in the same fold,
                # so the calibrator is never scored on a question it was fitted on
                all_rows = [(sd, a, b, c, y, i) for sd in feats for a, b, c, y, i in feats[sd]]
                ids = sorted({i for _, _, _, _, _, i in all_rows}); rng = random.Random(7); rng.shuffle(ids)
                fold_of = {i: k % 5 for k, i in enumerate(ids)}
                oof = []
                for k in range(5):
                    tr = [r for r in all_rows if fold_of[r[5]] != k]; te = [r for r in all_rows if fold_of[r[5]] == k]
                    if not tr or not te:
                        continue
                    Xtr, mu, sdv = standardise([[logit_feat(a), logit_feat(b), logit_feat(c)] for _, a, b, c, _, _ in tr])
                    w = logistic_fit(Xtr, [y for _, _, _, _, y, _ in tr])
                    for sd, a, b, c, y, i in te:
                        f = [(logit_feat(a) - mu[0]) / sdv[0], (logit_feat(b) - mu[1]) / sdv[1], (logit_feat(c) - mu[2]) / sdv[2]]
                        z = max(-700.0, min(700.0, w[0] + w[1] * f[0] + w[2] * f[1] + w[3] * f[2]))
                        oof.append({"id": f"{sd}:{i}", "score": 1 / (1 + math.exp(-z)), "correct": bool(y)})
                combined_oof[(ds, m)] = oof
                pooled_v = pooled(cells, ds, m, seeds, METHOD); pooled_mg = pooled(cells, ds, m, seeds, "margin_gate"); pooled_lp = pooled(cells, ds, m, seeds, "logprob_gate")
                combined.append({"dataset": ds, "model": m, "cv": "5-fold by question id, all seeds of an id in one fold; logit-transformed, standardised features", "n_oof": len(oof), "aurc_combined": round(aurc(oof), 4),
                                 "aurc_verifier": round(aurc(pooled_v), 4), "aurc_margin": round(aurc(pooled_mg), 4), "aurc_logprob": round(aurc(pooled_lp), 4)})
    # ---- H2 ordering transfer with bootstrap CI over items (unguarded excluded: constant score)
    transfer = []
    h2_excluded = []
    for m in models:
        # pre-registered format rule: a model above the format/structure-failure limit on a dataset is excluded from H2
        rates = [c["format_failure_rate"] for c in per_cell if c["model"] == m and c["dataset"] == "netconfeval_t1" and c.get("format_failure_rate") is not None]
        if rates and sum(rates) / len(rates) > format_limit:
            h2_excluded.append({"model": m, "netconfeval_format_failure_rate": round(sum(rates) / len(rates), 4), "rule": f"> {format_limit}"})
            continue
        for s in seeds:
            # policies present on BOTH datasets for this model and seed (the 70B lacks k=5 on NetConfEval by pre-registration)
            tpols = [p for p in policies if p != "unguarded" and ("teleqna", m, s, p) in cells and ("netconfeval_t1", m, s, p) in cells]
            if len(tpols) >= 3:
                ta = {pol: cells[("teleqna", m, s, pol)] for pol in tpols}; tb = {pol: cells[("netconfeval_t1", m, s, pol)] for pol in tpols}
                a = [aurc(ta[p]) for p in tpols]; b = [aurc(tb[p]) for p in tpols]
                ia = {pol: {r["id"]: r for r in ta[pol]} for pol in tpols}; ib = {pol: {r["id"]: r for r in tb[pol]} for pol in tpols}
                ids_a = sorted(ia[tpols[0]]); ids_b = sorted(ib[tpols[0]])
                rng = random.Random(1000 + s); taus = []
                for _ in range(B_TAU):
                    sa = [ids_a[rng.randrange(len(ids_a))] for _ in ids_a]; sb = [ids_b[rng.randrange(len(ids_b))] for _ in ids_b]
                    taus.append(kendall_tau([aurc([ia[p][i] for i in sa if i in ia[p]]) for p in tpols], [aurc([ib[p][i] for i in sb if i in ib[p]]) for p in tpols]))
                taus.sort()
                transfer.append({"model": m, "seed": s, "policies": tpols, "kendall_tau": round(kendall_tau(a, b), 3), "ci95": [round(taus[int(0.025 * B_TAU)], 3), round(taus[int(0.975 * B_TAU) - 1], 3)], "bootstrap_resamples": B_TAU})
    # ---------------------------------------------------------------- tables
    OUT.mkdir(exist_ok=True); (OUT / "tables").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    agg = defaultdict(list)
    for c in per_cell:
        agg[(c["dataset"], c["model"], c["method"])].append(c)
    mean = lambda cs, k: (sum(c[k] for c in cs if c.get(k) is not None) / max(1, sum(1 for c in cs if c.get(k) is not None))) if any(c.get(k) is not None for c in cs) else float("nan")
    tables = []

    def emit(name: str, lines: list[str]):
        p = OUT / "tables" / f"{name}.tex"; p.write_text("\n".join(lines) + "\n"); tables.append({"id": f"tbl-{name}", "path": f"analysis/tables/{name}.tex", "sha256": sha(p)})
    # main
    is_main = lambda ds, m, pol: ds in primary_datasets and m in bar_models and pol in policies
    lines = ["\\begin{tabular}{llrrrrrrrr}", "\\toprule", "Task & Model & Policy & AURC & Sel@0.9 & Sel@0.8 & Sel@0.7 & ECE & Sat. & AUROC \\\\", "\\midrule"]
    lines_s = list(lines)
    # print-size main table (table* at \footnotesize, no \resizebox): 8 columns, short labels, unguarded floor moved to floor.tex
    lines_c = ["\\begin{tabular}{lllrrrrr}", "\\toprule", "Task & Model & Policy & AURC & Sel@0.8 & ECE & Sat. & AUROC \\\\", "\\midrule"]
    from itertools import product
    seen = set()
    for (ds, m, pol), cs in sorted(agg.items()):
        seen.add((ds, m, pol))
        own = mean(cs, "own_answer_auroc"); own_s = f"{own:.3f}" if own == own else "--"
        row = f"{dlabel(ds)} & {mlabel(m)} & {plabel(pol)} & {mean(cs,'aurc'):.3f} & {mean(cs,'sel90'):.3f} & {mean(cs,'sel80'):.3f} & {mean(cs,'sel70'):.3f} & {mean(cs,'ece'):.3f} & {mean(cs,'saturation_rate'):.2f} & {own_s} \\\\"
        (lines if is_main(ds, m, pol) else lines_s).append(row)
        if is_main(ds, m, pol) and pol != "unguarded":
            lines_c.append(f"{sdlabel(ds)} & {smlabel(m)} & {plabel(pol)} & {mean(cs,'aurc'):.3f} & {mean(cs,'sel80'):.3f} & {mean(cs,'ece'):.3f} & {mean(cs,'saturation_rate'):.2f} & {own_s} \\\\")
        key = mkey(ds, m, pol)
        numbers[f"sat{key}"] = f"{100*mean(cs,'saturation_rate'):.0f}"; numbers[f"ownAuroc{key}"] = own_s
        numbers[f"aurc{key}"] = f"{mean(cs,'aurc'):.3f}"; numbers[f"selEighty{key}"] = f"{mean(cs,'sel80'):.3f}"; numbers[f"selNinety{key}"] = f"{mean(cs,'sel90'):.3f}"; numbers[f"selSeventy{key}"] = f"{mean(cs,'sel70'):.3f}"
        numbers[f"ece{key}"] = f"{mean(cs,'ece'):.3f}"; numbers[f"brier{key}"] = f"{mean(cs,'brier'):.3f}"; numbers[f"acc{key}"] = f"{100*mean(cs,'accuracy'):.1f}"
    # pre-registered cells that were excluded by the protocol appear as explicit rows so the omission is visible
    for ds in primary_datasets:
        for m in bar_models:
            for pol in policies:
                if (ds, m, pol) not in seen:
                    lines.append(f"{dlabel(ds)} & {mlabel(m)} & {plabel(pol)} & \\multicolumn{{7}}{{l}}{{not run (pre-registered compute reduction)}} \\\\")
                    lines_c.append(f"{sdlabel(ds)} & {smlabel(m)} & {plabel(pol)} & \\multicolumn{{5}}{{l}}{{not run (pre-registered compute reduction)}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("main_full", lines)
    lines_c += ["\\bottomrule", "\\end{tabular}"]; emit("main", lines_c)
    # floor table (the unguarded rows dropped from the compact main table): accuracy, format failures, declines
    lines = ["\\begin{tabular}{llrrr}", "\\toprule", "Task & Model & Unguarded acc. & Format fail. & Declined \\\\", "\\midrule"]
    for ds in primary_datasets:
        for m in bar_models:
            cs = agg.get((ds, m, "unguarded"))
            if cs:
                fv = mean(cs, "format_failure_rate"); dv = mean(cs, "declined_rate")
                lines.append(f"{sdlabel(ds)} & {smlabel(m)} & {100*mean(cs,'accuracy'):.1f}\\% & " + (f"{100*fv:.1f}\\%" if fv == fv else "--") + " & " + (f"{100*dv:.1f}\\%" if dv == dv else "--") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("floor", lines)
    if len(lines_s) > 4:
        lines_s += ["\\bottomrule", "\\end{tabular}"]; emit("sensitivity", lines_s)
    # cost table (per dataset, model, policy): tokens/item, extra tokens, seconds/item
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule", "Dataset & Model & Policy & Tokens/item & Extra tokens & s/item \\\\", "\\midrule"]
    for (ds, m, pol), cs in sorted(agg.items()):
        if not is_main(ds, m, pol):
            continue
        lines.append(f"{dlabel(ds)} & {mlabel(m)} & {plabel(pol)} & {mean(cs,'tokens_per_item'):.0f} & {mean(cs,'extra_tokens_per_item'):.0f} & {mean(cs,'latency_s_per_item'):.2f} \\\\")
        key = mkey(ds, m, pol); numbers[f"tok{key}"] = f"{mean(cs,'tokens_per_item'):.0f}"; numbers[f"extraTok{key}"] = f"{mean(cs,'extra_tokens_per_item'):.0f}"; numbers[f"lat{key}"] = f"{mean(cs,'latency_s_per_item'):.2f}"
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("cost", lines)
    # format failures per (dataset, model), first pass shared so any policy's rate is the model's rate
    lines = ["\\begin{tabular}{llrr}", "\\toprule", "Dataset & Model & Format-failure rate & Declined rate \\\\", "\\midrule"]
    for ds in datasets:
        for m in models:
            cs = [c for c in per_cell if c["dataset"] == ds and c["model"] == m and c.get("format_failure_rate") is not None]
            if cs:
                v = mean(cs, "format_failure_rate"); dv = mean(cs, "declined_rate")
                lines.append(f"{dlabel(ds)} & {mlabel(m)} & {100*v:.1f}\\% & {100*dv:.1f}\\% \\\\")
                numbers[f"fmtFail{mkey(ds, m)}"] = f"{100*v:.1f}"; numbers[f"declined{mkey(ds, m)}"] = f"{100*dv:.1f}"
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("format_failures", lines)
    # per-category AURC on TeleQnA (mean over seeds) per model x policy
    cats = sorted({r.get("category") for k, rows in cells.items() if k[0] == "teleqna" for r in rows if r.get("category")})
    if cats:
        percat = {}
        for m in models:
            for pol in policies:
                for cat in cats:
                    vals = []
                    for s in seeds:
                        rows = [r for r in cells.get(("teleqna", m, s, pol), []) if r.get("category") == cat]
                        if rows:
                            vals.append(aurc(rows))
                    if vals:
                        percat[(m, pol, cat)] = sum(vals) / len(vals)
        lines = ["\\begin{tabular}{ll" + "r" * len(cats) + "}", "\\toprule", "Model & Policy & " + " & ".join(texsafe(c) for c in cats) + " \\\\", "\\midrule"]
        for m in models:
            for pol in policies:
                if any((m, pol, c) in percat for c in cats):
                    lines.append(f"{mlabel(m)} & {plabel(pol)} & " + " & ".join(f"{percat[(m, pol, c)]:.3f}" if (m, pol, c) in percat else "--" for c in cats) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]; emit("per_category", lines)
    # H1 table: pre-registered (dataset, model) rows; single-seed sensitivity rows (dense model, full benchmark) in their own table
    OUTCOME = {"verifier_better": "verifier better", "free_signal_better": "free signal better", "no_significant_difference": "no significant difference"}
    lines_h = ["\\begin{tabular}{llrrrl}", "\\toprule", "Task & Model & AURC verifier & AURC best free & $\\Delta$ [95\\% CI] & Outcome \\\\", "\\midrule"]
    # the print-size H1 table is single-column (task and model in one cell, rows in task then model order); the
    # sensitivity table keeps the full header above
    lines = ["\\begin{tabular}{lrrrl}", "\\toprule", "Task, model & Verifier & Best free & $\\Delta$ [95\\% CI] & Outcome \\\\", "\\midrule"]
    OUTCOME_SHORT = {"verifier_better": "verifier", "free_signal_better": "free", "no_significant_difference": "n.s."}
    FREE_SHORT = {"logprob_gate": "logprob", "margin_gate": "margin"}  # column labels of the single-column print table
    _order = {(ds, m): i for i, (ds, m) in enumerate((ds, m) for ds in primary_datasets for m in bar_models)}
    for h in sorted(h1, key=lambda h: (_order.get((h["dataset"], h["model"]), len(_order)), h1.index(h))):
        bc = next(c for c in h["comparisons"] if c["baseline"] == h["best_free_signal"])
        row = f"{dlabel(h['dataset'])} & {mlabel(h['model'])} & {h['aurc_verifier']:.3f} & {h['aurc_best_free']:.3f} ({plabel(h['best_free_signal'])}) & {bc['aurc_delta']:+.3f} [{bc['ci'][0]:+.3f}, {bc['ci'][1]:+.3f}] & {OUTCOME.get(h['outcome'], h['outcome'])} \\\\"
        row_c = f"{sdlabel(h['dataset'])} {smlabel(h['model'])} & {h['aurc_verifier']:.3f} & {h['aurc_best_free']:.3f} ({FREE_SHORT.get(h['best_free_signal'], splabel(h['best_free_signal']))}) & {bc['aurc_delta']:+.3f} [{bc['ci'][0]:+.3f}, {bc['ci'][1]:+.3f}] & {OUTCOME_SHORT.get(h['outcome'], h['outcome'])} \\\\"
        if h["dataset"] in primary_datasets and h["model"] in bar_models:
            lines.append(row_c)  # print-size: short labels and outcome text
        else:
            lines_h.append(row)
        key = mkey(h["dataset"], h["model"]); numbers[f"hOneDelta{key}"] = f"{bc['aurc_delta']:+.3f}"; numbers[f"hOneCiLo{key}"] = f"{bc['ci'][0]:+.3f}"; numbers[f"hOneCiHi{key}"] = f"{bc['ci'][1]:+.3f}"; numbers[f"hOneP{key}"] = f"{bc['corrected_p']:.3g}"
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("h1", lines)
    if len(lines_h) > 4:
        lines_h += ["\\bottomrule", "\\end{tabular}"]; emit("h1_sensitivity", lines_h)
    # H3 table: gain over the floor at coverage 0.8, points, clustered bootstrap CI
    # wide layout: one row per (task, model), one column per policy: gain [lo, hi] in points
    h3_pols = [p for p in policies if p != "unguarded"]
    lines = ["\\begin{tabular}{ll" + "r" * len(h3_pols) + "}", "\\toprule", "Task & Model & " + " & ".join(plabel(p) for p in h3_pols) + " \\\\", "\\midrule"]
    h3_by = {(h["dataset"], h["model"], h["policy"]): h for h in h3}
    # print-size H3 table: short task/model/policy labels, cell text '+g [lo, hi]' with one decimal
    lines_c = ["\\begin{tabular}{ll" + "r" * len(h3_pols) + "}", "\\toprule", "Task & Model & " + " & ".join(splabel(p) for p in h3_pols) + " \\\\", "\\midrule"]
    for ds in primary_datasets:
        for m in bar_models:
            if any((ds, m, p) in h3_by for p in h3_pols):
                cells_txt = []; cells_c = []
                for p in h3_pols:
                    h = h3_by.get((ds, m, p))
                    cells_txt.append(f"{h['sel80_gain_points']:+.1f} [{h['ci'][0]:+.1f}, {h['ci'][1]:+.1f}]" if h else "not run")
                    cells_c.append(f"{h['sel80_gain_points']:+.1f} [{h['ci'][0]:.1f}, {h['ci'][1]:.1f}]" if h else "not run")
                lines.append(f"{dlabel(ds)} & {mlabel(m)} & " + " & ".join(cells_txt) + " \\\\")
                lines_c.append(f"{sdlabel(ds)} & {smlabel(m)} & " + " & ".join(cells_c) + " \\\\")
    for h in h3:
        key = mkey(h["dataset"], h["model"], h["policy"]); numbers[f"hThreeGain{key}"] = f"{h['sel80_gain_points']:+.1f}"; numbers[f"hThreeLo{key}"] = f"{h['ci'][0]:+.1f}"; numbers[f"hThreeHi{key}"] = f"{h['ci'][1]:+.1f}"
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("h3_full", lines)
    lines_c += ["\\bottomrule", "\\end{tabular}"]; emit("h3", lines_c)
    # seed agreement table
    pair_keys = [f"{skey(a_)}{skey(b_)}" for i, a_ in enumerate(seeds) for b_ in seeds[i + 1:]]
    lines = ["\\begin{tabular}{ll" + "r" * len(pair_keys) + "}", "\\toprule", "Task & Model & " + " & ".join(f"seeds {a_}={b_}" for i, a_ in enumerate(seeds) for b_ in seeds[i + 1:]) + " \\\\", "\\midrule"]  # identical option id / canonical specification
    for sa in seed_agreement:
        pr = sa["identical_prediction_share"]
        lines.append(f"{dlabel(sa['dataset'])} & {mlabel(sa['model'])} & " + " & ".join(f"{100*pr[k]:.1f}\\%" if k in pr else "--" for k in pair_keys) + " \\\\")
        for k, v in pr.items():
            numbers[f"seedSame{mkey(sa['dataset'], sa['model'])}{k}"] = f"{100*v:.1f}"
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("seed_agreement", lines)
    # ================================================================ post hoc reporting
    # Everything below is additional output, labelled post_hoc in the manifest. It reads cells/results and the pre-registered
    # outputs above but never changes them; the pre-registered functions are called unchanged on modified row copies.
    post_hoc = {}
    # ---- A. decline sensitivity (NetConfEval): declined rows forced to the bottom of every policy's ranking
    decl_ds = "netconfeval_t1"
    decl_pairs = [("VerifierMargin", METHOD, "margin_gate"), ("SelfConsistencyMargin", "self_consistency", "margin_gate"), ("VerifierFloor", METHOD, "error_abstain_floor")]
    decl_rows_out = []
    decl_moved = {}
    for m in bar_models:
        pol_rows = {pol: pooled(cells, decl_ds, m, seeds, pol) for pol in policies}
        pol_rows = {pol: rr for pol, rr in pol_rows.items() if rr}
        if METHOD not in pol_rows or "unguarded" not in pol_rows:
            continue
        flags = {pol: {f"{s}:{r['id']}" for s in seeds for r in cells.get((decl_ds, m, s, pol), []) if r.get("declined")} for pol in pol_rows}
        declined_ids = set().union(*flags.values())
        consistent = all(f == flags["unguarded"] for f in flags.values())
        moved = {pol: rows_declines_last(rr, declined_ids) for pol, rr in pol_rows.items()}
        moved["error_abstain_floor"] = moved["unguarded"]
        decl_moved[m] = moved  # reused by the H4 decline-rule check (section J); read only
        aurcs_ds ={pol: round(aurc(rr), 4) for pol, rr in moved.items()}
        aurcs_primary = {pol: round(aurc(rr), 4) for pol, rr in pol_rows.items()}
        tests = []
        for name, a_pol, b_pol in decl_pairs:
            if a_pol not in moved or b_pol not in moved:
                continue
            d, lo, hi, p = clustered_bootstrap(moved[a_pol], moved[b_pol], aurc, B, seed=505)
            tests.append({"pair": name, "a": a_pol, "b": b_pol, "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5)})
            numbers[f"declSensDelta{mkey(m)}{name}"] = f"{d:+.3f}"; numbers[f"declSensLo{mkey(m)}{name}"] = f"{lo:+.3f}"; numbers[f"declSensHi{mkey(m)}{name}"] = f"{hi:+.3f}"; numbers[f"declSensP{mkey(m)}{name}"] = f"{p:.3g}"
        for pol, v in aurcs_ds.items():
            numbers[f"declSensAurc{mkey(m, pol)}"] = f"{v:.3f}"
        decl_rows_out.append({"dataset": decl_ds, "model": m, "seeds_pooled": seeds, "n_items": len(pol_rows[METHOD]), "n_declined": len(declined_ids),
                              "n_declined_counted_correct": sum(1 for r in pol_rows["unguarded"] if r["id"] in declined_ids and r["correct"]), "declined_flags_identical_across_policies": consistent,
                              "aurc_declines_last": aurcs_ds, "aurc_primary_rule": aurcs_primary, "tests": tests,
                              "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 505), AURC difference a minus b"})
    post_hoc["decline_sensitivity"] = {"post_hoc": True, "note": "POST-HOC: NetConfEval AURC with every declined row ranked below all others for every policy (declines abstained first, tied); error-abstain floor = unguarded under that rule; pre-registered rule keeps each policy's own score for declines.",
                                       "rows": decl_rows_out}
    if decl_rows_out:
        dcols = [p for p in policies if p != "unguarded"] + ["error_abstain_floor"]
        lines = ["\\begin{tabular}{l" + "r" * len(dcols) + "}", "\\toprule", "Model & " + " & ".join(splabel(p) for p in dcols) + " \\\\", "\\midrule"]
        for dr in decl_rows_out:
            lines.append(f"{smlabel(dr['model'])} & " + " & ".join(f"{dr['aurc_declines_last'][p]:.3f}" if p in dr["aurc_declines_last"] else "--" for p in dcols) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]; emit("decline_sensitivity", lines)
    # ---- B. tie-resolution sensitivity: H1 AURCs when scores are compared at 1e-6 resolution
    tie_rows = []
    for h in h1:
        ds, m, best = h["dataset"], h["model"], h["best_free_signal"]
        ref = pooled(cells, ds, m, seeds, METHOD); fr = pooled(cells, ds, m, seeds, best)
        av, af = aurc(ref), aurc(fr)
        tv, tf = aurc(rows_at_resolution(ref)), aurc(rows_at_resolution(fr))
        free_tie = {f: aurc(rows_at_resolution(pooled(cells, ds, m, seeds, f))) for f in FREE if pooled(cells, ds, m, seeds, f)}
        prereg_row = ds in primary_datasets and m in bar_models
        tie_rows.append({"dataset": ds, "model": m, "pre_registered_row": prereg_row, "best_free_signal": best,
                         "aurc_verifier_primary": round(av, 6), "aurc_best_free_primary": round(af, 6), "gap_primary": round(av - af, 6),
                         "aurc_verifier_1e6": round(tv, 6), "aurc_best_free_1e6": round(tf, 6), "gap_1e6": round(tv - tf, 6), "gap_shift": round((tv - tf) - (av - af), 6),
                         "gap_sign_changed": (av - af) * (tv - tf) < 0, "best_free_signal_at_1e6": min(free_tie, key=free_tie.get) if free_tie else None})
        k = mkey(ds, m)
        numbers[f"tieAurcVerifier{k}"] = f"{tv:.4f}"; numbers[f"tieAurcFree{k}"] = f"{tf:.4f}"; numbers[f"tieGap{k}"] = f"{tv - tf:+.4f}"; numbers[f"tieGapPrimary{k}"] = f"{av - af:+.4f}"
    pre_tie = [t for t in tie_rows if t["pre_registered_row"]]
    tie_max_shift = max((abs(t["gap_shift"]) for t in pre_tie), default=0.0)
    tie_sign_changes = sum(1 for t in pre_tie if t["gap_sign_changed"])
    numbers["tieMaxGapShift"] = f"{tie_max_shift:.4f}"; numbers["tieSignChanges"] = tie_sign_changes
    post_hoc["tie_sensitivity"] = {"post_hoc": True, "note": "POST-HOC: pooled H1 AURCs (verifier, best free signal, gap) with scores rounded to 6 decimals before ranking; ties then ranked in expectation as in the primary rule.",
                                   "rows": tie_rows, "max_abs_gap_shift_pre_registered": round(tie_max_shift, 6), "sign_changes_pre_registered": tie_sign_changes}
    # ---- C. extra latency per (dataset, model, policy): policy latency minus unguarded latency of the same seed, mean over seeds
    extra_lat = []
    for (ds, m, pol) in sorted(agg):
        per_seed, source = {}, set()
        for s in seeds:
            mv = (results.get((ds, m, s, pol)) or {}).get("metric_values") or {}
            if mv.get("policy_latency_s_per_item") is not None:
                per_seed[s] = float(mv["policy_latency_s_per_item"]); source.add("policy_latency_s_per_item")
                continue
            mu = (results.get((ds, m, s, "unguarded")) or {}).get("metric_values") or {}
            if mv.get("latency_s_per_item") is not None and mu.get("latency_s_per_item") is not None:
                per_seed[s] = float(mv["latency_s_per_item"]) - float(mu["latency_s_per_item"]); source.add("latency minus unguarded latency")
        if per_seed:
            v = sum(per_seed.values()) / len(per_seed)
            numbers[f"extraLat{mkey(ds, m, pol)}"] = f"{v:.2f}"
            extra_lat.append({"dataset": ds, "model": m, "policy": pol, "extra_latency_s_per_item": round(v, 4), "per_seed": {str(k_): round(x, 4) for k_, x in per_seed.items()}, "source": sorted(source)})
    post_hoc["extra_latency"] = {"post_hoc": True, "note": "POST-HOC: seconds per item added by each policy over the unguarded first pass of the same (dataset, model, seed), mean over seeds; policy_latency_s_per_item used when the result records it.",
                                 "rows": extra_lat}
    # ---- D. H2 Kendall tau across seeds: min, max, number of seeds whose bootstrap interval lies above / below zero
    tau_range = []
    for m in models:
        ts = [t for t in transfer if t["model"] == m]
        if not ts:
            continue
        above = sum(1 for t in ts if t["ci95"][0] > 0); below = sum(1 for t in ts if t["ci95"][1] < 0)
        tmin = min(t["kendall_tau"] for t in ts); tmax = max(t["kendall_tau"] for t in ts)
        numbers[f"tauMin{mkey(m)}"] = f"{tmin:.2f}"; numbers[f"tauMax{mkey(m)}"] = f"{tmax:.2f}"; numbers[f"tauSeedsAboveZero{mkey(m)}"] = above; numbers[f"tauSeedsBelowZero{mkey(m)}"] = below
        tau_range.append({"model": m, "n_seeds": len(ts), "tau_min": tmin, "tau_max": tmax, "seeds_ci_above_zero": above, "seeds_ci_below_zero": below})
    post_hoc["tau_range"] = {"post_hoc": True, "note": "POST-HOC: range of the per-seed H2 Kendall tau and count of seeds whose bootstrap CI excludes zero (above / below).", "rows": tau_range}
    # ---- E(i). share of first-pass answers that were not the argmax option (TeleQnA, from the first-pass cache)
    nonarg = []
    for m in models:
        for s in seeds:
            urows = cells.get(("teleqna", m, s, "unguarded"))
            run = next((rr for rr in runs if rr["dataset"] == "teleqna" and rr["model"] == m and int(rr["seed"]) == s and rr["method"] == "unguarded"), None)
            cache_name = Path(str(((run or {}).get("extra") or {}).get("cache") or "")).name
            cache_path = LANE / "experiments" / "cache" / cache_name if cache_name else None
            if not urows or not cache_path or not cache_path.exists():
                continue
            ids = {r["id"] for r in urows}; elig = non = 0
            for line in cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id") not in ids or rec.get("pred") is None:
                    continue
                opts = {str(k).strip(): float(v) for k, v in ((rec.get("top") or {}).get("alts") or {}).items() if str(k).strip().isdigit()}
                if str(rec["pred"]) not in opts:
                    continue
                elig += 1; non += opts[str(rec["pred"])] < max(opts.values())
            if elig:
                numbers[f"nonArgmax{mkey(m)}{skey(s)}"] = f"{100 * non / elig:.1f}"
                nonarg.append({"dataset": "teleqna", "model": m, "seed": s, "n_eligible": elig, "n_non_argmax": non, "share": round(non / elig, 4), "cache": cache_name})
    if nonarg:
        post_hoc["first_pass_nonargmax_share"] = {"post_hoc": True, "note": "POST-HOC: share of TeleQnA first-pass answers whose option token was not the most probable option token in top.alts (items with a parsable option listed in alts).",
                                                  "rows": nonarg}
    # ---- E(ii). self-consistency sample diversity (only when prediction rows carry a 'samples' list)
    diversity = []
    for (ds, m, s, pol), rows in sorted(cells.items()):
        if pol != "self_consistency":
            continue
        sam = [r["samples"] for r in rows if isinstance(r.get("samples"), list) and r["samples"]]
        if not sam:
            continue
        nd = [len({json.dumps(x, sort_keys=True) for x in smp}) for smp in sam]
        diversity.append({"dataset": ds, "model": m, "seed": s, "n_items": len(sam), "mean_distinct": round(sum(nd) / len(nd), 4), "all_identical_share": round(sum(1 for x in nd if x == 1) / len(nd), 4)})
    if diversity:
        for ds, m in sorted({(d["dataset"], d["model"]) for d in diversity}):
            ds_ = [d for d in diversity if d["dataset"] == ds and d["model"] == m]
            numbers[f"scDistinct{mkey(ds, m)}"] = f"{sum(d['mean_distinct'] for d in ds_) / len(ds_):.2f}"
            numbers[f"scAllSame{mkey(ds, m)}"] = f"{100 * sum(d['all_identical_share'] for d in ds_) / len(ds_):.1f}"
        post_hoc["sample_diversity"] = {"post_hoc": True, "note": "POST-HOC: k=5 self-consistency samples per item: mean number of distinct samples and share of items whose samples are all identical.",
                                        "rows": diversity}
    # ================================================================ NetConfEval analyses and run-level summaries
    # Additional keys, tables and macros only. They read cells/runs/h1 and the first-pass caches and call the unchanged
    # pre-registered functions (aurc, sel_at, clustered_bootstrap) on filtered or relabelled row copies.
    nce_ds = "netconfeval_t1"
    nce_pts = next((list(d.get("policy_types") or []) for d in protocol.get("datasets") or [] if str(d.get("id")) == nce_ds), []) or ["reachability", "waypoint", "loadbalancing"]

    def nce_cache(m_, s_):
        """First-pass cache records of the (NetConfEval, model, seed) unguarded run, first writer wins (run_cell.refresh_cache)."""
        # latest matching record wins, as in the main cell loader (later registry records overwrite earlier ones)
        run = next((rr for rr in reversed(runs) if rr["dataset"] == nce_ds and rr["model"] == m_ and int(rr["seed"]) == s_ and rr["method"] == "unguarded"), None)
        cache_name = Path(str(((run or {}).get("extra") or {}).get("cache") or "")).name
        cache_path = LANE / "experiments" / "cache" / cache_name if cache_name else None
        recs = {}
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                recs.setdefault(rec.get("id"), rec)
        return cache_name, recs
    fp_cache = {(m, s): nce_cache(m, s) for m in bar_models for s in seeds if (nce_ds, m, s, "unguarded") in cells}
    # cache vs predictions: the strict correctness stored with the cached first pass must equal the prediction rows'
    cache_check = []
    for (m, s), (cname, recs) in sorted(fp_cache.items()):
        urows = cells.get((nce_ds, m, s, "unguarded"), [])
        cache_check.append({"model": m, "seed": s, "cache": cname, "n_rows": len(urows), "n_with_record": sum(1 for r in urows if r["id"] in recs),
                            "n_with_official_row": sum(1 for r in urows if isinstance((recs.get(r["id"]) or {}).get("official"), dict)),
                            "n_correct_mismatch": sum(1 for r in urows if r["id"] in recs and bool(recs[r["id"]].get("correct")) != bool(r["correct"]))})
    # requirement policy types: offline reconstruction of the fixed frame with the official sampler, validated by re-scoring
    # every cached first pass against the reconstructed expected specification with the official compare_result
    frame, frame_error = {}, None
    try:
        frame = nce_frame(nce_pts)
    except Exception as exc:
        frame_error = f"{type(exc).__name__}: {exc}"
    pred_ids = {r["id"] for k, rr in cells.items() if k[0] == nce_ds for r in rr}
    frame_check = {"policy_types": nce_pts, "n_ids_frame": len(frame), "ids_match_predictions": (pred_ids == set(frame)) if frame else None, "n_official_rows": 0, "n_rescored": 0, "n_rescored_identical": 0, "error": frame_error}
    if frame:
        for (m, s), (_, recs) in fp_cache.items():
            for iid, rec in recs.items():
                if iid not in frame or not isinstance(rec.get("official"), dict):
                    continue
                frame_check["n_official_rows"] += 1
                row = nce_official_row(frame[iid]["expected"], rec.get("text") or "")
                if row is None:
                    continue
                frame_check["n_rescored"] += 1
                frame_check["n_rescored_identical"] += int(all(row.get(k_) == rec["official"].get(k_) for k_ in ("total", "success", "fail", "wrong", "accuracy")))
    frame_ok = bool(frame) and bool(frame_check["ids_match_predictions"]) and frame_check["n_rescored"] > 0 and frame_check["n_rescored"] == frame_check["n_rescored_identical"]
    item_of = lambda pid: pid.split(":", 1)[1]
    nce_decl = {m: {f"{s}:{r['id']}" for s in seeds for r in cells.get((nce_ds, m, s, "unguarded"), []) if r.get("declined")} for m in bar_models}
    nce_best = {h["model"]: h["best_free_signal"] for h in h1 if h["dataset"] == nce_ds}
    nce_primary = {h["model"]: next(c for c in h["comparisons"] if c["baseline"] == h["best_free_signal"]) for h in h1 if h["dataset"] == nce_ds}
    # ---- F. NetConfEval by batch size (pre-registered reporting, protocol design_notes)
    bs_rows = []
    lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule", "Model & Batch & Items & Unguarded acc. & Declined & AURC verifier & AURC margin & $\\Delta$ \\\\", "\\midrule"]
    for m in bar_models:
        pr = {pol: pooled(cells, nce_ds, m, seeds, pol) for pol in ("unguarded", METHOD, "margin_gate")}
        if not all(pr.values()):
            continue
        for b in sorted(NCE_BATCH_WORD):
            sub = {pol: [r for r in rr if nce_batch_size(item_of(r["id"])) == b] for pol, rr in pr.items()}
            u = sub["unguarded"]
            if not u:
                continue
            acc = sum(r["correct"] for r in u) / len(u); dsh = sum(1 for r in u if r["id"] in nce_decl[m]) / len(u)
            av, am = aurc(sub[METHOD]), aurc(sub["margin_gate"])
            n_items = len({item_of(r["id"]) for r in u})
            lines.append(f"{smlabel(m)} & {b} & {n_items} & {100*acc:.1f}\\% & {100*dsh:.1f}\\% & {av:.3f} & {am:.3f} & {av - am:+.3f} \\\\")
            w = NCE_BATCH_WORD[b]
            numbers[f"nceAcc{mkey(m)}B{w}"] = f"{100*acc:.1f}"; numbers[f"nceVerMinusMargin{mkey(m)}B{w}"] = f"{av - am:+.3f}"
            bs_rows.append({"dataset": nce_ds, "model": m, "batch_size": b, "n_items_per_seed": n_items, "n_rows_pooled": len(u), "seeds_pooled": seeds, "unguarded_accuracy": round(acc, 4), "declined_share": round(dsh, 4),
                            "aurc_verifier": round(av, 4), "aurc_margin": round(am, 4), "aurc_verifier_minus_margin": round(av - am, 4)})
    if bs_rows:
        lines += ["\\bottomrule", "\\end{tabular}"]; emit("nce_batch_size", lines)
    post_hoc["nce_batch_size"] = {"post_hoc": False, "pre_registered_reporting": True,
                                  "note": "Pre-registered reporting (protocol design_notes: per-batch-size NetConfEval tables): per bar model and batch size, items pooled over seeds; unguarded accuracy, declined share, tie-aware AURC of verifier_gated and margin_gate; batch size read from the item id 'nce-it{i}-b{size}-c{chunk}'.",
                                  "rows": bs_rows}
    # ---- G. POST-HOC lone-load-balancing sensitivity: single-requirement batches whose only requirement is load balancing
    lb_rows = []
    lone_lb = {i for i, f in frame.items() if len(f["types"]) == 1 and f["types"][0] == "loadbalancing"} if frame_ok else set()
    lb_per_seed = {}
    for m in bar_models:
        u = pooled(cells, nce_ds, m, seeds, "unguarded"); best = nce_best.get(m)
        if not frame_ok or not u or not best:
            continue
        per_seed = {str(s): sum(1 for r in cells.get((nce_ds, m, s, "unguarded"), []) if r["id"] in lone_lb) for s in seeds}
        lb_per_seed[m] = per_seed
        in_lb = [r for r in u if item_of(r["id"]) in lone_lb]
        errs = [r for r in u if not r["correct"] and r["id"] not in nce_decl[m]]
        acc = sum(r["correct"] for r in in_lb) / len(in_lb) if in_lb else float("nan")
        share = sum(1 for r in errs if item_of(r["id"]) in lone_lb) / len(errs) if errs else float("nan")
        ref = [r for r in pooled(cells, nce_ds, m, seeds, METHOD) if item_of(r["id"]) not in lone_lb]
        fr = [r for r in pooled(cells, nce_ds, m, seeds, best) if item_of(r["id"]) not in lone_lb]
        d, lo, hi, p = clustered_bootstrap(ref, fr, aurc, B, seed=101)
        k = mkey(m)
        numbers[f"nceLbAcc{k}"] = f"{100*acc:.1f}"; numbers[f"nceLbErrShare{k}"] = f"{100*share:.1f}"
        numbers[f"nceNoLbDelta{k}"] = f"{d:+.3f}"; numbers[f"nceNoLbLo{k}"] = f"{lo:+.3f}"; numbers[f"nceNoLbHi{k}"] = f"{hi:+.3f}"; numbers[f"nceNoLbP{k}"] = f"{p:.3g}"
        lb_rows.append({"dataset": nce_ds, "model": m, "seeds_pooled": seeds, "lone_lb_items_per_seed": per_seed, "lone_lb_unguarded_accuracy": round(acc, 4),
                        "n_non_declined_errors": len(errs), "n_non_declined_errors_lone_lb": sum(1 for r in errs if item_of(r["id"]) in lone_lb), "lone_lb_error_share": round(share, 4),
                        "best_free_signal": best, "n_rows_excluded_lone_lb": len(pooled(cells, nce_ds, m, seeds, METHOD)) - len(ref), "n_rows_kept": len(ref),
                        "aurc_verifier": round(aurc(ref), 4), "aurc_best_free": round(aurc(fr), 4), "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5),
                        "primary_aurc_delta": nce_primary.get(m, {}).get("aurc_delta"), "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 101 as primary_h1), verifier minus the SAME best free signal as primary_h1, uncorrected two-sided p"})
    lb_counts = sorted({v for ps in lb_per_seed.values() for v in ps.values()})
    if len(lb_counts) == 1:
        numbers["nceLbItems"] = lb_counts[0]
    post_hoc["nce_lone_lb_sensitivity"] = {"post_hoc": True, "note": "POST-HOC: NetConfEval single-requirement batches whose only requirement is load balancing (types from the offline reconstruction of the official sampler frame, validated by re-scoring the cached first passes with the official compare_result); their unguarded accuracy, their share of non-declined strict-rule errors, and the pooled H1 comparison with those items excluded.",
                                           "frame_reconstruction": frame_check, "frame_validated": frame_ok, "cache_check": cache_check, "lone_lb_items_per_seed_values": lb_counts,
                                           "skipped": None if frame_ok else "requirement policy types not recoverable offline (frame reconstruction failed or did not validate)", "rows": lb_rows}
    # ---- H. POST-HOC official-accuracy sensitivity: correct iff the official checker's accuracy field equals 1.0
    #      (nothing missing; superfluous entries allowed); declines, parse and structure failures carry no official row -> wrong
    off_rows = []
    for m in bar_models:
        best = nce_best.get(m); recs_ms = {s: fp_cache.get((m, s), ("", {}))[1] for s in seeds}
        u = pooled(cells, nce_ds, m, seeds, "unguarded")
        if not u or not best:
            continue
        missing = [r["id"] for r in u if item_of(r["id"]) not in recs_ms[int(r["id"].split(":", 1)[0])]]
        if missing:
            off_rows.append({"dataset": nce_ds, "model": m, "skipped": f"{len(missing)} pooled rows without a first-pass cache record (official row not recoverable)"})
            continue

        def off_correct(pid):
            off = recs_ms[int(pid.split(":", 1)[0])][item_of(pid)].get("official")
            return isinstance(off, dict) and float(off.get("accuracy") or 0.0) == 1.0
        relabel = lambda rr: [dict(r, correct=off_correct(r["id"])) for r in rr]
        u_off = relabel(u); ref = relabel(pooled(cells, nce_ds, m, seeds, METHOD)); fr = relabel(pooled(cells, nce_ds, m, seeds, best))
        acc = sum(r["correct"] for r in u_off) / len(u_off)
        errs = [r for r in u if not r["correct"] and r["id"] not in nce_decl[m]]
        share = sum(1 for r in errs if off_correct(r["id"])) / len(errs) if errs else float("nan")
        d, lo, hi, p = clustered_bootstrap(ref, fr, aurc, B, seed=101)
        free_off = {f: aurc(relabel(pooled(cells, nce_ds, m, seeds, f))) for f in FREE if pooled(cells, nce_ds, m, seeds, f)}
        k = mkey(m)
        numbers[f"nceOffAcc{k}"] = f"{100*acc:.1f}"; numbers[f"nceOffErrShare{k}"] = f"{100*share:.1f}"
        # five decimals: under this rule the larger models' AURCs are of order 1e-4, which three decimals would print as -0.000
        numbers[f"nceOffDelta{k}"] = f"{d:+.5f}"; numbers[f"nceOffLo{k}"] = f"{lo:+.5f}"; numbers[f"nceOffHi{k}"] = f"{hi:+.5f}"; numbers[f"nceOffP{k}"] = f"{p:.3g}"
        off_rows.append({"dataset": nce_ds, "model": m, "seeds_pooled": seeds, "n_rows": len(u), "unguarded_accuracy_strict": round(sum(r["correct"] for r in u) / len(u), 4), "unguarded_accuracy_official": round(acc, 4),
                         "n_non_declined_strict_errors": len(errs), "n_strict_errors_official_accuracy_one": sum(1 for r in errs if off_correct(r["id"])), "strict_error_share_official_accuracy_one": round(share, 4),
                         "best_free_signal": best, "best_free_signal_under_official_rule": min(free_off, key=free_off.get) if free_off else None,
                         "aurc_verifier": round(aurc(ref), 4), "aurc_best_free": round(aurc(fr), 4), "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5),
                         "primary_aurc_delta": nce_primary.get(m, {}).get("aurc_delta"), "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 101 as primary_h1), verifier minus the SAME best free signal as primary_h1, uncorrected two-sided p"})
    post_hoc["nce_official_accuracy_sensitivity"] = {"post_hoc": True, "note": "POST-HOC: NetConfEval correctness redefined as the official compare_result 'accuracy' field (success/expected) equal to 1.0, read from the 'official' row stored with each cached first pass: nothing missing, superfluous ('wrong') entries allowed; declines and parse/structure failures have no official row and stay wrong. Same scores, same best free signal as primary_h1.",
                                                     "cache_check": cache_check, "rows": off_rows}
    # ---- J. POST-HOC decline-rule check of H4: self_consistency_k1 minus verifier_gated with declines ranked last (as section A)
    h4_rows = []
    for m in bar_models:
        moved = decl_moved.get(m) or {}
        if "self_consistency_k1" not in moved or METHOD not in moved:
            continue
        d, lo, hi, p = clustered_bootstrap(moved["self_consistency_k1"], moved[METHOD], aurc, B, seed=505)
        name = "KOneVerifier"
        numbers[f"declSensDelta{mkey(m)}{name}"] = f"{d:+.3f}"; numbers[f"declSensLo{mkey(m)}{name}"] = f"{lo:+.3f}"; numbers[f"declSensHi{mkey(m)}{name}"] = f"{hi:+.3f}"; numbers[f"declSensP{mkey(m)}{name}"] = f"{p:.3g}"
        h4_rows.append({"dataset": decl_ds, "model": m, "pair": name, "a": "self_consistency_k1", "b": METHOD, "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5),
                        "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 505 as decline_sensitivity), AURC difference a minus b, declines-last rule"})
    post_hoc["decline_sensitivity_h4"] = {"post_hoc": True, "note": "POST-HOC: H4 under the declines-last rule of decline_sensitivity: self-consistency k=1 minus verifier AURC on NetConfEval per bar model (positive = verifier better).", "rows": h4_rows}
    # ---- K. POST-HOC matched-coverage H1: selective accuracy at coverage 0.8, verifier minus the best free signal, points
    sel_rows = []
    for h in h1:
        ds, m, best = h["dataset"], h["model"], h["best_free_signal"]
        ref = pooled(cells, ds, m, seeds, METHOD); fr = pooled(cells, ds, m, seeds, best)
        d, lo, hi, p = clustered_bootstrap(ref, fr, lambda rr: sel_at(rr, 0.8), B, seed=101)
        k = mkey(ds, m)
        numbers[f"hOneSelDelta{k}"] = f"{100*d:+.1f}"; numbers[f"hOneSelLo{k}"] = f"{100*lo:+.1f}"; numbers[f"hOneSelHi{k}"] = f"{100*hi:+.1f}"; numbers[f"hOneSelP{k}"] = f"{p:.3g}"
        sel_rows.append({"dataset": ds, "model": m, "pre_registered_row": ds in primary_datasets and m in bar_models, "best_free_signal": best, "n_items": len(ref),
                         "sel80_verifier": round(sel_at(ref, 0.8), 4), "sel80_best_free": round(sel_at(fr, 0.8), 4), "sel80_delta_points": round(100 * d, 2), "ci": [round(100 * lo, 2), round(100 * hi, 2)], "p_value": round(p, 5),
                         "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 101 as primary_h1) on tie-aware Sel@0.8, uncorrected two-sided p"})
    post_hoc["h1_sel80"] = {"post_hoc": True, "note": "POST-HOC: for every primary_h1 row, the pooled difference in tie-aware selective accuracy at coverage 0.8 (verifier minus the best free signal of that row), in points, clustered-bootstrap CI.", "rows": sel_rows}
    # ---- L. compute hours: wall clock of every current-runner full and sensitivity run in the registry
    from datetime import datetime
    comp_runs = [r for r in registry["runs"] if r.get("runner_sha256") == current and r.get("kind") in ("full", "sensitivity")]
    secs, attempt_secs, no_times = 0.0, 0.0, []
    for r in comp_runs:
        try:
            secs += (datetime.fromisoformat(str(r["finished_at"])) - datetime.fromisoformat(str(r["started_at"]))).total_seconds()
        except (KeyError, ValueError, TypeError):
            no_times.append(r.get("run_id"))
        attempt_secs += sum(float(a.get("seconds") or 0.0) for a in r.get("attempts") or [])
    numbers["TotalComputeHours"] = f"{secs / 3600:.1f}"
    post_hoc["compute_hours"] = {"post_hoc": True, "note": "POST-HOC: sum over the current-runner full and sensitivity runs of the registry of finished_at minus started_at (per-cell wall clock; concurrent cells are summed, so this is cell-hours, not elapsed calendar time). First-pass generation is charged to whichever cell of a (dataset, model, seed) ran it first.",
                                 "runner_sha256": current, "n_runs": len(comp_runs), "by_kind": {k_: sum(1 for r in comp_runs if r.get("kind") == k_) for k_ in ("full", "sensitivity")},
                                 "by_status": {st: sum(1 for r in comp_runs if r.get("status") == st) for st in sorted({str(r.get("status")) for r in comp_runs})},
                                 "wall_clock_hours": round(secs / 3600, 3), "attempt_seconds_hours": round(attempt_secs / 3600, 3), "runs_without_timestamps": no_times}
    # ================================================================ post hoc controls and descriptive checks
    # Additional keys, tables and macros only. Tables of this section are emitted after achieved_evidence is computed, so
    # achieved_evidence and the tables above do not depend on them. Unchanged pre-registered functions are called on
    # row copies; clustered_bootstrap_by differs from clustered_bootstrap only in the cluster key.
    c4_tables = []  # (name, lines), emitted after achieved_evidence
    C4 = "POST-HOC"
    # ---- M. POST-HOC option-aware verifier control (TeleQnA, seed 1): records of kind posthoc_control, method verifier_gated_opt
    OPT = "verifier_gated_opt"
    opt_recs, opt_stale = {}, []
    # a control record counts only when its control_script_sha256 is the protocol's current
    # posthoc_controls.verifier_gated_opt.script_sha256 or is listed in its script_sha256_history (as reproduce_matrix.control_done)
    opt_spec = (protocol.get("posthoc_controls") or {}).get(OPT) or {}
    opt_hashes = {str(h) for h in [opt_spec.get("script_sha256"), *(opt_spec.get("script_sha256_history") or [])] if h}
    for r in registry["runs"]:  # append order: a later record of the same (dataset, model, seed) replaces an earlier one
        if r.get("kind") != "posthoc_control" or r.get("method") != OPT or r.get("status") != "ok":
            continue
        if r.get("runner_sha256") != current:
            opt_stale.append({"run_id": r.get("run_id"), "reason": "runner_sha256 differs from the current runner"}); continue
        if not r.get("control_script_sha256") or str(r.get("control_script_sha256")) not in opt_hashes:
            opt_stale.append({"run_id": r.get("run_id"), "reason": "control_script_sha256 " + (f"{str(r.get('control_script_sha256'))[:12]} is not the protocol's script_sha256 or in script_sha256_history" if r.get("control_script_sha256") else "is missing")}); continue
        opt_recs[(str(r.get("dataset")), str(r.get("model")), int(r.get("seed", -1)))] = r
    opt_rows, opt_skipped = [], []
    opt_data = {}  # the paired seed-1 rows and metric values of each analysed control, read only by option_aware_sensitivity
    opt_models =[m for m in bar_models if ("teleqna", m, 1) in opt_recs] + sorted(m for (d_, m, s_) in opt_recs if d_ == "teleqna" and s_ == 1 and m not in bar_models)
    for m in opt_models:
        rec = opt_recs[("teleqna", m, 1)]
        o_rows = load_preds(rec); v_rows = cells.get(("teleqna", m, 1, METHOD)) or []
        best = next((h["best_free_signal"] for h in h1 if h["dataset"] == "teleqna" and h["model"] == m), None)
        f_rows = cells.get(("teleqna", m, 1, best)) or [] if best else []
        if not o_rows or not v_rows or not f_rows:
            opt_skipped.append({"model": m, "run_id": rec.get("run_id"), "reason": "control predictions, seed-1 verifier_gated cell or seed-1 best-free-signal cell missing"}); continue
        io, iv, i_f = ({r["id"]: r for r in rr} for rr in (o_rows, v_rows, f_rows))
        ids = sorted(set(io) & set(iv) & set(i_f))
        O, V, F = ([d_[i] for i in ids] for d_ in (io, iv, i_f))
        fp_diff = sum(1 for i in ids if io[i].get("pred") != iv[i].get("pred") or bool(io[i]["correct"]) != bool(iv[i]["correct"]))
        a_o, a_v, a_f = aurc(O), aurc(V), aurc(F)
        dv, lov, hiv, pv = paired_bootstrap(O, V, aurc, B, seed=404)
        dfr, lof, hif, pf = paired_bootstrap(O, F, aurc, B, seed=404)
        ids_set = set(ids)
        free_s1 = {f: aurc([r for r in (cells.get(("teleqna", m, 1, f)) or []) if r["id"] in ids_set]) for f in FREE if cells.get(("teleqna", m, 1, f))}
        mv_o = rec.get("metric_values") or {}; mv_v = (results.get(("teleqna", m, 1, METHOD)) or {}).get("metric_values") or {}
        try:
            from datetime import datetime as _dt
            wall = (_dt.fromisoformat(str(rec["finished_at"])) - _dt.fromisoformat(str(rec["started_at"]))).total_seconds() / 3600
        except (KeyError, ValueError, TypeError):
            wall = None
        opt_data[m] = {"ids": ids, "O": O, "V": V, "F": F, "best": best, "rec": rec, "mv_o": mv_o, "mv_v": mv_v, "dv": dv, "dfr": dfr}
        k = mkey(m)
        numbers[f"optAurc{k}"] = f"{a_o:.3f}";numbers[f"optBaseAurc{k}"] = f"{a_v:.3f}"; numbers[f"optFreeAurc{k}"] = f"{a_f:.3f}"
        numbers[f"optMinusVer{k}"] = f"{dv:+.3f}"; numbers[f"optMinusVerLo{k}"] = f"{lov:+.3f}"; numbers[f"optMinusVerHi{k}"] = f"{hiv:+.3f}"; numbers[f"optMinusVerP{k}"] = fmt_boot_p(pv, B)
        numbers[f"optMinusFree{k}"] = f"{dfr:+.3f}"; numbers[f"optMinusFreeLo{k}"] = f"{lof:+.3f}"; numbers[f"optMinusFreeHi{k}"] = f"{hif:+.3f}"; numbers[f"optMinusFreeP{k}"] = fmt_boot_p(pf, B)
        opt_rows.append({"dataset": "teleqna", "model": m, "seed": 1, "run_id": rec.get("run_id"), "out_dir": rec.get("out_dir"), "control_script_sha256": rec.get("control_script_sha256"),
                         "n_items": len(ids), "n_control_rows": len(o_rows), "first_pass_rows_differing_from_verifier_cell": fp_diff,
                         "best_free_signal": best, "best_free_signal_source": "primary_h1 row of (teleqna, model), evaluated on the seed-1 rows", "best_free_signal_on_seed1_rows": min(free_s1, key=free_s1.get) if free_s1 else None,
                         "aurc_option_aware": round(a_o, 4), "aurc_verifier": round(a_v, 4), "aurc_best_free": round(a_f, 4),
                         "sel80_option_aware": round(sel_at(O, 0.8), 4), "sel80_verifier": round(sel_at(V, 0.8), 4), "sel80_best_free": round(sel_at(F, 0.8), 4),
                         "opt_minus_verifier": {"aurc_delta": round(dv, 5), "ci": [round(lov, 5), round(hiv, 5)], "p_value": round(pv, 5)},
                         "opt_minus_best_free": {"aurc_delta": round(dfr, 5), "ci": [round(lof, 5), round(hif, 5)], "p_value": round(pf, 5)},
                         "extra_tokens_per_item": {"option_aware": mv_o.get("extra_tokens_per_item"), "verifier": mv_v.get("extra_tokens_per_item")},
                         "policy_latency_s_per_item": {"option_aware": mv_o.get("policy_latency_s_per_item"), "verifier": mv_v.get("policy_latency_s_per_item")},
                         "control_wall_clock_hours": round(wall, 3) if wall is not None else None,
                         "test": "paired bootstrap over items (unchanged paired_bootstrap, seed 404) on tie-aware AURC, seed-1 rows paired by item id, uncorrected two-sided p"})
    post_hoc["option_aware_control"] = {"post_hoc": True, "label": C4, "status": "run" if opt_rows else "not run",
                                        "note": f"{C4}: verifier_gated_opt = the pre-registered verifier (same wording, temperature 0, top-20 YES/NO score, threshold 0.5, request seeds with the same role as verifier_gated, so listing the options is the only difference) shown the question's answer options before the proposed answer, scoring the same cached seed-1 first passes (environment/control_option_aware.py; registry kind posthoc_control). Compared with verifier_gated and the H1 best free signal on the same seed-1 rows. Control compute is not part of compute_hours (full and sensitivity runs only).",
                                        "rows": opt_rows, "skipped": opt_skipped, "stale_records_ignored": opt_stale}
    if opt_rows:
        lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Model & AURC opt. & AURC verifier & AURC free & $\\Delta$ opt.$-$verifier [95\\% CI] & $\\Delta$ opt.$-$free [95\\% CI] & Sel@0.8 opt./ver./free \\\\", "\\midrule"]
        for o in opt_rows:
            n_ = {x: numbers[f"{x}{mkey(o['model'])}"] for x in ("optAurc", "optBaseAurc", "optFreeAurc", "optMinusVer", "optMinusVerLo", "optMinusVerHi", "optMinusFree", "optMinusFreeLo", "optMinusFreeHi")}  # same strings as the macros
            lines.append(f"{smlabel(o['model'])} & {n_['optAurc']} & {n_['optBaseAurc']} & {n_['optFreeAurc']} ({splabel(o['best_free_signal'])}) & {n_['optMinusVer']} [{n_['optMinusVerLo']}, {n_['optMinusVerHi']}] & {n_['optMinusFree']} [{n_['optMinusFreeLo']}, {n_['optMinusFreeHi']}] & {o['sel80_option_aware']:.3f}/{o['sel80_verifier']:.3f}/{o['sel80_best_free']:.3f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        c4_tables.append(("option_aware", lines))
    # ---- N. POST-HOC NetConfEval H1 with the bootstrap clustered by sampler ITERATION (batches of one iteration share requirements)
    iter_rows, iter_clusters = [], set()
    for m in bar_models:
        bc = nce_primary.get(m); best = nce_best.get(m)
        if not bc or not best:
            continue
        ref = pooled(cells, nce_ds, m, seeds, METHOD); fr = pooled(cells, nce_ds, m, seeds, best)
        d, lo, hi, p, ncl = clustered_bootstrap_by(ref, fr, aurc, B, 101, nce_iteration)
        iter_clusters.add(ncl)
        k = mkey(m)
        numbers[f"nceIterDelta{k}"] = f"{round(d, 5):+.3f}"; numbers[f"nceIterLo{k}"] = f"{lo:+.3f}"; numbers[f"nceIterHi{k}"] = f"{hi:+.3f}"; numbers[f"nceIterP{k}"] = fmt_boot_p(p, B)
        iter_rows.append({"dataset": nce_ds, "model": m, "seeds_pooled": seeds, "best_free_signal": best, "n_rows": len(ref), "n_clusters": ncl,
                          "aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5), "delta_equals_primary": round(d, 5) == bc.get("aurc_delta"),
                          "primary_ci": bc.get("ci"), "primary_p_uncorrected": bc.get("p_value"), "primary_p_holm": bc.get("corrected_p"),
                          "test": "bootstrap over sampler iterations (clustered_bootstrap_by, seed 101): each drawn iteration brings every batch size, chunk and seed row of that iteration; verifier minus the SAME best free signal as primary_h1; uncorrected two-sided p; percentile CI from few clusters is approximate"})
    if len(iter_clusters) == 1:
        numbers["nceIterClusters"] = next(iter(iter_clusters))
    post_hoc["nce_iteration_cluster"] = {"post_hoc": True, "label": C4, "note": f"{C4}: the pooled NetConfEval H1 comparison (verifier minus best free signal) with the resampling unit widened from the requirement batch to the sampler iteration (item id nce-it{{i}}-b{{size}}-c{{chunk}}; cluster = it{{i}}), because the 1-, 2-, 5- and 10-requirement batches of one iteration are drawn from the same sampled requirements.",
                                         "rows": iter_rows}
    # ---- O. POST-HOC trained combination vs the best single signal (existing out-of-fold combined scores, no refit)
    cvb_rows = []
    for (ds, m), oof in combined_oof.items():
        if ds not in primary_datasets or not (m in bar_models or ds == "teleqna"):
            continue
        oof_ids = {r["id"] for r in oof}
        singles = {pol: [r for r in pooled(cells, ds, m, seeds, pol) if r["id"] in oof_ids] for pol in (METHOD, "margin_gate", "logprob_gate")}
        singles = {pol: rr for pol, rr in singles.items() if rr}
        if not oof or not singles:
            continue
        s_aurc = {pol: aurc(rr) for pol, rr in singles.items()}
        best = min(s_aurc, key=s_aurc.get)
        d, lo, hi, p = clustered_bootstrap(oof, singles[best], aurc, B, seed=707)
        k = mkey(ds, m)
        numbers[f"combMinusBest{k}"] = f"{d:+.3f}"; numbers[f"combMinusBestLo{k}"] = f"{lo:+.3f}"; numbers[f"combMinusBestHi{k}"] = f"{hi:+.3f}"; numbers[f"combMinusBestP{k}"] = fmt_boot_p(p, B)
        cvb_rows.append({"dataset": ds, "model": m, "pre_registered_row": m in bar_models, "seeds_pooled": sorted({int(r["id"].split(":", 1)[0]) for r in oof}), "n_oof": len(oof),
                         "best_single_signal": best, "aurc_single": {pol: round(v, 4) for pol, v in s_aurc.items()}, "aurc_combined": round(aurc(oof), 4),
                         "combined_minus_best": {"aurc_delta": round(d, 5), "ci": [round(lo, 5), round(hi, 5)], "p_value": round(p, 5)},
                         "selection_on_same_data": True,
                         "test": "clustered bootstrap over items (unchanged clustered_bootstrap, seed 707), combined out-of-fold score minus the single signal with the lowest pooled AURC on the same rows (chosen on these data, which favours the single signal); the interval and p-value are conditional on that selection and descriptive only, not an inferential test; uncorrected two-sided p"})
    post_hoc["combined_vs_best"] = {"post_hoc": True, "label": C4, "selection_on_same_data": True, "note": f"{C4}: DESCRIPTIVE ONLY. The best single signal is selected on the same rows it is then compared on, so the reported interval and p-value are conditional on that selection and are not an inferential test (the selection is not resampled). Pooled AURC difference between the trained combination (the existing 5-fold out-of-fold logistic scores of combined_signal, not refitted) and the best single signal among verifier, margin and max-logprob, per primary dataset and bar model plus the dense model on TeleQnA, clustered-bootstrap CI.",
                                    "rows": cvb_rows}
    # ---- P. McNemar at coverage 0.8 for the main H1 comparison per (dataset, bar model, seed): pre-registered secondary test
    mc_rows = []
    cmp_by_id = {c["id"]: c for c in comparisons}
    for h in h1:
        if h["dataset"] not in primary_datasets or h["model"] not in bar_models:
            continue
        for s in seeds:
            c = cmp_by_id.get(f"cmp-{h['dataset']}-{h['model']}-s{s}-{h['best_free_signal']}")
            if not c:
                continue
            n01, n10, pm = c["mcnemar"]["n01"], c["mcnemar"]["n10"], c["mcnemar"]["p"]
            mc_rows.append({"dataset": h["dataset"], "model": h["model"], "seed": s, "baseline": h["best_free_signal"], "comparison_id": c["id"], "n01": n01, "n10": n10, "p": pm,
                            "odds_ratio_n01_over_n10": round(n01 / n10, 4) if n10 else None, "odds_ratio_undefined": n10 == 0})
    if mc_rows:
        lines = ["\\begin{tabular}{lllrrrrr}", "\\toprule", "Task & Model & Free signal & Seed & $n_{01}$ & $n_{10}$ & $p$ & OR \\\\", "\\midrule"]
        for r in mc_rows:
            p_txt = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
            or_txt = f"{r['odds_ratio_n01_over_n10']:.2f}" if r["n10"] else "n/a$^\\dagger$"
            lines.append(f"{sdlabel(r['dataset'])} & {smlabel(r['model'])} & {splabel(r['baseline'])} & {r['seed']} & {r['n01']} & {r['n10']} & {p_txt} & {or_txt} \\\\")
        lines.append("\\bottomrule")
        if any(r["n10"] == 0 for r in mc_rows):
            lines.append("\\multicolumn{8}{l}{$^\\dagger$ $n_{10}=0$: odds ratio $n_{01}/n_{10}$ undefined.} \\\\")
        lines.append("\\end{tabular}")
        c4_tables.append(("mcnemar_h1", lines))
    post_hoc["mcnemar_h1"] = {"post_hoc": False, "pre_registered_reporting": True, "generated_in": "post hoc reporting",
                              "note": "Pre-registered secondary test (statistics.paired_test, statistics.assumptions.mcnemar), table generated with the post hoc analyses: exact McNemar at matched coverage 0.8 for verifier_gated vs the primary_h1 best free signal per (dataset, bar model, seed), values from comparisons[].mcnemar; n01 = answered-correct by the verifier only, n10 = by the free signal only; odds ratio n01/n10 (> 1 favours the verifier; undefined when n10 = 0). Uncorrected.",
                              "rows": mc_rows}
    # ================================================================ option-aware control sensitivity
    # Additional key, table and macros only; the table is emitted after the tables of sections M-P, so their order is
    # unchanged. Reads the seed-1 rows of section M (opt_data) and calls the unchanged paired_bootstrap (local random.Random).
    c6_tables = []
    C6 = "POST-HOC"
    # ---- Q. POST-HOC option-aware control: 1e-6 tie resolution, matched-coverage Sel@0.8, extra tokens and seconds per item
    def extra_latency_s1(mv: dict, m_: str) -> tuple[float | None, str | None]:
        """Section C's rule on one seed-1 result: policy_latency_s_per_item when recorded, else latency minus the unguarded latency."""
        if mv.get("policy_latency_s_per_item") is not None:
            return float(mv["policy_latency_s_per_item"]), "policy_latency_s_per_item"
        mu = (results.get(("teleqna", m_, 1, "unguarded")) or {}).get("metric_values") or {}
        if mv.get("latency_s_per_item") is not None and mu.get("latency_s_per_item") is not None:
            return float(mv["latency_s_per_item"]) - float(mu["latency_s_per_item"]), "latency minus unguarded latency"
        return None, None
    opts_rows = []
    for m in opt_models:
        od = opt_data.get(m)
        if not od:
            continue
        O, V, F = od["O"], od["V"], od["F"]
        O6, V6, F6 = rows_at_resolution(O), rows_at_resolution(V), rows_at_resolution(F)
        tf, tlof, thif, tpf = paired_bootstrap(O6, F6, aurc, B, seed=404)
        tv, tlov, thiv, tpv = paired_bootstrap(O6, V6, aurc, B, seed=404)
        sel80 = lambda rr: sel_at(rr, 0.8)
        sf, slof, shif, spf = paired_bootstrap(O, F, sel80, B, seed=404)
        sv, slov, shiv, spv = paired_bootstrap(O, V, sel80, B, seed=404)
        ids_set = set(od["ids"])  # the paired id set of section M, not the option-aware rows' own ids
        free6 = {f: aurc(rows_at_resolution([r for r in (cells.get(("teleqna", m, 1, f)) or []) if r["id"] in ids_set])) for f in FREE if cells.get(("teleqna", m, 1, f))}
        tok_o, tok_v = od["mv_o"].get("extra_tokens_per_item"), od["mv_v"].get("extra_tokens_per_item")
        lat_o, src_o = extra_latency_s1(od["mv_o"], m); lat_v, src_v = extra_latency_s1(od["mv_v"], m)
        k = mkey(m)
        numbers[f"optTieMinusFree{k}"] = f"{tf:+.3f}"; numbers[f"optTieMinusFreeLo{k}"] = f"{tlof:+.3f}"; numbers[f"optTieMinusFreeHi{k}"] = f"{thif:+.3f}"; numbers[f"optTieMinusFreeP{k}"] = fmt_boot_p(tpf, B)
        numbers[f"optTieMinusVer{k}"] = f"{tv:+.3f}"; numbers[f"optTieMinusVerLo{k}"] = f"{tlov:+.3f}"; numbers[f"optTieMinusVerHi{k}"] = f"{thiv:+.3f}"; numbers[f"optTieMinusVerP{k}"] = fmt_boot_p(tpv, B)
        numbers[f"optSelMinusFree{k}"] = f"{100*sf:+.1f}"; numbers[f"optSelMinusFreeLo{k}"] = f"{100*slof:+.1f}"; numbers[f"optSelMinusFreeHi{k}"] = f"{100*shif:+.1f}"; numbers[f"optSelMinusFreeP{k}"] = fmt_boot_p(spf, B)
        numbers[f"optSelMinusVer{k}"] = f"{100*sv:+.1f}"; numbers[f"optSelMinusVerLo{k}"] = f"{100*slov:+.1f}"; numbers[f"optSelMinusVerHi{k}"] = f"{100*shiv:+.1f}"; numbers[f"optSelMinusVerP{k}"] = fmt_boot_p(spv, B)
        if tok_o is not None:
            numbers[f"optExtraTok{k}"] = f"{float(tok_o):.0f}"
        if tok_v is not None:
            numbers[f"optBaseExtraTok{k}"] = f"{float(tok_v):.0f}"
        if lat_o is not None:
            numbers[f"optExtraLat{k}"] = f"{lat_o:.2f}"
        if lat_v is not None:
            numbers[f"optBaseExtraLat{k}"] = f"{lat_v:.2f}"
        opts_rows.append({"dataset": "teleqna", "model": m, "seed": 1, "run_id": od["rec"].get("run_id"), "n_items": len(O), "best_free_signal": od["best"],
                          "best_free_signal_on_seed1_rows_at_1e6": min(free6, key=free6.get) if free6 else None,
                          "tie_1e6": {"aurc_option_aware": round(aurc(O6), 6), "aurc_verifier": round(aurc(V6), 6), "aurc_best_free": round(aurc(F6), 6),
                                      "opt_minus_best_free": {"aurc_delta": round(tf, 5), "ci": [round(tlof, 5), round(thif, 5)], "p_value": round(tpf, 5), "primary_aurc_delta": round(od["dfr"], 5)},
                                      "opt_minus_verifier": {"aurc_delta": round(tv, 5), "ci": [round(tlov, 5), round(thiv, 5)], "p_value": round(tpv, 5), "primary_aurc_delta": round(od["dv"], 5)}},
                          "sel80": {"option_aware": round(sel_at(O, 0.8), 4), "verifier": round(sel_at(V, 0.8), 4), "best_free": round(sel_at(F, 0.8), 4),
                                    "opt_minus_best_free_points": {"delta": round(100 * sf, 2), "ci": [round(100 * slof, 2), round(100 * shif, 2)], "p_value": round(spf, 5)},
                                    "opt_minus_verifier_points": {"delta": round(100 * sv, 2), "ci": [round(100 * slov, 2), round(100 * shiv, 2)], "p_value": round(spv, 5)}},
                          "cost": {"extra_tokens_per_item": {"option_aware": tok_o, "verifier": tok_v}, "extra_latency_s_per_item": {"option_aware": None if lat_o is None else round(lat_o, 4), "verifier": None if lat_v is None else round(lat_v, 4)},
                                   "latency_source": {"option_aware": src_o, "verifier": src_v}, "source": "metric_values of the control record and of the seed-1 verifier_gated result (all frame items, the rows paired above); latency rule of extra_latency"},
                          "test": "paired bootstrap over items (unchanged paired_bootstrap, seed 404 as option_aware_control), seed-1 rows paired by item id, uncorrected two-sided p; tie rows: scores rounded to 6 decimals (rows_at_resolution, as tie_sensitivity); Sel@0.8 tie-aware"})
    post_hoc["option_aware_sensitivity"] = {"post_hoc": True, "label": C6, "status": "run" if opts_rows else "not run",
                                            "note": f"{C6}: the option-aware control of option_aware_control on the same seed-1 rows and the same best free signal, (i) with scores compared at 1e-6 resolution (rounded to 6 decimals before ranking, ties ranked in expectation, as tie_sensitivity), (ii) as differences in tie-aware selective accuracy at coverage 0.8 (points), (iii) extra tokens and extra seconds per item of the option-aware verifier and of verifier_gated over the shared first pass. Uncorrected, single seed.",
                                            "rows": opts_rows}
    if opts_rows:
        lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
                 "& \\multicolumn{2}{c}{$\\Delta$AURC at $10^{-6}$ resolution [95\\% CI]} & \\multicolumn{2}{c}{$\\Delta$Sel@0.8, points [95\\% CI]} & \\multicolumn{2}{c}{Extra per item, opt./ver.} \\\\",
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
                 "Model & opt.$-$free & opt.$-$verifier & opt.$-$free & opt.$-$verifier & tokens & seconds \\\\", "\\midrule"]
        for o in opts_rows:
            k = mkey(o["model"]); n_ = lambda x: numbers.get(f"{x}{k}", "--")  # same strings as the macros
            lines.append(f"{smlabel(o['model'])} & {n_('optTieMinusFree')} [{n_('optTieMinusFreeLo')}, {n_('optTieMinusFreeHi')}] & {n_('optTieMinusVer')} [{n_('optTieMinusVerLo')}, {n_('optTieMinusVerHi')}]"
                         f" & {n_('optSelMinusFree')} [{n_('optSelMinusFreeLo')}, {n_('optSelMinusFreeHi')}] & {n_('optSelMinusVer')} [{n_('optSelMinusVerLo')}, {n_('optSelMinusVerHi')}]"
                         f" & {n_('optExtraTok')}/{n_('optBaseExtraTok')} & {n_('optExtraLat')}/{n_('optBaseExtraLat')} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        c6_tables.append(("option_aware_sensitivity", lines))
    # ---------------------------------------------------------------- figures (matplotlib; skipped if unavailable)
    figures = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType (Type 42) fonts, not Type 3
        import matplotlib.pyplot as plt

        def savefig(fig, name: str, caption: str):
            p = OUT / "figures" / f"{name}.pdf"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            figures.append({"id": f"fig-{name}", "path": f"analysis/figures/{name}.pdf", "sha256": sha(p), "caption": caption})
        # print-size figures: risk-coverage figures are included 7.0 in wide (one panel per model), the cost scatter and the
        # reliability diagram 3.4 in wide; fonts >= 7 pt (ticks, legend) and >= 8 pt (axis labels, titles), lines >= 1.0 pt
        FS_TICK, FS_LABEL, FS_LEGEND = 7, 8, 7
        from matplotlib.lines import Line2D
        # ONE fixed colour and line style per policy, keyed by the pre-registered policy list
        # (protocol matrix.policies_pre_registered order, so a policy missing from a panel never shifts the others), used
        # identically in every panel, every policy-coloured figure and every legend; bar models get fixed colours outside the
        # policy palette (h3_gain colours by model)
        pol_order = [str(p) for p in ((protocol.get("matrix") or {}).get("policies_pre_registered") or [])]
        pol_order += [p for p in policies if p not in pol_order]
        POL_COLOR = {p: f"C{i % 10}" for i, p in enumerate(pol_order)}
        POL_STYLE = {p: ({"lw": 1.8, "ls": "-"} if p in (METHOD,) + FREE else {"lw": 1.0, "ls": "--"}) for p in pol_order}
        MODEL_COLOR = {m: c for m, c in zip(bar_models, ("C7", "C8", "C9"))} if len(pol_order) <= 7 else {m: f"C{j % 10}" for j, m in enumerate(bar_models)}
        # 1-2: risk-coverage curves per dataset (one panel per model), mean curve over seeds; one shared legend below the panels
        for ds in datasets:
            ms = [m for m in models if any(k[0] == ds and k[1] == m for k in cells)]
            if not ms:
                continue
            ms = [m for m in bar_models if m in ms] or ms  # panels in pre-registered order (7B, 30B, 70B)
            fig, axes = plt.subplots(1, len(ms), figsize=(7.0, 1.65) if len(ms) > 1 else (3.4, 2.4), squeeze=False)
            plotted = set()
            for ax, m in zip(axes[0], ms):
                for pol in policies:
                    curves = [risk_coverage_curve(cells[(ds, m, s, pol)]) for s in seeds if (ds, m, s, pol) in cells]
                    if curves:
                        xs = [c[0] for c in curves[0]]; ys = [sum(c[i][1] for c in curves) / len(curves) for i in range(len(xs))]
                        ax.plot(xs, ys, label=POLICY_LABEL.get(pol, pol), color=POL_COLOR[pol], **POL_STYLE[pol]); plotted.add(pol)
                ax.set_title(SHORT_MODEL.get(m, MODEL_LABEL.get(m, m)), fontsize=FS_LABEL); ax.set_xlabel("coverage", fontsize=FS_LABEL); ax.grid(alpha=0.3); ax.tick_params(labelsize=FS_TICK)
            axes[0][0].set_ylabel("selective risk", fontsize=FS_LABEL)
            fig.tight_layout()
            # legend from the fixed mapping (every policy plotted in any panel), not from one panel's artists
            h_ = [Line2D([0], [0], color=POL_COLOR[p], **POL_STYLE[p]) for p in policies if p in plotted]; l_ = [POLICY_LABEL.get(p, p) for p in policies if p in plotted]
            fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4 if len(ms) > 1 else 2, fontsize=FS_LEGEND, frameon=False)
            savefig(fig, f"rc_{ds}", f"Risk-coverage curves on {ds} (mean over seeds), one panel per model.")
        # 3: cost vs AURC scatter (TeleQnA); two-column legend below the axes
        fig, ax = plt.subplots(figsize=(3.4, 1.35))  # column-wide and short, printed at natural size; policies by the fixed colours (legend in Fig. rc), models by marker
        markers = {"qwen2.5-coder-7b-instruct": "o", "qwen3-coder-30b-a3b-instruct": "s", "llama-3.3-70b-instruct": "^"}
        colors = POL_COLOR  # the fixed policy mapping
        for (ds, m, pol), cs in sorted(agg.items()):
            if ds == "teleqna" and m in bar_models and pol in policies and cs[0].get("extra_tokens_per_item") is not None:
                ax.scatter(mean(cs, "extra_tokens_per_item"), mean(cs, "aurc"), s=24, marker=markers.get(m, "o"), color=colors[pol], linewidths=1.0)
        handles = [Line2D([0], [0], marker="o", color=colors[p], ls="", label=POLICY_LABEL.get(p, p)) for p in policies] + \
                  [Line2D([0], [0], marker=markers[m], color="k", ls="", label=SHORT_MODEL.get(m, MODEL_LABEL.get(m, m))) for m in bar_models if m in markers]
        # short y label so it fits the 1.35 in figure height
        ax.set_xlabel("extra tokens per item (TeleQnA)", fontsize=FS_LABEL); ax.set_ylabel("AURC", fontsize=FS_LABEL); ax.grid(alpha=0.3); ax.tick_params(labelsize=FS_TICK); fig.tight_layout()
        savefig(fig, "cost_aurc", "AURC against extra tokens per item on TeleQnA: free signals sit at zero extra cost.")
        # 4: reliability diagrams (TeleQnA, per policy, pooled models/seeds); two-column legend below the axes
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        for pol in policies:
            rows = [r for k, rr in cells.items() if k[0] == "teleqna" and k[3] == pol and k[1] in bar_models for r in rr]  # the three pre-registered models, as the axis label states
            if not rows or pol == "unguarded":
                continue
            bins = defaultdict(list)
            for r in rows:
                bins[min(14, int(max(0.0, min(1.0, r["score"])) * 15))].append(r)
            xs = [sum(x["score"] for x in b) / len(b) for _, b in sorted(bins.items())]; ys = [sum(x["correct"] for x in b) / len(b) for _, b in sorted(bins.items())]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.0, ls=POL_STYLE[pol]["ls"], color=POL_COLOR[pol], label=POLICY_LABEL.get(pol, pol))
        ax.plot([0, 1], [0, 1], "k--", lw=1.0); ax.set_xlabel("mean score (TeleQnA, three models pooled)", fontsize=FS_LABEL); ax.set_ylabel("accuracy", fontsize=FS_LABEL); ax.grid(alpha=0.3); ax.tick_params(labelsize=FS_TICK); fig.tight_layout()
        ax.legend(fontsize=FS_LEGEND, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        savefig(fig, "reliability", "Reliability diagrams of each policy score on TeleQnA (all items, 15 bins).")
        # 5: H3 gain over the unguarded floor at coverage 0.8 (h3_gain_over_floor), one panel per task, policies on
        # the y axis, one marker/colour per bar model, horizontal 95% CI bars, dashed reference line at 3 points
        h3_by = {(h["dataset"], h["model"], h["policy"]): h for h in h3}
        h3_pols = [p for p in policies if p != "unguarded"]
        g_ds = [ds for ds in primary_datasets if any(k[0] == ds and k[1] in bar_models for k in h3_by)]
        g_ms = [m for m in bar_models if any(k[1] == m for k in h3_by)]
        if g_ds and g_ms:
            fig, axes = plt.subplots(1, len(g_ds), figsize=(7.0, 1.4), sharey=True, squeeze=False)
            gmark = {"qwen2.5-coder-7b-instruct": "o", "qwen3-coder-30b-a3b-instruct": "s", "llama-3.3-70b-instruct": "^"}
            off = {m: (j - (len(g_ms) - 1) / 2) * 0.22 for j, m in enumerate(g_ms)}
            for ax, ds in zip(axes[0], g_ds):
                for j, m in enumerate(g_ms):
                    pts = [(yi, h3_by[(ds, m, p)]) for yi, p in enumerate(h3_pols) if (ds, m, p) in h3_by]  # a cell that was not run is simply absent
                    if not pts:
                        continue
                    xs = [h["sel80_gain_points"] for _, h in pts]; ys = [yi + off[m] for yi, _ in pts]
                    err = [[h["sel80_gain_points"] - h["ci"][0] for _, h in pts], [h["ci"][1] - h["sel80_gain_points"] for _, h in pts]]
                    mc = MODEL_COLOR.get(m, f"C{j}")  # fixed per bar model, outside the policy palette
                    ax.errorbar(xs, ys, xerr=err, fmt=gmark.get(m, "o"), ms=3.5, color=mc, ecolor=mc, elinewidth=1.0, capsize=1.5, lw=1.0, label=SHORT_MODEL.get(m, MODEL_LABEL.get(m, m)))
                ax.axvline(3.0, color="k", ls="--", lw=1.0)
                ax.set_yticks(range(len(h3_pols))); ax.set_yticklabels([SHORT_POLICY.get(p, POLICY_LABEL.get(p, p)) for p in h3_pols]); ax.set_ylim(len(h3_pols) - 0.5, -0.5)
                ax.set_title(SHORT_DATASET.get(ds, ds), fontsize=FS_LABEL); ax.set_xlabel("Sel@0.8 gain over unguarded (points)", fontsize=FS_LABEL); ax.grid(alpha=0.3, axis="x"); ax.tick_params(labelsize=FS_TICK)
            fig.tight_layout()
            h_ = [Line2D([0], [0], marker=gmark.get(m, "o"), ms=3.5, color=MODEL_COLOR.get(m, f"C{j}"), lw=1.0) for j, m in enumerate(g_ms)]  # fixed mapping, every model
            l_ = [SHORT_MODEL.get(m, MODEL_LABEL.get(m, m)) for m in g_ms]
            fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(g_ms), fontsize=FS_LEGEND, frameon=False)
            savefig(fig, "h3_gain", "Selective-accuracy gain over the unguarded floor at coverage 0.8 (points, pooled seeds, clustered-bootstrap 95% CI) per policy and bar model; dashed line: the 3-point reference.")
    except Exception as exc:  # pragma: no cover
        print(f"figures skipped: {exc}", file=sys.stderr)
    # ---------------------------------------------------------------- numbers, manifest
    numbers["NumCells"] = len(per_cell); numbers["NumComparisons"] = len(comparisons); numbers["NumSeeds"] = len(seeds)
    numbers["NumMatrixCells"] = len([c for c in per_cell if c["dataset"] in primary_datasets and c["model"] in bar_models and c["method"] in policies])
    numbers["NumControlCells"] = numbers["NumCells"] - numbers["NumMatrixCells"]
    numbers["NumModels"] = len([m for m in models if m in bar_models]); numbers["NumModelsAll"] = len(models)
    numbers["NumItemsTeleQnA"] = max([c["n"] for c in per_cell if c["dataset"] == "teleqna"] or [0])
    numbers["NumItemsNetConfEval"] = max([c["n"] for c in per_cell if c["dataset"] == "netconfeval_t1"] or [0])
    numbers["BootstrapResamples"] = f"{B:,}"  # printed with a thousands separator (10,000), as elsewhere in the paper
    for wc in wrong_control:
        numbers[f"wrongAuroc{mkey(wc['model'])}{skey(wc['seed'])}"] = f"{wc['auroc_right_vs_wrong']:.3f}"
    for cb in combined:
        numbers[f"combinedAurc{mkey(cb['dataset'], cb['model'])}"] = f"{cb['aurc_combined']:.3f}"
    for t in transfer:
        numbers[f"tau{mkey(t['model'])}{skey(t['seed'])}"] = f"{t['kendall_tau']:.2f}"; numbers[f"tauLo{mkey(t['model'])}{skey(t['seed'])}"] = f"{t['ci95'][0]:.2f}"; numbers[f"tauHi{mkey(t['model'])}{skey(t['seed'])}"] = f"{t['ci95'][1]:.2f}"
    macros = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(numbers.items()))
    (OUT / "generated_numbers.tex").write_text(macros)
    achieved = {"datasets": len([d for d in datasets if d in primary_datasets]), "baselines": len([x for x in policies if x != METHOD]), "instances": numbers["NumItemsTeleQnA"] + numbers["NumItemsNetConfEval"], "seeds": len(seeds),
                "models_or_configs": len([m for m in models if m in bar_models]),
                "stat_tests": True, "effect_sizes": True, "independent_ground_truth": True, "figures": len(figures), "tables": len(tables)}
    for name, lines in sorted(c4_tables, key=lambda t: t[0] == "option_aware"):  # emitted after achieved_evidence (which counts only the tables above); option_aware last, so running the control appends and never reorders
        emit(name, lines)
    for name, lines in c6_tables:  # after the section M-P tables, so their order is unchanged
        emit(name, lines)
    # compact per-policy AURC table (pre-registered reporting; the AURC cells of tables/main.tex, mean over seeds,
    # same values and formatting), emitted last so achieved_evidence and the order of the other tables do not depend on it
    ap_pols = [p for p in policies if p != "unguarded"]
    AP_SHORT = {"logprob_gate": "logprob", "self_consistency": "SC5", "self_consistency_k1": "SC1", "confidence_gate": "verbal."}  # narrow headers for the single-column print table
    # the unguarded floor (accuracy and declined share, as in tables/floor.tex) is included as two columns
    lines = ["\\begin{tabular}{l" + "r" * len(ap_pols) + "rr}", "\\toprule", "Task, model & " + " & ".join(AP_SHORT.get(p, splabel(p)) for p in ap_pols) + " & Acc. & Decl. \\\\", "\\midrule"]
    for ds in primary_datasets:
        for m in bar_models:
            if any((ds, m, p) in agg for p in ap_pols):
                ug = agg.get((ds, m, "unguarded")); dv = mean(ug, "declined_rate") if ug else float("nan")
                floor_cols = (f"{100*mean(ug,'accuracy'):.1f}\\%" if ug else "--") + " & " + (f"{100*dv:.1f}\\%" if dv == dv else "--")
                lines.append(f"{sdlabel(ds)} {smlabel(m)} & " + " & ".join(f"{mean(agg[(ds, m, p)], 'aurc'):.3f}" if (ds, m, p) in agg else "--" for p in ap_pols) + f" & {floor_cols} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]; emit("aurc_by_policy", lines)
    tables[-1]["pre_registered_reporting"] = True; tables[-1]["note"] = "pre-registered reporting (per-policy AURC, compact form of tables/main.tex)"
    manifest = {"generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "source_registry_sha256": sha(REG), "analysis_script_sha256": sha(Path(__file__)),
                "achieved_evidence": achieved, "primary_h1": h1, "comparisons": comparisons, "per_cell": per_cell, "ordering_transfer": transfer, "wrong_answer_control": wrong_control, "combined_signal": combined,
                "tables": tables, "figures": figures, "numbers": numbers, "bootstrap_resamples": B, "policies_pre_registered": policies, "sensitivity_policies": sensitivity,
                "runner_sha256": current, "stale_runs_ignored": stale, "h2_excluded_models": h2_excluded, "bar_models": bar_models, "h3_gain_over_floor": h3, "seed_agreement": seed_agreement, "tie_handling": "exact-equality ties (pre-registered) ranked in expectation under uniform random tie order (AURC, Sel@c, AUROC); McNemar answered set breaks ties by item id; saturation_rate is a DISCLOSURE counting scores within 1e-6 of 0 or 1 (near-ties are ranked by their sub-1e-6 differences)"}
    manifest.update(post_hoc)  # POST-HOC keys, each flagged post_hoc: true with a one-line note
    json.dump(manifest, open(OUT / "results_manifest.json", "w"), indent=1)
    print(json.dumps({"cells": len(per_cell), "comparisons": len(comparisons), "h1": [(h["model"], h["outcome"]) for h in h1], "transfer": transfer[:3], "achieved": achieved}))


if __name__ == "__main__":
    main()
