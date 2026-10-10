#!/usr/bin/env python3
"""Server check (not a matrix cell): are the reported token probabilities raw, i.e. independent of the request temperature?

Usage: python3 probe_raw_probs.py --host http://127.0.0.1:1234 [--model qwen2.5-coder-7b-instruct] [--dataset teleqna|netconfeval_t1]
           [--n 150 (teleqna) | 75 (netconfeval_t1)] [--teleqna <TeleQnA.txt>] [--out probe_raw_probs.json]
       python3 probe_raw_probs.py --check probe_raw_probs.json     (no server call: apply the pass/fail rules to every stored probe)

The stored file holds a list of probes, one per (model, dataset); a run replaces the probe of the same (model, dataset)
or appends a new one, and leaves the others untouched.

Pass/fail (exit code 1 when any rule fails in any probe; the same rules apply to a live run and to --check):
  (i)   max |p(repeat T=0) - p(T)| over items, for T in 0.3, 0.7, 1.0, must not exceed TOLERANCE (0.005 absolute);
  (ii)  max |p(first T=0) - p(repeat T=0)| over items must not exceed TOLERANCE;
  (iii) at least 95% of items must have a parsable top-option probability at every temperature (all five requests).

The free signals (logprob_gate, margin_gate) read the first-pass answer token's probability from `top_logprobs`, and the
first pass is sampled at T=0.3. If the server reported temperature-scaled probabilities, the signals would depend on the
sampling temperature rather than on the model. This probe sends the SAME first-pass request as artifact/run_cell.py
(its prompt_for() and chat() are imported, not copied) for n items of the fixed 1,500-item TeleQnA frame at T = 0, 0.3,
0.7 and 1.0, with logprobs, top_logprobs=20 and one fixed seed per item (the first-pass seed of protocol seed 1,
request_seed(1, item, "first")). Items: every (1500 // n)-th item of the frame in frame order (deterministic, balanced
across the five categories).

Per item and temperature it records the probability of the top option token (the highest-probability option id among
the listed alternatives, tokenization variants aggregated as run_cell.chat does) and, separately, the sampled token and
its reported probability. It reports, for each T != 0, the maximum and mean absolute difference from T=0 of the top
option probability, plus the maximum over all request pairs. A fifth request per item repeats T=0 after the others
("0.0_repeat") as an order control: if it matches the later temperatures rather than the first T=0 request, a difference
is due to the request being the first for that prompt, not to the temperature. The number of items on which a T=0
request returned a token other than the top option is reported too (T=0 is not necessarily greedy on the server). Writes probe_raw_probs.json next to this file; the server
URL is not stored. No answer key is stored for any item.

Dataset netconfeval_t1: the prompt is run_cell.prompt_for() of the NetConfEval first pass (the official system prompt for the
policy types in PAPER_NCE_POLICIES, default reachability,waypoint,loadbalancing as in the registered runs), items are the
batch-size-1 and batch-size-2 items of run_cell.load_netconfeval_t1(450) (the registered frame), every k-th in frame order
(default 75 items). The probe concerns the probabilities the server reports, not full generations, so max_tokens is 8 and the
quantity compared is the probability of the FIRST generated token: "top_option" is the highest-probability token among the
listed alternatives of the first position (raw token strings, no aggregation) and "top_option_prob" its probability;
"sampled_token"/"sampled_token_prob" are the token actually generated first and its reported probability.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent
LANE = ENV_DIR.parent
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
# requests per item, in this order; "0.0_repeat" (T=0 again, sent last) separates a temperature effect from an effect of
# being the first request for a prompt (prompt-cache state)
REQUESTS = [(str(t), t) for t in TEMPERATURES] + [("0.0_repeat", 0.0)]
PROTOCOL_SEED = 1
DATASETS = ("teleqna", "netconfeval_t1")
DEFAULT_N = {"teleqna": 150, "netconfeval_t1": 75}
MIN_N = {"teleqna": 100, "netconfeval_t1": 60}
NCE_PROBE_BATCH_SIZES = ("batch1", "batch2")
NCE_POLICIES = "reachability,waypoint,loadbalancing"  # the policy types of the registered runs (manuscript/COMMANDS.md)
PROBE_MAX_TOKENS = 8
TOLERANCE = 0.005  # absolute difference in the top-option probability
MIN_PARSABLE_FRACTION = 0.95


def check_rows(rows: list[dict]) -> tuple[bool, list[str]]:
    """Apply the pass/fail rules to per-item records ({"by_temperature": {label: {"top_option_prob": ...}}})."""
    labels = [label for label, _ in REQUESTS]
    lines, ok = [], True

    def prob(r: dict, label: str):
        v = (r.get("by_temperature", {}).get(label) or {}).get("top_option_prob")
        return float(v) if isinstance(v, (int, float)) else None

    def max_diff(a: str, b: str) -> tuple[float | None, int]:
        d = [abs(x - y) for r in rows for x, y in [(prob(r, a), prob(r, b))] if x is not None and y is not None]
        return (max(d) if d else None), len(d)

    for b in ("0.3", "0.7", "1.0"):
        m, n = max_diff("0.0_repeat", b)
        good = m is not None and m <= TOLERANCE
        ok &= good
        lines.append(f"(i)   max |p(T=0 repeat) - p(T={b})| = {m if m is None else format(m, '.6f')} over {n} items  [{'PASS' if good else 'FAIL'}; tolerance {TOLERANCE}]")
    m, n = max_diff("0.0", "0.0_repeat")
    good = m is not None and m <= TOLERANCE
    ok &= good
    lines.append(f"(ii)  max |p(T=0 first) - p(T=0 repeat)| = {m if m is None else format(m, '.6f')} over {n} items  [{'PASS' if good else 'FAIL'}; tolerance {TOLERANCE}]")
    full = sum(1 for r in rows if all(prob(r, lb) is not None for lb in labels))
    frac = full / len(rows) if rows else 0.0
    good = frac >= MIN_PARSABLE_FRACTION
    ok &= good
    lines.append(f"(iii) items with a parsable top-option probability at every temperature = {full}/{len(rows)} ({frac:.3f})  [{'PASS' if good else 'FAIL'}; minimum {MIN_PARSABLE_FRACTION}]")
    lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
    return ok, lines


def top_option(info: dict, n_options: int) -> tuple[int | None, float | None]:
    alts = ((info.get("top") or {}).get("alts")) or {}
    opts = {int(k): p for k, p in alts.items() if k.isascii() and k.isdigit() and str(int(k)) == k and 1 <= int(k) <= n_options}  # canonical ASCII option ids only: str.isdigit() is true for '²', and '02' would overwrite option 2
    if not opts:
        return None, None
    best = max(opts, key=lambda k: opts[k])
    return best, opts[best]


def first_token(raw: dict) -> dict:
    """First generated position of a raw chat-completions response: the generated token and its probability, and the
    highest-probability listed alternative (raw token strings; equal-probability ties resolve to the first listed)."""
    content = (((raw.get("choices") or [{}])[0].get("logprobs") or {}).get("content")) or []
    if not content:
        return {"top_option": None, "top_option_prob": None, "sampled_token": None, "sampled_token_prob": None}
    tok = content[0]
    alts = [(str(a.get("token")), math.exp(a["logprob"])) for a in (tok.get("top_logprobs") or []) if a.get("logprob") is not None]
    best = max(alts, key=lambda kv: kv[1]) if alts else (None, None)
    return {"top_option": best[0], "top_option_prob": best[1], "sampled_token": str(tok.get("token")),
            "sampled_token_prob": math.exp(tok["logprob"]) if tok.get("logprob") is not None else None}


class _Capture:
    """Records the body of the last response run_cell.chat() received, so the first position of the logprobs can be read
    with the runner's own request code (chat() itself only keeps the first digit-bearing or YES/NO token)."""
    def __init__(self, run_cell):
        self.mod, self.body = run_cell, None

    def __enter__(self):
        self.orig = self.mod.urllib.request.urlopen
        outer = self

        def urlopen(*a, **k):
            resp = outer.orig(*a, **k)
            data = resp.read()
            outer.body = json.loads(data.decode())
            import io
            return io.BytesIO(data)
        self.mod.urllib.request.urlopen = urlopen
        return self

    def __exit__(self, *exc):
        self.mod.urllib.request.urlopen = self.orig


def server_model_info(host: str, model: str) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{host}/api/v0/models", timeout=30) as r:
            data = json.loads(r.read().decode()).get("data") or []
    except Exception as exc:  # not every server has this LM Studio endpoint
        return {"error": type(exc).__name__}
    m = next((x for x in data if x.get("id") == model), {})
    return {k: m.get(k) for k in ("id", "arch", "quantization", "compatibility_type", "state", "max_context_length") if k in m}


def probe_items(run_cell, dataset: str, n: int, teleqna_path: Path) -> tuple[list[dict], str]:
    if dataset == "teleqna":
        frame = run_cell.load_teleqna(teleqna_path, 1500, None)
        stride = max(1, len(frame) // n)
        return frame[::stride][:n], f"every {stride}th item of run_cell.load_teleqna(TeleQnA.txt, 1500) in frame order"
    os.environ.setdefault("PAPER_NCE_POLICIES", NCE_POLICIES)
    frame = [it for it in run_cell.load_netconfeval_t1(450) if it["category"] in NCE_PROBE_BATCH_SIZES]
    stride = max(1, len(frame) // n)
    return frame[::stride][:n], (f"every {stride}th of the {len(frame)} batch-size-1 and batch-size-2 items of run_cell.load_netconfeval_t1(450) "
                                  f"(PAPER_NCE_POLICIES={os.environ['PAPER_NCE_POLICIES']}) in frame order")


def run_probe(args, run_cell, artifact_sha256) -> dict:
    dataset = args.dataset
    n = args.n or DEFAULT_N[dataset]
    items, selection = probe_items(run_cell, dataset, n, Path(args.teleqna))
    nce = dataset == "netconfeval_t1"
    rows = []
    for i, item in enumerate(items, start=1):
        seed = run_cell.request_seed(PROTOCOL_SEED, item["id"], "first")
        row = {"id": item["id"], "category": item["category"], "request_seed": seed, "by_temperature": {}}  # no answer key is stored
        for label, t in REQUESTS:
            if nce:
                with _Capture(run_cell) as cap:
                    text, info = run_cell.chat(args.model, run_cell.prompt_for(item), temperature=t, seed=seed, max_tokens=PROBE_MAX_TOKENS, logprobs=True)
                rec = first_token(cap.body or {})
                rec["generated_text"] = text
            else:
                text, info = run_cell.chat(args.model, run_cell.prompt_for(item), temperature=t, seed=seed, max_tokens=PROBE_MAX_TOKENS, logprobs=True)
                opt, p = top_option(info, len(item["options"]))
                top = info.get("top") or {}
                rec = {"top_option": opt, "top_option_prob": p, "sampled_token": top.get("token"), "sampled_token_prob": top.get("prob"),
                       "parsed_answer": run_cell.parse_option(text, len(item["options"]))}
            row["by_temperature"][label] = rec
        rows.append(row)
        if i % 25 == 0:
            print(f"{args.model} {dataset}: {i}/{len(items)} items", flush=True)

    def probs(label: str) -> list[float | None]:
        return [r["by_temperature"][label]["top_option_prob"] for r in rows]

    summary: dict = {}
    for label, _ in REQUESTS:
        vals = [p for p in probs(label) if p is not None]
        sampled_is_top = sum(1 for r in rows if str(r["by_temperature"][label]["sampled_token"]) == str(r["by_temperature"][label]["top_option"]))
        summary[label] = {"n_with_option_probability": len(vals), "mean_top_option_prob": round(statistics.fmean(vals), 6) if vals else None,
                           "sampled_token_is_top_option": sampled_is_top}
    diffs: dict = {}
    labels = [label for label, _ in REQUESTS]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            d = [abs(x - y) for x, y in zip(probs(a), probs(b)) if x is not None and y is not None]
            same_top = sum(1 for r in rows if r["by_temperature"][a]["top_option"] == r["by_temperature"][b]["top_option"])
            diffs[f"{a}_vs_{b}"] = {"n_pairs": len(d), "max_abs_diff": max(d) if d else None, "mean_abs_diff": statistics.fmean(d) if d else None,
                                    "n_exactly_equal": sum(1 for x in d if x == 0.0), "same_top_option": same_top}
    vs_t0 = {k: v for k, v in diffs.items() if k.startswith("0.0_vs_") and "repeat" not in k}
    request = {"prompt": "run_cell.prompt_for (first-pass prompt)", "max_tokens": PROBE_MAX_TOKENS, "logprobs": True, "top_logprobs": 20, "reasoning_effort": "none",
              "seed": f"request_seed({PROTOCOL_SEED}, item_id, 'first') per item, identical across temperatures", "temperatures": list(TEMPERATURES),
              "request_order_per_item": [label for label, _ in REQUESTS]}
    if nce:
        request["compared_quantity"] = "probability of the first generated position: top_option = highest-probability listed alternative (raw token), sampled_token = generated token"
    return {
        "probe": "raw-probability check: top option-token probability vs request temperature (server check, not a matrix cell)",
        "dataset": dataset,
        "date": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "model": args.model, "server_model_info": server_model_info(args.host.rstrip("/"), args.model),
        "request": request,
        "items": {"n": len(rows), "selection": selection,
                  "by_category": {c: sum(1 for r in rows if r["category"] == c) for c in sorted({r["category"] for r in rows})}},
        "runner_sha256": artifact_sha256(runner_only=True),
        "per_temperature": summary,
        "differences": diffs,
        "headline": {"max_abs_diff_vs_T0": max((v["max_abs_diff"] for v in vs_t0.values() if v["max_abs_diff"] is not None), default=None),
                     "mean_abs_diff_vs_T0": {k: v["mean_abs_diff"] for k, v in vs_t0.items()},
                     "max_abs_diff_any_pair": max((v["max_abs_diff"] for v in diffs.values() if v["max_abs_diff"] is not None), default=None),
                     "t0_non_top_option_token": {label: len(rows) - summary[label]["sampled_token_is_top_option"] for label in ("0.0", "0.0_repeat")}},
        "per_item": rows,
    }


def stored_probes(doc: dict) -> list[dict]:
    """The probes of a stored file: the list under "probes", or (an older file) the single probe the file itself holds."""
    return doc["probes"] if isinstance(doc.get("probes"), list) else [doc]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("PAPER_LMSTUDIO_HOST", "http://127.0.0.1:1234"))
    parser.add_argument("--model", default="qwen2.5-coder-7b-instruct")
    parser.add_argument("--dataset", choices=DATASETS, default="teleqna")
    parser.add_argument("--n", type=int, default=None, help="items (default 150 for teleqna, at least 100; 75 for netconfeval_t1, at least 60)")
    parser.add_argument("--teleqna", default=os.environ.get("TELEQNA_PATH", str(LANE.parents[1] / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt")))
    parser.add_argument("--out", type=Path, default=ENV_DIR / "probe_raw_probs.json")
    parser.add_argument("--check", type=Path, help="apply the pass/fail rules to every probe of a stored output; no server call, nothing written")
    args = parser.parse_args(argv)
    if args.check:
        stored = json.loads(args.check.read_text(encoding="utf-8"))
        probes = stored_probes(stored)
        all_ok = True
        print(f"check of {args.check} (tolerance {TOLERANCE}; {len(probes)} probe(s))")
        for p in probes:
            ok, lines = check_rows(p.get("per_item") or [])
            all_ok &= ok
            print(f"\n== model {p.get('model')}, dataset {p.get('dataset', 'teleqna')}, {len(p.get('per_item') or [])} items")
            print("\n".join(lines))
        print("\nOVERALL: " + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1
    if args.n is not None and args.n < MIN_N[args.dataset]:
        parser.error(f"--n must be at least {MIN_N[args.dataset]} for {args.dataset}")
    os.environ["PAPER_LMSTUDIO_HOST"] = args.host.rstrip("/")  # run_cell reads its host at import time
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(LANE / "artifact"))
    sys.path.insert(0, str(ENV_DIR))
    from reproduce_matrix import artifact_sha256  # noqa: E402
    import run_cell  # noqa: E402  (the runner's own prompt and request code)

    report = run_probe(args, run_cell, artifact_sha256)
    diffs, rows = report["differences"], report["per_item"]
    existing = stored_probes(json.loads(args.out.read_text(encoding="utf-8"))) if args.out.exists() else []
    key = (report["model"], report["dataset"])
    slot = next((i for i, p in enumerate(existing) if (p.get("model"), p.get("dataset", "teleqna")) == key), None)
    if slot is None:
        existing.append(report)
    else:
        existing[slot] = report
    args.out.write_text(json.dumps({"probes": existing}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(rows), "differences": {k: {kk: v[kk] for kk in ("max_abs_diff", "mean_abs_diff", "n_exactly_equal", "same_top_option")} for k, v in diffs.items()}}, indent=1))
    print(f"wrote {args.out} ({len(existing)} probe(s); {'replaced' if slot is not None else 'appended'} {key})")
    ok, lines = check_rows(rows)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
