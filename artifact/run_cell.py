#!/usr/bin/env python3
"""One experiment cell for the VCC 2027 lane: selective prediction of small local LLMs.

Design (after two rounds of pre-mortems):
- ONE first-pass answer per (dataset, model, seed, item), sampled once at T=0.3 and cached under
  experiments/cache/, so every abstention policy scores the SAME answer; policies differ only in the confidence
  signal and its cost. This removes the decoding-distribution confound.
- Sampling scheme v2: every request carries its own seed request_seed(seed, item_id, role, j) =
  sha256("seed|item|role|j") mod (2^31 - 1). The server re-seeds its sampler on every request, so passing the raw
  protocol seed would apply ONE sampling quantile to every item (seeds would not be independent per-item draws and the
  self-consistency samples would collapse to shared draws). Per-item request seeds make each draw independent across
  items while staying reproducible from (seed, item, role, j).
- Every policy exposes a CONTINUOUS score (so risk-coverage curves and matched-coverage points are comparable):
    unguarded          score = 1 (coverage 1.0, the floor)
    logprob_gate       score = probability of the first answer token (Hendrycks & Gimpel 2017 max-softmax analogue)
    margin_gate        score = top-1 minus top-2 probability among option tokens
    self_consistency   score = agreement of k=5 extra samples (T=0.7) with the first-pass answer (Wang et al. 2023)
    self_consistency_k1 score = agreement of ONE extra sample (T=0.7) with the first-pass answer: the cost-matched control (one extra pass, like the verifier)
    confidence_gate    score = verbalized confidence 0-1 elicited for the first-pass answer (Tian et al. 2023 style)
    verifier_gated     score = sigmoid(log P(YES) - log P(NO)) of an independent verification pass on the first-pass answer (Kadavath et al.
                         2022 P(True) self-evaluation); YES/NO read from top-20 alternatives with tokenization variants aggregated; when NO is
                         absent from the top-20 list its probability is bounded by the smallest listed alternative (disclosed: scores can
                         saturate, so the analysis uses tie-aware AURC)
- Ground truth: TeleQnA official exact-match on option id (evaluation_tools.py semantics); NetConfEval task 1 official
  compare_result on one requirement batch (the abstention unit), strict: correct iff nothing missing and nothing wrong.
Outputs: <out>/result.json (metric_values, evaluator_id, n) and <out>/predictions.jsonl with per-item score, answered,
correct, category, cost. All aggregate numbers are recomputed by the analysis script from predictions.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from pathlib import Path

HOST = os.environ.get("PAPER_LMSTUDIO_HOST", "http://127.0.0.1:1234").rstrip("/")
FRAME_IDS: set = set()
LANE = Path(__file__).resolve().parents[1]
NETCONFEVAL = Path(os.environ.get("NETCONFEVAL_PATH", str(LANE.parents[1] / "runtime" / "ground_truth" / "NetConfEval")))
FIRST_PASS_TEMPERATURE = 0.3
SAMPLE_TEMPERATURE = 0.7  # self-consistency samples
VERIFIER_TEMPERATURE = 0.0  # verifier passes (all verifier variants)
CONFIDENCE_TEMPERATURE = 0.0  # verbalized-confidence pass
SAMPLING_SCHEME = "per-item request seeds: sha256(seed|item|role|j)"
CACHE_SCHEME = "v2"  # first-pass cache tag: caches written under the old shared-seed scheme are never served


def request_seed(seed: int, item_id: str, role: str, j: int = 0) -> int:
    """Per-request sampler seed. The server re-seeds on every request, so the raw protocol seed would draw the SAME
    sampling quantile for every item; hashing (seed, item, role, j) gives independent draws per item and per role while
    staying reproducible. Self-consistency sample j depends only on j (not on k), so the k=1 control is sample 0 of k=5."""
    return int(hashlib.sha256(f"{seed}|{item_id}|{role}|{j}".encode()).hexdigest()[:8], 16) % 2147483647


_MODEL_READY: set = set()


def ensure_model(model: str) -> None:
    """LM Studio JIT-loads a model on first use but refuses when resident models leave too little memory (the 70B needs
    ~43 GB). When `lms` is available on this host (i.e. the cell runs on the Studio) and the model is not loaded yet,
    unload everything else first. Through a tunnel (no `lms`), this is a no-op and the operator unloads by hand."""
    if model in _MODEL_READY:
        return
    _MODEL_READY.add(model)
    try:
        with urllib.request.urlopen(f"{HOST}/api/v0/models", timeout=30) as r:
            data = json.loads(r.read().decode()).get("data") or []
    except Exception:
        return
    loaded = [m.get("id") for m in data if m.get("state") == "loaded"]
    if model in loaded:
        return
    import shutil, subprocess
    lms = shutil.which("lms") or (str(Path.home() / ".lmstudio" / "bin" / "lms") if (Path.home() / ".lmstudio" / "bin" / "lms").exists() else None)
    if loaded and lms:
        subprocess.run([lms, "unload", "--all"], capture_output=True, text=True, timeout=120)


def chat(model: str, messages: list[dict], *, temperature: float, seed: int, max_tokens: int = 64, logprobs: bool = False) -> tuple[str, dict]:
    ensure_model(model)
    # reasoning_effort "none": thinking-capable models (gemma-4) otherwise spend the whole token budget in a hidden
    # reasoning channel and return empty content; the other models accept and ignore the field (checked on the Studio).
    payload = {"model": model, "messages": messages, "temperature": temperature, "seed": seed, "max_tokens": max_tokens, "stream": False, "reasoning_effort": "none"}
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    req = urllib.request.Request(f"{HOST}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode())
    choice = d["choices"][0]
    text = choice["message"]["content"]
    usage = d.get("usage") or {}
    info = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "seconds": round(time.time() - t0, 3), "top": None}
    lp = (choice.get("logprobs") or {}).get("content") or []
    lps = [t["logprob"] for t in lp if t.get("logprob") is not None]
    if lps:
        info["seq_mean_logprob"] = sum(lps) / len(lps)
        info["seq_min_logprob"] = min(lps)
    for tok in lp:
        token = str(tok.get("token", ""))
        if any(ch.isdigit() for ch in token) or token.strip().upper() in {"YES", "NO", "Y", "N"}:
            alts: dict = {}
            for a in (tok.get("top_logprobs") or []):
                if a.get("logprob") is None:
                    continue
                key = str(a.get("token", "")).strip().strip("=_")
                alts[key] = alts.get(key, 0.0) + math.exp(a["logprob"])  # aggregate tokenization variants (" YES", "YES", "=YES")
            listed = [a["logprob"] for a in (tok.get("top_logprobs") or []) if a.get("logprob") is not None]
            info["top"] = {"token": token.strip(), "prob": math.exp(tok["logprob"]) if tok.get("logprob") is not None else None, "alts": alts, "min_listed_prob": math.exp(min(listed)) if listed else None}
            break
    return text, info


# ----------------------------------------------------------------------------- datasets
def load_teleqna(path: Path, limit: int, category: str | None) -> list[dict]:
    data = json.load(open(path))
    items = []
    for key, q in data.items():
        if category and q.get("category") != category:
            continue
        options = {k: v for k, v in q.items() if k.startswith("option ")}
        gold = re.match(r"option (\d+):", str(q.get("answer", "")))
        if gold:
            items.append({"id": key, "question": q["question"], "options": options, "gold": int(gold.group(1)), "category": q.get("category")})
    # stratified fixed frame: equal share per category, deterministic (sampling frame seed 1000), identical across seeds/models/policies
    by_cat = collections.defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    rng = random.Random(1000)
    per = max(1, limit // max(1, len(by_cat)))
    chosen = []
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        rng.shuffle(rows)
        chosen.extend(rows[:per])
    return chosen[:limit]


def load_netconfeval_t1(limit: int, batch_sizes=(1, 2, 5, 10)) -> list[dict]:
    import sys
    sys.path.insert(0, str(NETCONFEVAL))
    from sortedcontainers import SortedSet
    from netconfeval.common.utils import load_csv, pick_sample, transform_sample_to_expected, convert_to_human_language, chunk_list
    # policy types exactly as the official script selects them (reachability always; waypoint; loadbalancing needs waypoint)
    policy_types = SortedSet([x.strip() for x in os.environ.get("PAPER_NCE_POLICIES", "reachability").split(",") if x.strip()])
    if "loadbalancing" in policy_types:
        from netconfeval.prompts.step_1_reachability_waypoint_load import SETUP_PROMPT, FUNCTION_PROMPT, ASK_FOR_RESULT_PROMPT
    elif "waypoint" in policy_types:
        from netconfeval.prompts.step_1_reachability_waypoint import SETUP_PROMPT, FUNCTION_PROMPT, ASK_FOR_RESULT_PROMPT
    else:
        from netconfeval.prompts.step_1_reachability import SETUP_PROMPT, FUNCTION_PROMPT, ASK_FOR_RESULT_PROMPT
    dataset = load_csv(str(NETCONFEVAL / "assets" / "step_1_policies.csv"), policy_types)
    items = []
    random.seed(1000)  # convert_to_human_language draws phrasings from this RNG -> fixed frame
    for it in range(0, 25):
        samples = pick_sample(max(batch_sizes), dataset, it, policy_types)
        for b in batch_sizes:
            for ci, chunk in enumerate(chunk_list(samples, b)):
                items.append({"id": f"nce-it{it}-b{b}-c{ci}", "category": f"batch{b}", "expected": transform_sample_to_expected(chunk), "human": " ".join(convert_to_human_language(chunk)), "n_req": len(chunk), "system": render_template(f"{SETUP_PROMPT}\n{FUNCTION_PROMPT}\n{ASK_FOR_RESULT_PROMPT}")})
                if len(items) >= limit:
                    return items
    return items


def render_template(t: str) -> str:
    """The official prompts are LangChain ChatPromptTemplate strings: '{{' and '}}' render to single braces. Sending them
    raw makes some models copy the doubled braces (the 70B smoke produced {{"status": ...}} on 12/12 batches)."""
    return t.replace("{{", "{").replace("}}", "}")


def netconfeval_score(item: dict, text: str) -> tuple[bool, dict]:
    import sys
    sys.path.insert(0, str(NETCONFEVAL))
    from netconfeval.common.utils import compare_result
    raw = text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    try:
        out = json.loads(raw[start:end + 1]) if start != -1 else {}
    except json.JSONDecodeError:
        return False, {"format_error": True, "declined": False}
    if str(out.get("status", "")).lower() == "error":
        return False, {"declined": True}
    result = out.get("result", out)
    if not isinstance(result, dict):
        return False, {"format_error": True, "declined": False}
    # structure failure: the JSON parses but carries none of the top-level policy keys the official checker looks up
    # (e.g. the 7B emits {"result": {"router": [...]}} without the "reachability" wrapper on 12/12 smoke batches).
    # Official semantics count it as fail (wrong); we additionally tag it as a format failure for the pre-registered rule.
    if not any(k in result for k in item["expected"].keys()):
        return False, {"format_error": True, "declined": False, "structure_failure": True}
    row = {"total": 0, "success": 0, "fail": 0, "wrong": 0, "accuracy": 0}
    try:
        compare_result(item["expected"], result, row)
    except Exception as exc:
        return False, {"format_error": True, "declined": False, "error": str(exc)[:80]}
    return (row.get("fail", 0) == 0 and row.get("wrong", 0) == 0 and row.get("total", 0) > 0), {"official": row, "declined": False}


# NetConfEval generation cap: the official harness allows 4096 completion tokens; batch-size-10 specifications are the longest
# (about 10 reachability entries). 2048 keeps every batch inside the 8192-token LM Studio context together with the prompt.
NCE_MAX_TOKENS = 2048


def spec_key(text: str) -> str | None:
    """Canonical, label-free key of the parsed specification, all policy types (for agreement between samples and as the verifier input)."""
    raw = text.strip(); start, end = raw.find("{"), raw.rfind("}")
    if start == -1:
        return None
    try:
        out = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
    result = out.get("result", out) if isinstance(out, dict) else None
    if not isinstance(result, dict):
        return None
    # canonical form of the WHOLE result: every policy key the official checker reads (reachability, waypoint, loadbalancing);
    # a bare reachability map (no wrapper) is normalised the same way so agreement is comparable across output styles
    if not any(k in result for k in ("reachability", "waypoint", "loadbalancing")):
        result = {"reachability": result}

    def canon(v):
        if isinstance(v, dict):
            return {str(k): canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, list):
            # scalars stay plain strings (single-encoded); nested containers are serialised once so the list can be sorted
            return sorted(json.dumps(canon(x), sort_keys=True) if isinstance(x, (dict, list)) else str(x) for x in v)
        return str(v)
    return json.dumps(canon(result), sort_keys=True)


_FEWSHOT: list[dict] | None = None


def fewshot_examples(data_path: Path, frame_ids: set) -> list[dict]:
    """Three fixed TeleQnA exemplars outside the evaluation frame: one correct proposal (YES) and two wrong (NO)."""
    global _FEWSHOT
    if _FEWSHOT is not None:
        return _FEWSHOT
    data = json.load(open(data_path))
    pool = [k for k in sorted(data) if k not in frame_ids]
    rng = random.Random(4242)
    rng.shuffle(pool)
    out = []
    for key in pool:
        q = data[key]
        gold = re.match(r"option (\d+):", str(q.get("answer", "")))
        options = {k: v for k, v in q.items() if k.startswith("option ")}
        if not gold or len(options) < 3:
            continue
        g = int(gold.group(1))
        if len(out) == 0:
            cand, label = g, "YES"
        else:
            cand = next(int(k.split()[1]) for k in options if int(k.split()[1]) != g); label = "NO"
        out.append({"question": q["question"], "cand": cand, "text": options[f"option {cand}"], "label": label})
        if len(out) == 3:
            break
    _FEWSHOT = out
    return out


def is_nce(item: dict) -> bool:
    return "expected" in item


def prompt_for(item: dict) -> list[dict]:
    if is_nce(item):
        return [{"role": "system", "content": item["system"]}, {"role": "user", "content": "Here are my requirements: " + item["human"]}]
    opts = "\n".join(f"{k}: {v}" for k, v in sorted(item["options"].items(), key=lambda kv: int(kv[0].split()[1])))
    return [{"role": "system", "content": "You are a telecommunications expert. Answer multiple-choice questions with only the option number, e.g. '2'."},
            {"role": "user", "content": f"{item['question']}\n{opts}\nAnswer with only the option number."}]


def parse_option(text: str, n_options: int) -> int | None:
    m = re.search(r"option\s*(\d+)|\b(\d+)\b", text.strip().lower())
    if not m:
        return None
    val = int(m.group(1) or m.group(2))
    return val if 1 <= val <= n_options else None


def load_teleqna_full(path: Path) -> list[dict]:
    data = json.load(open(path))
    items = []
    for key, q in data.items():
        options = {k: v for k, v in q.items() if k.startswith("option ")}
        gold = re.match(r"option (\d+):", str(q.get("answer", "")))
        if gold:
            items.append({"id": key, "question": q["question"], "options": options, "gold": int(gold.group(1)), "category": q.get("category")})
    return items


# ----------------------------------------------------------------------------- first pass (shared across policies)
def refresh_cache(cache: dict, cache_path: Path, state: dict) -> None:
    """Pick up first-pass records appended by a concurrent cell (same dataset/model/seed) since the last read."""
    if not cache_path.exists():
        return
    with cache_path.open() as f:
        f.seek(state.get("offset", 0))
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "spec_key" in rec:
                # the canonical specification is DERIVED from the cached text with the current spec_key(): a cache written by
                # an earlier runner (reachability-only projection) must never feed an old key to the signals
                rec["spec_key"] = spec_key(rec.get("text") or "")
            cache.setdefault(rec["id"], rec)  # first writer wins; all cells then score the same answer
        state["offset"] = f.tell()


def first_pass(item: dict, model: str, seed: int, cache: dict, cache_path: Path, cost: collections.Counter, state: dict | None = None) -> dict:
    """One sampled answer per item, with logprob info, cached so every policy scores the same answer."""
    state = state if state is not None else {}
    if item["id"] not in cache:
        refresh_cache(cache, cache_path, state)
    if item["id"] in cache:
        return cache[item["id"]]
    max_tok = NCE_MAX_TOKENS if is_nce(item) else 8
    text, u = chat(model, prompt_for(item), temperature=FIRST_PASS_TEMPERATURE, seed=request_seed(seed, item["id"], "first"), max_tokens=max_tok, logprobs=True)
    cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
    if is_nce(item):
        ok, info = netconfeval_score(item, text)
        rec = {"text": text[:8000], "pred": None, "correct": bool(ok), "declined": bool(info.get("declined")), "top": u["top"], "official": info.get("official"), "format_error": bool(info.get("format_error")),
               "seq_mean_logprob": u.get("seq_mean_logprob"), "seq_min_logprob": u.get("seq_min_logprob"), "spec_key": spec_key(text)}
    else:
        pred = parse_option(text, len(item["options"]))
        rec = {"text": text[:100], "pred": pred, "correct": pred == item["gold"], "declined": pred is None, "top": u["top"], "seq_mean_logprob": u.get("seq_mean_logprob"), "seq_min_logprob": u.get("seq_min_logprob")}
    rec["first_pass_cost"] = {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"], "seconds": u["seconds"]}
    cache[item["id"]] = rec
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({"id": item["id"], **rec}) + "\n")
    return rec


# ----------------------------------------------------------------------------- policies (signal on the shared answer)
def signal(method: str, item: dict, fp: dict, model: str, seed: int, cost: collections.Counter, args) -> tuple[float, dict]:
    """Returns (score in [0,1], extra). Extra passes are charged to `cost` (the policy's own cost)."""
    if method == "unguarded":
        return 1.0, {}
    if method == "logprob_gate":
        if is_nce(item):  # sequence-level confidence of the generated JSON (mean token probability)
            m = fp.get("seq_mean_logprob")
            return (math.exp(m) if m is not None else 0.0), {"signal": "seq_mean_prob"}
        top = fp.get("top") or {}
        return float(top.get("prob") or 0.0), {}
    if method == "margin_gate":
        if is_nce(item):  # no option margin exists for free-form JSON; use the minimum token probability (weakest link)
            m = fp.get("seq_min_logprob")
            return (math.exp(m) if m is not None else 0.0), {"signal": "seq_min_prob"}
        top = fp.get("top") or {}
        alts = top.get("alts") or {}
        probs = sorted((p for t, p in alts.items() if t and (t.isdigit() or t.upper() in {"YES", "NO"})), reverse=True)
        if not probs:
            return float(top.get("prob") or 0.0), {}
        return float(probs[0] - (probs[1] if len(probs) > 1 else 0.0)), {"top2": probs[:2]}
    if method in ("self_consistency", "self_consistency_k1"):
        k = 5 if method == "self_consistency" else 1  # k=1 is the cost-matched control: exactly one extra pass, like the verifier
        agree = 0
        samples: list = []  # the sampled answers themselves, stored for audit (independence of draws is checkable)
        for j in range(k):
            # sample j's request seed depends on (seed, item, j) only, so the k=1 sample equals sample 0 of k=5
            text, u = chat(model, prompt_for(item), temperature=SAMPLE_TEMPERATURE, seed=request_seed(seed, item["id"], "sc", j), max_tokens=NCE_MAX_TOKENS if is_nce(item) else 8)
            cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
            if is_nce(item):
                sk = spec_key(text)
                agree += int(sk is not None and sk == fp.get("spec_key"))  # label-free: same parsed specification
                samples.append(sk)
            else:
                sp = parse_option(text, len(item["options"]))
                agree += int(sp == fp["pred"])
                samples.append(sp)
        n_distinct = len({json.dumps(s) for s in samples})
        stored = [s[:300] if isinstance(s, str) else s for s in samples]
        return agree / k, {"k": k, "agree": agree, "samples": stored, "n_distinct_samples": n_distinct}
    if method == "confidence_gate":
        if is_nce(item):
            q = f"Requirements: {item['human']}\nProposed specification: {fp.get('spec_key') or fp['text'][:4000]}\nOn a scale of 0-100, how confident are you that this specification translates every requirement exactly (reachability, waypoint and load-balancing entries alike: nothing missing, nothing extra, nothing wrong)? Reply with only the number."
        else:
            chosen = item["options"].get(f"option {fp['pred']}", "") if fp.get("pred") else "(no answer)"
            q = f"{item['question']}\nProposed answer: option {fp.get('pred')}: {chosen}\nOn a scale of 0-100, how confident are you that this answer is correct? Reply with only the number."
        text, u = chat(model, [{"role": "system", "content": "Reply with only an integer between 0 and 100."}, {"role": "user", "content": q}], temperature=CONFIDENCE_TEMPERATURE, seed=request_seed(seed, item["id"], "confidence"), max_tokens=6)
        cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
        m = re.search(r"\d+", text)
        return (min(100, int(m.group(0))) / 100 if m else 0.0), {"raw": text.strip()[:20]}
    if method in ("verifier_gated_pB", "verifier_wrong_control", "verifier_gated_fs"):
        # pB: neutral verifier wording (prompt-sensitivity control). wrong_control: verify a deliberately WRONG option;
        # the verifier should assign LOW P(YES) (error-detector AUROC), which separates "detects errors" from "rates difficulty".
        if is_nce(item):
            return 1.0, {"na": True}
        if method == "verifier_wrong_control":
            # deliberately wrong option: never the gold and never the model's own first-pass prediction (otherwise the
            # control coincides with the verifier's real input on items the model got wrong and measures nothing)
            wrong = [int(k.split()[1]) for k in item["options"] if int(k.split()[1]) not in (item["gold"], fp.get("pred"))]
            if not wrong:
                return 1.0, {"na": True}
            cand = wrong[(seed + len(item["id"])) % len(wrong)]
            chosen = item["options"].get(f"option {cand}", "")
            sysmsg = "You are a strict telecommunications standards reviewer. Reply with exactly YES or NO."
        elif method == "verifier_gated_fs":
            cand = fp.get("pred"); chosen = item["options"].get(f"option {cand}", "") if cand else "(no answer)"
            sysmsg = "You are a strict telecommunications standards reviewer. Reply with exactly YES or NO."
        else:
            cand = fp.get("pred"); chosen = item["options"].get(f"option {cand}", "") if cand else "(no answer)"
            sysmsg = "Reply with exactly YES or NO."
        messages = [{"role": "system", "content": sysmsg}]
        if method == "verifier_gated_fs":  # few-shot elicitation (Kadavath et al. 2022 report this matters for small models)
            for ex in fewshot_examples(Path(args.data), FRAME_IDS):
                messages.append({"role": "user", "content": f"Question: {ex['question']}\nProposed answer: option {ex['cand']}: {ex['text']}\nIs the proposed answer correct? Reply YES or NO only."})
                messages.append({"role": "assistant", "content": ex["label"]})
        q = f"Question: {item['question']}\nProposed answer: option {cand}: {chosen}\nIs the proposed answer correct? Reply YES or NO only."
        messages.append({"role": "user", "content": q})
        vrole = {"verifier_gated_pB": "verifier_pB", "verifier_gated_fs": "verifier_fs", "verifier_wrong_control": "verifier_wrong"}[method]
        text, u = chat(model, messages, temperature=VERIFIER_TEMPERATURE, seed=request_seed(seed, item["id"], vrole), max_tokens=3, logprobs=True)
        cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
        p_yes, p_no, norm = p_yes_from(u)
        return norm, {"verifier_text": text.strip()[:8], "control_option": cand, "p_yes_raw": p_yes, "p_no_raw": p_no, "p_no_listed": p_no > 0}
    if method in ("verifier_gated", "verifier_gated_x"):
        # verifier_gated_x: the verification pass is answered by a DIFFERENT model than the one that produced the answer
        # (independent-verifier control; pre-registered sensitivity cell). The answer model's first pass is cached, so
        # only the verifier model is queried here.
        vmodel = XMODEL.get(model, model) if method == "verifier_gated_x" else model
        if is_nce(item):
            q = f"Requirements: {item['human']}\nProposed specification JSON: {fp.get('spec_key') or fp['text'][:4000]}\nDoes the specification translate every requirement exactly (reachability, waypoint and load-balancing entries alike: nothing missing, nothing extra, nothing wrong)? Reply YES or NO only."
            sysmsg = "You are a strict network engineer reviewing a formal specification. Reply with exactly YES or NO."
        else:
            chosen = item["options"].get(f"option {fp['pred']}", "") if fp.get("pred") else "(no answer)"
            q = f"Question: {item['question']}\nProposed answer: option {fp.get('pred')}: {chosen}\nIs the proposed answer correct? Reply YES or NO only."
            sysmsg = "You are a strict telecommunications standards reviewer. Reply with exactly YES or NO."
        vrole = "verifier_x" if method == "verifier_gated_x" else "verifier"
        text, u = chat(vmodel, [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], temperature=VERIFIER_TEMPERATURE, seed=request_seed(seed, item["id"], vrole), max_tokens=3, logprobs=True)
        cost.update({"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"]}); cost["seconds"] += u["seconds"]
        p_yes, p_no, norm = p_yes_from(u)
        return norm, {"verifier_text": text.strip()[:8], "p_yes_raw": p_yes, "p_no_raw": p_no, "p_no_listed": p_no > 0, "verifier_model": vmodel}
    raise SystemExit(f"unknown method {method}")


NCE_THRESHOLDS = {"logprob_gate": 0.9, "margin_gate": 0.05}


def p_yes_from(u: dict) -> tuple[float, float, float]:
    """Returns (p_yes_raw, p_no_raw, p_yes_normalized = yes/(yes+no)) aggregating tokenization variants."""
    top = u.get("top") or {}
    alts: dict = {}
    for k, v in (top.get("alts") or {}).items():
        alts[k.upper()] = alts.get(k.upper(), 0.0) + v
    p_yes = alts.get("YES", alts.get("Y"))
    p_no = alts.get("NO", alts.get("N"))
    if p_yes is None:
        p_yes = float(top.get("prob") or 0.0) if str(top.get("token", "")).upper().startswith("Y") else 0.0
    if p_no is None:
        p_no = float(top.get("prob") or 0.0) if str(top.get("token", "")).upper().startswith("N") else 0.0
    # log-odds score: sigmoid(log p_yes - log p_no). When one side is absent from the listed alternatives its probability is
    # bounded above by the smallest listed alternative (or 1e-12), which keeps the ranking continuous instead of collapsing to 1.0/0.0.
    floor = float(top.get("min_listed_prob") or 1e-12)
    ly = math.log(p_yes) if p_yes > 0 else math.log(max(floor, 1e-12))
    ln = math.log(p_no) if p_no > 0 else math.log(max(floor, 1e-12))
    z = max(-700.0, min(700.0, ly - ln))
    norm = 1.0 / (1.0 + math.exp(-z))
    return float(p_yes), float(p_no), float(norm)


# cross-model verifier mapping (pre-registered): the 30B verifies the 7B and the 70B; the 70B verifies the 30B; the dense
# robustness model is verified by the 30B
XMODEL = {"qwen2.5-coder-7b-instruct": "qwen3-coder-30b-a3b-instruct", "qwen3-coder-30b-a3b-instruct": "llama-3.3-70b-instruct",
          "llama-3.3-70b-instruct": "qwen3-coder-30b-a3b-instruct", "gemma-4-31b-it-qat": "qwen3-coder-30b-a3b-instruct"}
THRESHOLDS = {"verifier_gated_x": 0.5, "unguarded": 0.0, "logprob_gate": 0.6, "margin_gate": 0.3, "self_consistency": 0.8, "self_consistency_k1": 1.0, "confidence_gate": 0.7, "verifier_gated": 0.5, "verifier_gated_pB": 0.5, "verifier_wrong_control": 0.5, "verifier_gated_fs": 0.5}


def run(args: argparse.Namespace) -> None:
    if args.dataset.startswith("netconfeval"):
        items = load_netconfeval_t1(min(args.limit, int(os.environ.get("PAPER_NCE_LIMIT", "450"))))
    elif args.dataset == "teleqna_full":
        # subset-robustness cell: every TeleQnA question with a parsable gold option (no stratified frame)
        items = load_teleqna_full(Path(args.data))
    else:
        items = load_teleqna(Path(args.data), args.limit, args.category)
    FRAME_IDS.update(i["id"] for i in items)
    # the cache key carries the dataset definition: NetConfEval items keep their ids across policy-type settings, so a
    # reachability-only first pass must never be served to a three-policy cell (this happened once in a registered smoke)
    variant = ""
    if args.dataset.startswith("netconfeval"):
        variant = "-p" + "".join(x.strip()[0] for x in os.environ.get("PAPER_NCE_POLICIES", "reachability").split(",") if x.strip())
    # the sampling-scheme tag (CACHE_SCHEME) keeps first passes drawn under the old shared-seed scheme from ever being served
    cache_path = LANE / "experiments" / "cache" / f"firstpass-{args.dataset}{variant}-{CACHE_SCHEME}-{args.model}-s{args.seed}.jsonl"
    cache: dict = {}
    cache_state: dict = {"offset": 0}
    if args.dataset == "teleqna_full":
        # the full-benchmark robustness cell scores the SAME cached answers for the 1,500 frame items as the primary cells
        frame_cache = LANE / "experiments" / "cache" / f"firstpass-teleqna-{CACHE_SCHEME}-{args.model}-s{args.seed}.jsonl"
        refresh_cache(cache, frame_cache, {"offset": 0})
    refresh_cache(cache, cache_path, cache_state)
    fp_cost, policy_cost = collections.Counter(), collections.Counter()
    rows = []
    for item in items:
        fp = first_pass(item, args.model, args.seed, cache, cache_path, fp_cost, cache_state)
        score, extra = signal(args.method, item, fp, args.model, args.seed, policy_cost, args)
        declined = bool(fp.get("declined"))
        threshold = THRESHOLDS.get(args.method, 0.0)
        if is_nce(item):  # sequence-level signals live on a different scale; operating points pre-registered per dataset
            threshold = NCE_THRESHOLDS.get(args.method, threshold)
        answered = (not declined) and score >= threshold
        rows.append({"id": item["id"], "category": item["category"], "gold": item.get("gold"), "pred": fp.get("pred"), "score": float(score), "answered": bool(answered), "correct": bool(fp["correct"]), "declined": declined, "format_error": bool(fp.get("format_error")), **extra})
    n = len(rows)
    answered_rows = [r for r in rows if r["answered"]]
    coverage = len(answered_rows) / n if n else 0.0
    sel_acc = (sum(r["correct"] for r in answered_rows) / len(answered_rows)) if answered_rows else 0.0
    acc_all = sum(r["correct"] for r in rows) / n if n else 0.0
    ranked = sorted(rows, key=lambda r: (-r["score"], r["id"]))
    aurc, wrong = 0.0, 0
    for i, r in enumerate(ranked, start=1):
        wrong += 0 if r["correct"] else 1
        aurc += wrong / i
    aurc = aurc / n if n else 0.0
    def sel_at(cov: float) -> float:
        k = max(1, int(round(cov * n)))
        top = ranked[:k]
        return sum(r["correct"] for r in top) / len(top)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    per_item_extra_tokens = (policy_cost["prompt_tokens"] + policy_cost["completion_tokens"]) / n if n else 0.0
    per_item_fp_tokens = sum(((c.get("first_pass_cost") or {}).get("prompt_tokens", 0) + (c.get("first_pass_cost") or {}).get("completion_tokens", 0)) for c in (cache[i["id"]] for i in items)) / n if n else 0.0
    per_item_fp_seconds = sum((c.get("first_pass_cost") or {}).get("seconds", 0.0) for c in (cache[i["id"]] for i in items)) / n if n else 0.0
    format_rate = sum(1 for r in rows if r.get("format_error")) / n if n else 0.0
    declined_rate = sum(1 for r in rows if r.get("declined")) / n if n else 0.0
    result = {"metric_values": {"format_failure_rate": round(format_rate, 4), "declined_rate": round(declined_rate, 4), "aurc": round(aurc, 4), "selective_accuracy": round(sel_acc, 4), "coverage": round(coverage, 4), "accuracy": round(acc_all, 4),
                                "selective_accuracy_at_0.9": round(sel_at(0.9), 4), "selective_accuracy_at_0.8": round(sel_at(0.8), 4), "selective_accuracy_at_0.7": round(sel_at(0.7), 4),
                                "tokens_per_item": round(per_item_fp_tokens + per_item_extra_tokens, 1), "extra_tokens_per_item": round(per_item_extra_tokens, 1),
                                "latency_s_per_item": round(per_item_fp_seconds + (policy_cost["seconds"] / n if n else 0.0), 3),
                                "policy_latency_s_per_item": round(policy_cost["seconds"] / n if n else 0.0, 3)},
              "sampling": {"scheme": SAMPLING_SCHEME, "first_pass_temperature": FIRST_PASS_TEMPERATURE, "sample_temperature": SAMPLE_TEMPERATURE,
                           "verifier_temperature": VERIFIER_TEMPERATURE, "confidence_temperature": CONFIDENCE_TEMPERATURE},
              "evaluator_id": "e_netconfeval" if args.dataset.startswith("netconfeval") else "e_teleqna", "n": n, "dataset": args.dataset, "model": args.model, "method": args.method, "seed": args.seed, "host": HOST,
              "threshold": (NCE_THRESHOLDS.get(args.method, THRESHOLDS.get(args.method)) if args.dataset.startswith("netconfeval") else THRESHOLDS.get(args.method)), "first_pass_temperature": FIRST_PASS_TEMPERATURE, "cache": str(cache_path)}
    json.dump(result, (out / "result.json").open("w"), indent=1)
    print(json.dumps(result["metric_values"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(THRESHOLDS)); ap.add_argument("--dataset", default="teleqna"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.environ.get("TELEQNA_PATH", str(LANE.parents[1] / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt")))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("PAPER_LIMIT", "1500"))); ap.add_argument("--category", default=os.environ.get("PAPER_CATEGORY") or None)
    run(ap.parse_args())
