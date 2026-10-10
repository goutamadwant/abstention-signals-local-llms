#!/usr/bin/env python3
"""Audit of the TeleQnA scorer against the official evaluation_tools.py semantics, on RAW model text.

The official script (netop-team/TeleQnA, evaluation_tools.py) marks a prediction correct when the model's
answer string, "option {id}: {text}", equals the gold entry. Our runner asks for the option number only and
compares option ids, so three things must hold and are checked here on 500 sampled first-pass records:

 A. gold extraction: the id we parsed from the gold "option N: text" string is the N of the official answer field;
 B. answer parsing: an INDEPENDENT parser (written without looking at run_cell.parse_option: first standalone
    integer or "option N" in the raw text, ambiguous multi-number texts flagged) yields the same option id as the
    runner's parse on the raw `text` field, and non-bare-digit outputs are counted separately;
 C. correctness: "option {pred}: {options[pred]}" == gold string  <=>  runner's `correct` flag (the official equality).

Any disagreement in A, B or C counts. Kill criterion: disagreement rate > 0.5%.
"""
from __future__ import annotations

import json, random, re, sys
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else LANE.parents[1] / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt"))
cache_files = sorted((LANE / "experiments" / "cache").glob("firstpass-teleqna-*.jsonl"))
if not cache_files:
    raise SystemExit("no first-pass cache yet")
rows = []
for f in cache_files:
    for line in f.read_text().splitlines():
        rows.append(json.loads(line))
random.Random(7).shuffle(rows)
sample = rows[:500]


def independent_parse(text: str, n_options: int):
    """Independent re-implementation: prefer 'option N'; else the FIRST standalone integer; flag if several distinct ints."""
    t = text.strip().lower()
    m = re.search(r"option\s*(\d+)", t)
    ints = [int(x) for x in re.findall(r"(?<![\d.])(\d+)(?![\d.])", t)]
    if m:
        val = int(m.group(1))
    elif ints:
        val = ints[0]
    else:
        return None, False
    ambiguous = len(set(ints)) > 1 and not m
    return (val if 1 <= val <= n_options else None), ambiguous


disagree, ambiguous, non_bare, details = 0, 0, 0, []
for r in sample:
    q = data[r["id"]]
    options = {k: v for k, v in q.items() if k.startswith("option ")}
    gold_str = str(q["answer"]).strip()
    gm = re.match(r"option\s*(\d+)", gold_str.lower())
    gold_id = int(gm.group(1)) if gm else None
    raw = str(r.get("text", ""))
    if not re.fullmatch(r"\s*\d+\s*", raw):
        non_bare += 1
    ind, amb = independent_parse(raw, len(options))
    ambiguous += amb
    pred = r.get("pred")
    official_correct = pred is not None and f"option {pred}: {options.get(f'option {pred}', '')}".strip() == gold_str
    problems = []
    if gold_id is None or (r.get("gold") is not None and gold_id != r.get("gold")):
        problems.append("gold_id")
    if ind != pred:
        problems.append(f"parse(ind={ind},runner={pred})")
    if official_correct != bool(r.get("correct")):
        problems.append("correctness")
    if problems:
        disagree += 1
        details.append({"id": r["id"], "raw": raw[:40], "problems": problems})
rate = disagree / len(sample)
report = {"audited": len(sample), "disagreements": disagree, "rate": round(rate, 4), "non_bare_digit_outputs": non_bare, "ambiguous_multi_number_outputs": ambiguous,
          "checks": ["A gold id extraction", "B independent parser vs runner parse on raw text", "C official string equality vs runner correct flag"],
          "kill_threshold": 0.005, "pass": rate <= 0.005, "examples": details[:20]}
json.dump(report, open(LANE / "experiments" / "scorer_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "examples"}))
