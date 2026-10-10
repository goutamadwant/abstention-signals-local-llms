# Free Signals or Self-Verification? Abstention signals for local LLMs

Code, data and analysis outputs for the paper *Free Signals or Self-Verification? A Cost-Accounted Case Study of Local LLM Abstention in Telecom* (Goutam Adwant).

The study compares abstention policies for local open models served by LM Studio: free signals read from the answer
distribution (max-logprob, top-two margin), sampling agreement (self-consistency with one or five extra samples),
verbalised confidence and a P(True)-style self-verification gate. Every policy scores the same cached first-pass
answer; policies are compared by the area under the risk-coverage curve (AURC) and by selective accuracy at matched
coverage, with token and latency cost. Tasks: TeleQnA (multiple choice, 1,500-question stratified frame) and NetConfEval
task 1 (450 requirement batches). Models: Qwen2.5-Coder-7B, Qwen3-Coder-30B-A3B and Llama-3.3-70B (Q4 GGUF), three
request seeds; Gemma 4 31B QAT as a single-seed control.

## Layout

```
artifact/      run_cell.py (one experiment cell), analyze.py (all tables, figures and numbers), audit_scorer.py, requirements.txt
environment/   reproduce_matrix.py (full re-run), control_option_aware.py (option-aware verifier control),
               check_benchmarks.py, verify_models.py + models.json (model and benchmark pins),
               audit_scorer_v2.py + scorer_audit_v2.json (TeleQnA scorer audit), probe_raw_probs.py + probe_raw_probs.json,
               serving_evidence.json, verify_headline_reproduction.py + headline_expected.json (re-run check)
protocol/      protocol.yaml: pre-registered design, hypotheses, analysis plan, controls and dated amendments
experiments/   registry.json and per-run outputs (redacted, see below), first-pass caches read by the analysis
analysis/      results_manifest.json, generated_numbers.tex, tables/, figures/
```

## Setup

- Python 3.12 or newer: `pip install -r artifact/requirements.txt`
- Benchmarks (not redistributed), checked out at the pinned commits:
  - TeleQnA: https://github.com/netop-team/TeleQnA, commit `d7ee5e3` (`TeleQnA.txt`; access-protected archive)
  - NetConfEval: https://github.com/RedHatResearch/conext24-NetConfEval, commit `24edc15`

  ```bash
  export TELEQNA_PATH=/path/to/TeleQnA/TeleQnA.txt
  export NETCONFEVAL_PATH=/path/to/conext24-NetConfEval
  python3 environment/check_benchmarks.py
  ```
- For new runs only: LM Studio with its local server enabled (default `http://127.0.0.1:1234`) and the GGUF files
  listed in `environment/models.json` (`python3 environment/verify_models.py <LM Studio models directory>`).

## Reproduce the analysis (no model calls)

```bash
cd artifact
python3 analyze.py 10000      # the reported resample count; about 3 h on one machine
python3 analyze.py 200        # quick check: identical point estimates, noisier intervals
```

This regenerates `analysis/` from the shipped run records. `NETCONFEVAL_PATH` is needed to rebuild the NetConfEval frame.

## Run one cell

Move the shipped `experiments/` aside first (`mv experiments experiments.redacted`): the scripts write to
`experiments/`, and with the redacted copy in place a run would reuse its caches.

```bash
cd artifact
PAPER_LMSTUDIO_HOST=http://127.0.0.1:1234 PAPER_LIMIT=1500 PAPER_NCE_LIMIT=450 \
PAPER_NCE_POLICIES=reachability,waypoint,loadbalancing \
python3 run_cell.py --method logprob_gate --dataset teleqna --seed 1 --model qwen2.5-coder-7b-instruct --out /tmp/cell
```

Methods: `unguarded`, `logprob_gate`, `margin_gate`, `self_consistency`, `self_consistency_k1`, `confidence_gate`,
`verifier_gated`; datasets `teleqna`, `netconfeval_t1`.

## Re-run everything

```bash
mv experiments experiments.redacted
cd artifact
python3 ../environment/reproduce_matrix.py --dry-run
python3 ../environment/reproduce_matrix.py --host http://127.0.0.1:1234 --models-dir <LM Studio models directory> --resume
python3 analyze.py 10000
python3 ../environment/verify_headline_reproduction.py --out /tmp/headline_check.json
```

`reproduce_matrix.py` runs the 123 matrix cells, the 17 sensitivity and control cells and the option-aware verifier
control (`environment/control_option_aware.py`, TeleQnA, seed 1), then the analysis. The full run took 20.8 cell-hours
on an Apple M3 Ultra (96 GB) with one model resident at a time.

A re-run reproduces the results within tolerance, not digit for digit: the server's reported token probabilities can
differ slightly between a first and a repeated identical request. `verify_headline_reproduction.py` checks a re-run
against `environment/headline_expected.json`. For TeleQnA and the three main models, it requires:

- the same H1 outcome;
- the same sign of the pooled AURC difference, with its 95% interval on the same side of zero;
- a pooled difference within 0.02 AURC of the reference;
- the same sign for every per-seed difference;
- complete per-seed cells.

NetConfEval and single-seed rows are reported, not decisive. Exit codes: 0 pass; 1 a check fails; 2 incomplete input
or different code; 3 the manifest comes from the shipped registry rather than a re-run.

The released code is the code used for the reported runs, with comments edited for release. The run records and
`analysis/results_manifest.json` carry the hashes of the code as executed. `run_cell.py` and `audit_scorer.py` are
unchanged, so their hash still matches the run records.

## Data: what the shipped copy changes

`experiments/` holds the 503 run records and the outputs of the 143 runs the analysis reads, redacted as follows:

- Machine paths and the hostname are replaced by placeholders; command strings, environments and attempt logs are not
  shipped.
- TeleQnA question ids are replaced by keyed pseudonyms, each question's option labels are permuted with a secret
  per-question permutation, and the correct-option field is removed. The benchmark's answer key cannot be recovered
  without the key, which is not published.
- Free text that could carry question content is removed: verifier replies keep only the YES/NO verdict, and non-digit
  first-pass text is dropped. Scores, correctness, categories, probabilities, token counts and latency are kept.
- Rows are shuffled within each file.

Someone who already holds the TeleQnA questions and re-runs the same models could link rows through the kept scores and
token counts, but such a reader already holds the answer key.

Effect on the analysis: per-cell metrics, H1 differences and H1 outcomes are identical on this copy. Steps that order
item ids before drawing random numbers or assigning folds see the pseudonyms in a different order. These are the
bootstrap intervals and p-values, the H2 tau bootstrap, the cross-validated combination and McNemar's tie-break. As a
result, on TeleQnA:

- p-values and intervals differ within Monte Carlo error;
- one discrete tau bound moves by a step (0.60 to 0.47);
- the cross-validated combination differs by up to about 0.002 AURC;
- McNemar counts against tied baselines differ.

At 10,000 resamples, 44 of the 1,310 generated numbers differ, all of them of these kinds. NetConfEval numbers are
reproduced exactly.

The TeleQnA scorer audit (`environment/audit_scorer_v2.py`) needs the raw replies and real question ids. It runs on the
caches of a re-run, not on this copy. Its stored result covers 23,500 replies with pseudonymous ids:
`environment/scorer_audit_v2.json`.

## Models and licences

- Code and generated outputs: MIT (`LICENSE`).
- Models: Qwen2.5-Coder-7B, Qwen3-Coder-30B-A3B and Gemma 4 31B QAT (Apache-2.0); Llama 3.3 70B (Llama 3.3 Community
  License). All are Q4 GGUF builds served by LM Studio 0.4.25 with the llama.cpp runtime. Builds are pinned by SHA-256 in
  `environment/models.json`; the weights are not redistributed.
- Benchmarks: TeleQnA and NetConfEval (MIT), not redistributed.

## Citation

See `CITATION.cff`.
