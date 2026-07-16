# StateScope — AACL-IJCNLP 2026 Demo Track: Reproduction Commands

> StateScope: an interactive model-agnostic system for detecting, localizing, and
> explaining solution-state drift in mathematical reasoning.

This document is the exact command sequence for the AACL demo-paper experiments.
**Open models only in the headline experiment path**: every reported matrix model
is downloaded into the repo-local `models/` directory and served locally with
vLLM. Optional hosted routes exposed by the UI are not used on this reproduction
path. The CAS (SymPy) is the correctness judge; drift explanations are
deterministic, generated from the state diff.

The target AACL model set (each is one config entry — adding a model means adding
a YAML under `configs/models/` and one line in `driftmath/models/aacl_models.py`):

| key | Hugging Face id | config |
|-----|-----------------|--------|
| `qwen25_1_5b` | Qwen/Qwen2.5-1.5B-Instruct | `configs/models/open_qwen25_1_5b.yaml` |
| `qwen3_4b` | Qwen/Qwen3-4B | `configs/models/open_qwen3_4b.yaml` |
| `qwen3_8b` | Qwen/Qwen3-8B | `configs/models/open_qwen3_8b.yaml` |
| `qwen25_math_7b` | Qwen/Qwen2.5-Math-7B-Instruct | `configs/models/open_qwen25_math_7b.yaml` |
| `r1_distill_qwen_14b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | `configs/models/open_deepseek_r1_distill_qwen_14b.yaml` |
| `qwen3_14b` | Qwen/Qwen3-14B | `configs/models/open_qwen3_14b.yaml` |
| `qwen3_30b_a3b` | Qwen/Qwen3-30B-A3B-Instruct-2507 | `configs/models/open_qwen3_30b_a3b_instruct_2507.yaml` |
| `qwen3_30b_a3b_thinking` *(optional)* | Qwen/Qwen3-30B-A3B-Thinking-2507 | `configs/models/open_qwen3_30b_a3b_thinking_2507.yaml` |

## 1. Install dependencies

```powershell
pip install -e ".[dev,aacl]"
```

The `aacl` extra provides `huggingface_hub` (model downloads) and `openai` (used
only as the OpenAI-compatible *client* for local vLLM endpoints). On the GPU box,
also install vLLM — on an AMD MI300X use a ROCm build of vLLM (e.g. the official
`rocm/vllm` container or a ROCm wheel); the commands below are identical.

## 2. Download models into `models/`

```powershell
python scripts/download_open_models.py --list                 # keys + local status
python scripts/download_open_models.py --model qwen3_4b       # one model
python scripts/download_open_models.py --model qwen3_4b,qwen3_14b
python scripts/download_open_models.py --all-aacl             # the full required set
python scripts/download_open_models.py --all-aacl --dry-run   # plan only, no network
```

Models land in stable directories such as `models/Qwen__Qwen3-4B` and
`models/Qwen__Qwen3-30B-A3B-Instruct-2507`; the local path is printed after each
download. Gated models: log in first with `huggingface-cli login` (or set
`HF_TOKEN`); the script never handles tokens itself.

**Fallback:** if a model has not been downloaded yet, the printed serve command
falls back to the HF id and vLLM will pull it from the hub into its own cache.
Download first for a repo-local, offline-reproducible setup.

## 3. Serve a model locally (vLLM)

Print the serve command for any configured model (it uses the `models/` path when
present, with `--served-model-name` kept at the canonical HF id):

```powershell
python -m driftmath.models.vllm_server --config configs/models/open_qwen3_4b.yaml --print-command
python -m driftmath.models.vllm_server --config configs/models/open_qwen3_30b_a3b_instruct_2507.yaml --print-command
```

Then run the printed command on the GPU box, e.g.:

```text
vllm serve models/Qwen__Qwen3-4B --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --dtype bfloat16 --host 127.0.0.1 --port 8000 \
  --trust-remote-code --served-model-name Qwen/Qwen3-4B
```

Serve one model at a time on port 8000 (the config default), or use a different
`--port` and pass `--base-url http://127.0.0.1:<port>/v1` to the batch runner.
On the MI300X (192GB) every model in the set — including the 30B MoE — fits on a
single GPU at full context; the per-config `vllm:` block already reflects that.

## 4. Build the 250-problem DemoBench

```powershell
python scripts/build_aacl_demobench.py --n 250 --seed 2026 --out data/aacl_statescope_demobench.jsonl
```

Writes:
- `data/aacl_statescope_demobench.jsonl` — 250 synthetic problems, balanced
  63/63/62/62 across families A/B/C/D, each with a CAS-verified gold trace
- `data/aacl_statescope_demobench.manifest.json` — seed, family counts, git SHA,
  package version, timestamp, provenance policy
- `data/aacl_statescope_demobench.curated.jsonl` — the small hand-picked UI subset
  (screenshot examples only; **not** part of the 250-problem evidence set)

## 5. Smoke evaluation

Offline (no GPU, mock model, 4 problems — verifies the whole pipeline; a separate
output dir keeps debug rows out of the real results):

```powershell
python scripts/run_aacl_batch.py --model mock --limit 4 --out-dir results/aacl_smoke
```

Against a served model (first 5 problems):

```powershell
python scripts/run_aacl_batch.py --model qwen3_4b --limit 5 --redo-completed
```

## 6. Full AACL batch evaluation

With the model's vLLM server running:

```powershell
python scripts/run_aacl_batch.py --model qwen3_4b --resume
```

One model at a time (recommended — swap the served model between runs):

```powershell
python scripts/run_aacl_batch.py --model qwen25_1_5b --resume
python scripts/run_aacl_batch.py --model qwen3_4b --resume
python scripts/run_aacl_batch.py --model qwen3_8b --resume
python scripts/run_aacl_batch.py --model qwen25_math_7b --resume
python scripts/run_aacl_batch.py --model r1_distill_qwen_14b --resume
python scripts/run_aacl_batch.py --model qwen3_14b --resume
python scripts/run_aacl_batch.py --model qwen3_30b_a3b --resume
```

or all keys in one invocation: `--model all --resume`. Useful flags:
`--base-url http://127.0.0.1:8001/v1` (non-default port), `--concurrency 4`
(parallel in-flight requests; vLLM batches server-side), `--systems`, `--limit`.

Outputs land in `results/aacl_open_models/`:

```text
metrics.jsonl    # one row per (problem, system, model) unit — appended + flushed per unit
traces.jsonl     # full candidate traces, appended per unit
manifest.json    # config, dataset, models, git SHA, versions
run_state.json   # live progress checkpoint (updated after every unit)
```

Each metrics row carries the stable unit key `problem_id|family|system|model|seed|sample`
and everything the paper tables need: final-answer correctness, state fidelity, COD/first
drift, propagation length, recovery, constraint fidelity, aligned/gold/executed step
counts, first-drift components, the CAS verdict at the drift step, per-problem and
per-step latency, token usage, and per-kind agentic failure counters
(parse / invalid op / CAS verification / missing state).

## 7. Resume an interrupted run

Kill the job at any point; completed units are already on disk. Then:

```powershell
python scripts/run_aacl_batch.py --model qwen3_4b --resume
# or equivalently
python scripts/run_aacl_batch.py --model qwen3_4b --only-missing
```

Units whose last row is `status=ok` are skipped; failed units are retried.
Without `--resume`, the runner refuses to overwrite completed units and tells you so.

## 8. Intentionally re-run completed units

```powershell
python scripts/run_aacl_batch.py --model qwen3_4b --redo-completed
```

Re-runs every requested unit and appends superseding rows with a bumped `attempt`
counter. The files are append-only; **the last row per unit key wins** everywhere
(resume logic and the report both deduplicate that way), so older rows remain in
the file as a clearly superseded audit trail.

## 9. Generate the report

```powershell
python scripts/make_aacl_report.py --input results/aacl_open_models/metrics.jsonl --out-dir results/aacl_open_models
```

Writes `aacl_summary.md` / `.csv` / `.json` plus one CSV per paper table:

```text
aacl_table1_performance.csv       model | size | family | system | acc | SF | COD | PL | recovery | CF
aacl_table2_system_benefit.csv    model | family | dSF(D-C) | dAcc(D-C) | dCF(D-C)
aacl_table3_failure_taxonomy.csv  family | drift_type | count | % | top first failed component
aacl_table4_runtime.csv           model | latency/problem | latency/step | parse fail | repair success | tokens | throughput
```

The markdown report opens with the headline block and also includes the COD
distribution (mean/median, early/mid/late shares, histogram data for the paper
figure), agentic failure rates, per-model C-vs-D gaps, first-drift components,
and case-study candidates.

## Metrics (what the paper reports)

All correctness judgments are CAS-based (SymPy). No LLM judges mathematical
correctness anywhere in the pipeline; an open LLM may at most rate explanation
readability in the human study (secondary UI metric only).

Headline metrics:

| metric | definition |
|---|---|
| final answer accuracy | candidate final answer symbolically equals gold |
| state fidelity (SF) | correct symbolic states / aligned reasoning steps |
| first drift (COD) | first aligned step whose state diverges from the oracle |
| propagation length (PL) | later aligned steps still wrong after COD |
| **hidden drift rate** | runs where COD exists **and** the final answer is correct — right answer, wrong intermediate state; answer checking alone misses these |
| **drifted failure rate** | runs where COD exists and the final answer is wrong — the failure traces to a specific earlier step |
| recovery rate | drifted runs that later return to the oracle state / drifted runs |
| constraint fidelity | aligned steps whose constraint sets match gold (domains, exclusions, extraneous-root guards) |
| C-vs-D gaps | ΔSF, ΔAccuracy, ΔConstraintFidelity = System D − System C, in points — the value of the external typed ledger |

Failure taxonomy: each drifted run is typed by the CAS verdict of the operation
executed at the drift step — `state_tracking` (the op was valid; the carried state
went stale), `invalid_operation` (the op itself failed CAS), or `unverified` —
together with the first diverged state component (bindings / constraints /
equation / candidates / final answer).

Agentic failure rates: parse failure, repair success (parse errors that the bounded
repair loop fixed), invalid operation, CAS verification failure, missing claimed state.

Efficiency: latency per problem and per reasoning step, completion tokens per
problem, throughput in problems/hour. Resume overhead is one linear scan of
`metrics.jsonl`, recorded as `resume_scan_s` in the run manifest (near zero).

Diagnostic-utility (UI) metrics — drift-point identification accuracy, time to
locate drift, explanation correctness, usefulness Likert — come from the small
human study described in [docs/aacl_human_study.md](docs/aacl_human_study.md):

```powershell
python scripts/score_human_study.py --input results/human_study/responses.csv --out-dir results/human_study
```

## 10. Start the UI

```powershell
python -m apps.statescope.server --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The model selector lists every configured open-weight
and proprietary model from `GET /api/models`. A separate execution-route selector
offers valid backends for that model: local vLLM or OpenRouter for open models, and
the vendor API or OpenRouter for proprietary models. Hosted credentials are read
only from server-side environment variables; the browser receives readiness status,
never key values. Every exported trace records sanitized model and route provenance.
The headline AACL results should still use the preregistered open/local matrix unless
the paper explicitly reports hosted runs as a separate analysis. "Mock (offline
debug)" is not an evaluation model. The UI and vLLM cannot share a port; if vLLM is
on 8000, start the UI on another port (for example, `--port 8080`).

For optional hosted routes, place credentials in the git-ignored repository-root
`.env` file and restart StateScope. Use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`; `.env.example` lists the supported names.
Keys are loaded with existing process environment variables taking precedence, and
secret values are never returned by `/api/models` or exported traces.

Interactive controls, all backed by the same runtime the batch evaluation uses:

- **Controlled comparison** is the presentation default. It holds the CAS-derived
  operation schedule and typed arguments constant across C and D so the side-by-side
  view isolates state ownership. **Autonomous diagnostic** preserves free operation
  choice for exploratory traces. When `OPENAI_API_KEY` is ready, the UI initially
  selects the low-cost `gpt-5-nano` native route; the headline paper matrix remains
  open/local and is not changed by this UI default.
- **Step through ▸** executes the paired C/D run one model turn per click, with
  per-step CAS status, oracle diffs, and metrics updating live; the finished run
  becomes a normal remembered run.
- **what-if** on any step opens the step editor. Edit the *operation* (both
  systems) or — System C only — the *claimed state itself*: C's model owns its
  state, so you can inject a stale binding or correct a wrong one, while System D
  refuses state edits by design. Invalid op edits are rolled back to the last safe
  ledger state and retained as editable failure nodes. Then **▶ Continue with model** (the same model keeps solving live from
  the edited, CAS-checked state — new real turns through the identical protocol)
  or **Re-derive (no model)** for the deterministic counterfactual. The what-if
  panel compares before/after metrics and remains fully editable. Each result is a
  new branch shown in the branch navigator; failed or omitted steps expose a direct
  edit-and-retry path, and quick-restore controls load the scheduled operation or
  oracle claimed state.
- **↻ fresh** regenerates the selected trap with new parameters (dynamic,
  contamination-resistant instances, demonstrated live).
- The first drift step shows the **oracle-expected state** inline; steps display
  the model's one-line rationale and a "repaired" badge when the bounded JSON
  repair loop fired. Original and what-if sessions export as JSON/Markdown.

The UI uses relative API paths, so it also works behind a path-prefixing proxy
(e.g. `jupyter-server-proxy` at `/proxy/<port>/`).

## Tests

```powershell
python -m pytest -q
```

All tests are offline: no model downloads, no GPU, no network.

## Known limitations

- The runner assumes the selected model's vLLM endpoint is up; a dead endpoint
  produces `status=failed` rows (retried on the next `--resume`) rather than
  crashing the run.
- All configs default to `http://localhost:8000/v1`; multi-port setups use
  `--base-url` (batch) or the endpoint field (UI).
- DemoBench is synthetic-only by policy (no MathQA / raw MATH text) to avoid
  contamination in headline demo evidence.
