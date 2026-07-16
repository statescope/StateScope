# StateScope (AACL-IJCNLP system demo)

StateScope makes **solution-state drift** visible: a model solves a math problem one
operation at a time, and StateScope compares every step to a SymPy oracle to show
*where* and *how* the model loses the thread — with System C (the model owns state in
prose) and System D (a typed ledger owns state) side by side.

It is the demo app on top of the benchmark core (`driftmath/`). Every example carries
a SymPy-verified gold trace and the CAS oracle supplies ground truth at runtime; the
CAS is the correctness judge, and drift explanations are generated deterministically
from the state diff — never by an LLM.

The interface exposes every committed open-weight and proprietary model configuration.
Model identity is separate from execution route: open models offer local vLLM and
OpenRouter routes; proprietary models offer their native API and OpenRouter. Hosted
credentials remain server-side environment variables and are never entered in or
returned to the browser. The headline AACL experiment matrix can remain open/local
while the broader catalog demonstrates backend extensibility.

## Run it

Zero external dependencies (Python stdlib HTTP server):

```bash
# from the repo root, with the project installed (pip install -e .)
python -m apps.statescope.server --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

The **Mock (offline debug)** model works fully offline — pick an example, toggle
*plant drift at step k*, and watch System C drift while System D's ledger re-derives
state and stays correct. It is a debug mode, not the demo path.

### Select an execution route

1. Download it into the repo-local `models/` store and serve it with vLLM
   (see [README_AACL_DEMO.md](../../README_AACL_DEMO.md) for exact commands):

   ```bash
   python scripts/download_open_models.py --model qwen3_4b
   python -m driftmath.models.vllm_server --config configs/models/open_qwen3_4b.yaml --print-command
   ```

2. Start StateScope, select the model and **Local vLLM**, adjust the endpoint if
   needed, and Run. For a hosted route, set the corresponding environment variable
   before starting StateScope: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
   `GOOGLE_API_KEY`, or `OPENROUTER_API_KEY`. Alternatively, enter a key in the
   masked UI field for the selected route. A UI-entered key is request-scoped: it
   is not written to disk, browser storage, exported traces, or provenance. Leave
   the field blank to use the server-side environment key.

The repository root contains a git-ignored `.env` ready for local credentials and
a committed `.env.example` template. Paste values after the equals sign, then
restart StateScope so the process reloads them:

```dotenv
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
```

The real `.env` is ignored by Git. Do not put keys in YAML model configs,
command-line arguments, exported traces, or `.env.example`. Keys entered through
the masked browser field remain only in the current page and request lifecycle.

All routes use the controlled text-JSON operation protocol. JSON and Markdown
exports include sanitized model, backend, route, endpoint, and config provenance so
runs remain attributable and comparable.

## What you see

- **Two panels, one model.** The default controlled comparison supplies the same
  CAS-derived operation schedule and typed arguments to System C (text state) and
  System D (ledger), isolating *state ownership*. Autonomous diagnostic mode leaves
  operation choice to the model and labels schedule divergence separately.
- **Per-step trace** with the operation, a compact state snapshot (bindings / constraints /
  candidates / equation / answer), a **CAS status** (ok / failed / not-checkable), and
  the divergence-from-oracle components.
- **Drift analysis panel**: first drift point, which state components diverged, the CAS
  verdict at the drift step, and a deterministic explanation generated from the diff.
- **What-if (edit & continue)**: click *what-if* on any step of the trace the model just
  produced. Edit the **operation** (both systems) or — on System C only — the **claimed
  state itself**: C's model owns its state, so you can inject a stale binding or correct a
  wrong one; System D refuses state edits by design (the ledger owns state — you may only
  edit operations). Invalid operation attempts roll back transactionally to the last safe
  ledger state and remain visible as editable failure nodes. Then either **let the same model
  continue live from the edited state** (new real model turns through the identical
  per-turn protocol) or **re-derive downstream deterministically** (no model). The result
  is scored against the same gold trace and compared side-by-side with the original run —
  so you can see whether the model recovers from (or compounds) an injected or corrected
  error. A branch navigator preserves earlier experiments, and every result remains
  editable at every retained step; a protocol failure that omitted its step exposes an
  **edit and retry** recovery action. Mock runs use the deterministic path.
- **Step-through mode**: run the same paired comparison one model turn per click, with
  per-step CAS status, diffs, and metrics updating live; the finished run becomes a normal
  remembered run (what-if and export work on it).
- **Fresh instances (↻)**: regenerate the selected trap with new parameters on demand —
  the live proof that evaluation instances are dynamic, not a static test file.
- **Oracle-expected view**: the first drift step shows the state the CAS oracle expected,
  right under the state the system recorded.
- **Export**: download any original or what-if session as JSON or Markdown.

## API

```
GET  /api/health      -> {ok, mode: "statescope-research"}
GET  /api/models      -> {ok, models: [{key, model_id, access, routes, ...}]}
GET  /api/examples    -> {ok, examples: [...]}
GET  /api/ops         -> {ok, ops: {family: [{op, description, example_args, ...}]}}
GET  /api/export      ?run=<run_id>&system=c|d&scope=run|whatif&fmt=json|md
POST /api/run         {example_id, model_key, route_key?, base_url?, execution_mode: "controlled"|"autonomous", system: "both"|"c"|"d", drift_step?}
POST /api/continue    {run_id, system: "c"|"d", step, base_branch_id?, op?, args?, claimed_state?, mode: "model"|"replay"}
POST /api/live/start  {example_id, model_key, route_key?, base_url?, execution_mode?, drift_step?} -> {live_id}
POST /api/live/step   {live_id} -> {systems: {c, d}, done, run_id?}   # one model turn per call
POST /api/live/stop   {live_id}
POST /api/regenerate  {example_id, seed?} -> {example}                # fresh-parameter reinstantiation
POST /api/replay      {example_id, system, step, formula?|args?}      # gold-trace sandbox
```

`/api/continue` is the interactive core: it addresses the remembered run or a prior
`base_branch_id`, keeps the trace prefix, applies the edited op to a fresh ledger
replay (CAS-checked) — or, for System C, records the user-authored claimed state —
then either continues with the run's own model ("model") or re-derives
deterministically ("replay"). The response returns a new `branch_id` that can be used
as the base of the next intervention. Up to 128 branches per system are retained in a
server session; older experiments remain visible in the browser until the page is reset.

## Example problems (curated traps)

All four families, chosen because multi-step solvers tend to drift on them:

| id | family | the trap |
|----|--------|----------|
| `radical_extraneous` | B | squaring manufactures an extraneous root the domain must reject |
| `log_domain` | B | the log-domain constraint must be carried the whole way |
| `abs_branches` | B | both ± branches must survive |
| `rational_exclusion` | B | cancelling needs `x ≠ excluded` recorded |
| `two_state_recurrence` | C | coupled x,y; one swapped cross-binding propagates |
| `linear_recurrence` | C | off-by-one: report `aₙ` when the state is `aₙ₋₁` |
| `compound_balance` | C | grow-then-withdraw with exact rationals; stale accumulators |
| `u_substitution` | A | `u` must stay live and be discharged at the end |
| `binding_chain` | A | fan-in > 1 let-bindings; a stale value diverges the chain |
| `triple_product_derivative` | D | product rule over 3 factors + log domain; fan-in 3 |
| `chain_product_derivative` | D | composite factor whose derivative is its own lemma chain |

These curated examples are for the UI only; the paper evidence set is the released
250-problem DemoBench (`scripts/build_aacl_demobench.py`).

## Backend (importable)

```
apps/statescope/
  server.py            # stdlib HTTP server + JSON API (run / replay / examples / models / health)
  examples.py          # curated Problems (each with a gold trace)
  web/index.html       # single-file dashboard (no build step)
  backend/
    session.py         # run_session -> Session (trace, gold, metrics, diffs, COD, logs)
    replay.py          # counterfactual_replay (edit a step, re-derive downstream, no model)
    stepper.py         # LiveSolver: the C/D per-turn rules as a resumable, steppable object
    intervene.py       # intervene_and_continue (edit a step, the *live model* continues)
    export.py          # export_session_json / export_session_markdown
```
