# DriftMath tool & operation layer

Two distinct kinds of "tools" live here -- keep them separate:

1. **CAS / helper tools** (`driftmath/runtime/tool_api.py`): pure, SymPy-backed
   functions (`simplify`, `solveset`, `compute_next`, `check_candidate`, `differentiate`,
   `integrate`, ...). No arbitrary Python; every input is parsed via `parse_expr_safe`.
   These are the *engine*.
2. **Agent operation tools** (the op vocabulary): the discrete steps a model may take
   (`bind`, `square_both_sides`, `establish_lemma`, ...). Each maps to a ledger handler
   and, where possible, a CAS check. These are what the model *chooses*.

```
LLM -> OperationAdapter -> parsed_ops -> System C/D -> verified tool_api / typed ledger
```

## op_specs is the single source of truth

`driftmath/runtime/op_specs.py` declares every operation once:

- `name`, `families`, `description`
- typed argument schema (`args`), `required` args, `additional_args`
- `mutates_ledger`, `cas_verified`, `state_fields` touched
- a prompt `example_args`

Everything else is **derived** from it, so vocabularies cannot drift:

| consumer | derives from op_specs |
|----------|-----------------------|
| `adapters.protocol.allowed_ops` | `ops_for_family` |
| `adapters.prompts.OP_HELP` / system prompt | descriptions + examples |
| `adapters.native_tools.op_tool_schemas` | `OpSpec.tool_schema()` (typed, required, `additionalProperties:false`) |
| `runtime.tool_api.validate_op` | `validate_args` |

A test asserts `op_specs.ALL_OPS == tool_api.KNOWN_OPS` (every handler has a spec and
vice-versa). This is what caught `check_both_valid` being missing from the old hardcoded
family-B list.

Native tool names are **identical** to text-JSON op names, so the two paths are
interchangeable. Text-JSON is the controlled main path; native is an optional ablation.

## Strict argument validation

`validate_op(op, args)` checks: op exists, `args` is an object, required args present,
arg types match the schema, and no unknown args (unless the spec allows them). It returns
a clear error string used as an agentic-failure event.

## CAS verification and ToolResult

`apply_op_verified(ledger, op) -> ToolResult` does **validate -> apply -> CAS-verify**:

```python
ToolResult(ok, op, error, before_state, after_state, verified, verification)
```

`verification.status` is `ok` / `failed` / `skipped` (skipped = unverifiable, never
blocks). Per-family checks (best-effort; gold traces always pass):

- **A**: `bind` recomputes the value; `report` checks final == target; `differentiate_substitution`
  checks `du == d/dx(u)`; `integrate_u` checks `d/du(antiderivative) == integrand`;
  `back_substitute` checks `u` no longer appears.
- **B**: `state_equation`/`exponentiate` parse; `square_both_sides` checks the squared
  equation actually follows; `cancel_factor` requires an exclusion constraint;
  `solve*`/`split_branches` check candidates == `solveset`; `check_both_valid` /
  `reject_extraneous` check candidates against the original equation + constraints;
  `finalize` checks the final set == current candidates.
- **C**: `bind` verifies the recurrence update from the ledger; `report` verifies the target.
- **D**: `state_function` parses; `establish_lemma` CAS-checks the lemma identity;
  `combine_lemmas` checks the final identity and requires dependency fan-in >= 2.

## System C vs System D (state ownership)

Both systems receive identical `parsed_ops` and use the same `tool_api`. The only
manipulated variable is who owns the state:

- **System D** applies each op via `apply_op_verified` to the typed ledger; the ledger
  snapshot is the state of record. On any invalid/failed/unverified op it records a
  `failure_event` and **stops**.
- **System C** records the model's `claimed_state` (prose) as the state of record. It
  still runs `apply_op_verified` on a scratch ledger to surface failures, but it keeps
  recording the claim and continues (so it can drift).

Failures are recorded in `trace.metadata["failure_events"]` with a `kind`
(`invalid_op` / `verification_failed` / `missing_state` / `parse_error`) and the
`verification` detail.

## How this supports both deliverables

- **MathNLP paper**: the controlled text-JSON op protocol + CAS verification gives clean,
  comparable SF/COD/recovery and a precise C-vs-D contrast (state ownership), with agentic
  failures recorded as first-class events.
- **StateScope demo**: `ToolResult` (before/after/verified/verification) and
  `failure_events` are exactly what the UI renders -- the live ledger, the first drift
  point, CAS pass/fail per step, and counterfactual replay.

## Run

```bash
pytest -q
python scripts/run_eval.py --experiment configs/experiments/smoke.yaml \
  --model-role large=mock --model-role small=mock
```

All offline: no network, no real APIs, no model downloads, no GPU.
