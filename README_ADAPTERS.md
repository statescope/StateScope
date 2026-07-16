# DriftMath operation adapters

The adapter turns any model's output into the DriftMath operation protocol:

```
LLM -> OperationAdapter -> parsed_ops -> System C/D -> SymPy tool_api / ledger
```

It produces, per turn, a `ModelResponse` with the **exact** shape the MockModel emits
(`parsed_ops=[{"op","args"}]`, `raw["claimed_state"]`, `raw["done"]`), so System C/D
treat mock and real models identically.

## Why text-JSON is the main research path

For the MathNLP paper the C-vs-D comparison must be **controlled**: every model — open
or closed, tool-capable or not — uses the *same* `text_json` operation protocol (a
strict single-JSON-object-per-turn format). This removes provider-specific tool-calling
behaviour as a confound, so any C-vs-D difference is attributable to **state ownership**,
not to how a provider implements tools.

`native` tool calling is offered only as an **optional ablation / demo path**. It is not
the default and is never used for the headline scientific comparison.

## How C vs D stays controlled

Both systems receive the identical `parsed_ops` stream from the adapter. The only
manipulated variable is who owns the state:

- **System C** records the model's `claimed_state` (prose) as the state of record — it
  can drift. If a step has no claimed state, an empty state is recorded and the adapter
  parse error is logged.
- **System D** ignores `claimed_state` and applies the op to the typed ledger; the
  ledger snapshot is the state of record.

Every parsed/native op is validated against `tool_api` (`validate_op`) **before** it is
applied. Invalid ops, tool-apply failures, and unrepaired parse failures are recorded as
agentic-failure events (`trace.metadata["failure_events"]`); System D stops on an invalid
op, System C logs it and continues recording prose state.

## Adapter modes

```yaml
adapter:
  mode: text_json      # main path: strict JSON, same for every model
  repair_budget: 2     # invalid output -> repair prompt, up to N retries, then stop
  native_fallback: false
```

```yaml
adapter:
  mode: native         # ablation/demo: provider tool calling (op names == text-JSON ops)
  repair_budget: 2
  native_fallback: true # if a model lacks tool support or a call fails, use text_json
```

- **text_json**: strict prompt (allowed ops + JSON schema, no markdown/prose), one op +
  `claimed_state` per turn, `{"done": true, "op": null}` to finish. The JSON parser
  tolerates accidental code fences and prose around one object, normalizes the native
  shape `{"name","arguments"}`, and validates the op against the family's vocabulary.
- **native**: passes provider-agnostic tool schemas (one per op) to
  `generate_with_tools`; tool calls are normalized to `{"op","args"}`. Used only when
  `model.supports_tools` is true; otherwise it falls back to `text_json` (if configured)
  or stops with a clear error.

## Example commands

```bash
# main scientific path (text-JSON), open models via vLLM:
python scripts/run_eval.py --experiment configs/experiments/gonogo_open_textjson.yaml

# native tool-calling ablation (fallback to text-JSON where unsupported):
python scripts/run_eval.py --experiment configs/experiments/gonogo_open_native_ablation.yaml

# override roles on any experiment:
python scripts/run_eval.py --experiment configs/experiments/gonogo_open_textjson.yaml \
  --model-role large=configs/models/open_qwen3_14b.yaml \
  --model-role small=configs/models/open_qwen3_4b.yaml

python scripts/make_report.py --input results/gonogo_open_textjson/metrics.jsonl
```

> `injected:*` conditions are a **mock-only** experimental control (they replay a
> corrupted gold script). With a real model behind an adapter there is no script to
> corrupt, so the runner solves the problem directly and those rows measure the model's
> *organic* drift on the same problems.
