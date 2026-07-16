# Data sources, provenance, and licenses

DriftMath uses four **provenance modes**, recorded per problem in `Problem.meta`
(`source`, `provenance`, `license`, `contamination_risk`, optional `original_id`,
`near_duplicate`):

| mode | meaning | contamination risk |
|------|---------|--------------------|
| `synthetic` | generated from scratch by SymPy templates (Family A, Family B) | none |
| `template_reinstantiation` | a known *form* re-instantiated with fresh parameters (Family B; MATH-seeded) | none–low |
| `program_lift` | a real record's program executed symbolically into a gold trace (MathQA) | high |
| `raw_natural` | a real problem solved freely, scored by step-consistency (no gold trace) | high |

## External datasets

These are referenced for provenance/seed forms only. **No dataset is downloaded by
the test suite** — tests load small local fixtures under `tests/fixtures/`. The HF
path is used only when a caller passes a source config without `local_jsonl`.

| dataset | HF path | license | how DriftMath uses it |
|---------|---------|---------|-----------------------|
| MathQA | `math_qa` | Apache-2.0 | **program lift**: parse `linear_formula`, execute over a SymPy whitelist, keep only records whose executed answer matches the labelled option. |
| MATH | `hendrycks/competition_math` | MIT | **form seeding only**: never parse free-form solutions; use subjects/forms to decide which Family B templates to reinstantiate with fresh parameters. |
| GSM8K | `openai/gsm8k` | MIT | (reserved) candidate source for natural-drift runs. |
| Lila | `allenai/lila` | varies by subset (see upstream) | (reserved) verify per-subset license before use. |

## Contamination notes

- `program_lift` (MathQA) reproduces the *original problem text*, so it carries a
  **high** contamination risk (it may appear in model training data); flag results
  accordingly and prefer it for analysis, not headline claims.
- `template_reinstantiation` (MATH-seeded) uses **fresh** generated parameters, so the
  emitted problem is not the original item (**low** risk). `original_id` links back to
  the seed form for traceability only.
- Synthetic Family A/B content is original to DriftMath (`CC0-1.0`, risk **none**).

Always re-check upstream licenses before redistributing any derived data. The license
strings recorded in `meta` come from the source config, not from this file.
