# AACL StateScope diagnostic-utility study (UI metrics)

A small within-subjects study measuring whether the StateScope UI helps humans
diagnose solution-state drift faster and more accurately than reading a raw trace.
This produces the paper's "diagnostic user accuracy/time" numbers.

**Judging policy.** Ground truth for every item (the true first drift step, the
diverged components) comes from the CAS oracle — never from a human or an LLM.
An open LLM may optionally rate *explanation readability/helpfulness* as a
secondary metric, but it must not judge mathematical correctness.

## Design

- **Conditions** (within-subjects, counterbalanced order):
  - `raw` — the model's per-step trace as plain text (op + claimed state per step)
  - `statescope` — the same run in the StateScope UI (drift highlighting + drift-analysis panel)
- **Items**: 8–12 runs where drift occurred (pick from the report's case-study
  candidates, `results/aacl_open_models/aacl_summary.json` → `case_studies`;
  balance families and include at least two *hidden drift* cases).
- **Task per item**: "Identify the first step at which the solution state diverges
  from a correct derivation." Participant reports a step index; time them.
- **Participants**: 6–10 (grad students / colleagues familiar with algebra/calculus).
  Each participant sees every item exactly once, half per condition.

## Per-item measurements

Record one CSV row per (participant, item):

```csv
participant,condition,problem_id,identified_step,true_step,time_s,explanation_correct,usefulness
p1,raw,family_b-2026-0007,3,1,84.2,,
p1,statescope,family_c-2026-0012,2,2,21.5,1,5
```

- `condition`: `raw` or `statescope`
- `identified_step`: the step the participant named (empty if they gave up)
- `true_step`: the CAS-oracle first drift step (from the run's `cod`)
- `time_s`: seconds from item shown to answer committed
- `explanation_correct` (statescope condition only): 1 if the participant judged the
  panel's deterministic explanation to correctly describe the divergence they found,
  0 otherwise, blank for `raw`
- `usefulness`: 1–5 Likert ("this presentation helped me locate the error"), blank allowed

## Metrics (computed by the scorer)

- **drift-point identification accuracy** = `identified_step == true_step`, per condition
- **time to locate drift** = median and mean `time_s`, per condition
- **explanation correctness** = mean of `explanation_correct` (statescope only)
- **usefulness** = mean Likert, per condition

## Scoring

```powershell
python scripts/score_human_study.py --input results/human_study/responses.csv --out-dir results/human_study
```

Writes `human_study_summary.md` / `.json` with the per-condition table and the
raw/statescope deltas. Report the deltas in the paper alongside the automated
headline metrics.
