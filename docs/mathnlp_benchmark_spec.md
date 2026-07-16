# DriftMath for MathNLP 2026: frozen paper and benchmark specification

Status: **design freeze, 15 July 2026**. Any change to the primary task, test
composition, labels, or headline metrics after test evaluation begins must be
recorded as a protocol deviation. Numerical findings remain unfrozen until the
final runs finish.

## 1. Paper story

### Title

**DriftMath: Outcome-Equivalent Minimal Pairs for Diagnosing State Drift in
Mathematical Reasoning**

### One-sentence thesis

Final-answer correctness does not establish process correctness; DriftMath tests
whether a model can detect and localize controlled changes to the live mathematical
state when the problem and final answer are held fixed.

### Scope

The paper studies *solution-state drift*: a binding, constraint, candidate set,
current expression, recurrence value, or dependency silently ceases to be justified
by the preceding solution. It does **not** claim to evaluate all mathematical
reasoning, prove arbitrary natural-language arguments, or establish that a CAS is a
universal mathematical judge.

### Distinction from existing process-error benchmarks

ProcessBench already asks for the earliest erroneous step and includes model
solutions whose final answers are correct. DriftMath therefore cannot claim novelty
from that condition alone. Its contribution is a controlled, paired diagnostic:

1. the same problem appears as a clean solution, an outcome-masked drift solution,
   and a wrong-answer drift control;
2. the clean and drifted solutions are minimal pairs with declared changed spans;
3. labels identify the first drift, all affected steps, state component, error
   mechanism, propagation, and recovery mechanism; and
4. formal evidence is retained where exact verification is supported, while
   natural-language and unsupported steps require independent human annotation.

### Research questions

- **RQ1 — Outcome-controlled detection.** Can a model distinguish a clean process
  from a drifted process when both have the same correct final answer?
- **RQ2 — Localization and diagnosis.** Can it identify the first drift step, the
  affected state component, and the drift mechanism?
- **RQ3 — Propagation and recovery.** How do performance and confidence change when
  drift is explicitly corrected, silently reset, cancelled by a compensating error,
  or rendered irrelevant by a discarded branch?
- **RQ4 — Generalization.** Do findings persist across fresh controlled problems,
  natural-language program-lifted problems, curated competition problems, state
  difficulty, and an independently sourced naturalistic challenge set?
- **RQ5 — Mitigation.** During solving, does runtime-owned verified state reduce
  hidden drift relative to model-maintained textual state under a controlled
  operation schedule?

### Contributions we may claim before seeing results

1. A definition and taxonomy of solution-state drift that separates process
   validity, answer validity, propagation, and recovery.
2. A benchmark of outcome-equivalent minimal pairs/triplets with explicit symbolic
   state annotations and provenance.
3. A hybrid validation protocol combining operator-level symbolic checks,
   independent numerical/property checks, and double human annotation where formal
   verification is incomplete.
4. Paired metrics that prevent final-answer shortcuts and expose detection,
   localization, component diagnosis, and recovery sensitivity.
5. An evaluation and controlled ledger intervention across multiple open models.

### Claims that are prohibited without final evidence

- “CAS-verified” must apply only to the exact fields and transitions checked by the
  verifier, never to an entire arbitrary natural-language solution.
- Do not claim coverage of all mathematics, proofs, geometry, or informal argument.
- Do not call the benchmark contamination-free. Fresh parameterization reduces
  instance overlap; external text remains contamination-prone and is reported
  separately.
- Do not claim that every deviation from one reference trajectory is an error.
  Benchmark labels are based on local step validity and state continuity; ambiguous
  alternative derivations are excluded or human-adjudicated.
- No numerical superiority, model ranking, or ledger benefit is a contribution
  until paired confidence intervals and completeness checks are available.

## 2. Benchmark task

### Unit shown to an evaluated model

Each evaluation item contains:

1. a natural-language mathematical problem;
2. a numbered candidate solution of at least three steps; and
3. its stated final answer.

The hidden label is one of:

- `clean`: the process and final answer are correct;
- `outcome_masked_drift`: at least one step is invalid or carries an unjustified
  state, but the stated final answer is correct; or
- `wrong_answer_drift`: the process drifts and the final answer is wrong.

For the core benchmark, these three candidates form a matched triplet for the same
base problem. This design prevents the evaluator from treating final-answer
correctness as a proxy for process correctness.

### Required model outputs

The primary critic task returns:

- `process_valid`: yes/no;
- `first_error_step`: `null` or a zero-based index;
- `affected_components`: one or more taxonomy labels;
- `drift_types`: one or more mechanism labels; and
- `confidence`: a number in `[0,1]`.

Free-form critique text is collected for qualitative analysis but is not used as a
correctness judge.

### Drift taxonomy

State components:

- `binding`
- `constraint`
- `current_expression`
- `current_equation`
- `candidate_set`
- `dependency`
- `index_or_iteration`
- `lemma`
- `final_answer`

Mechanisms:

- `sign_or_arithmetic`
- `stale_value`
- `name_or_variable_swap`
- `dropped_constraint`
- `invalid_cancellation`
- `branch_loss`
- `extraneous_candidate`
- `index_shift`
- `false_lemma`
- `over_retention`
- `state_reset`
- `other_adjudicated`

Recovery modes for outcome-masked drift:

- `explicit_correction`: the solution acknowledges and fixes the error;
- `silent_reset`: a later step resumes the correct state without a valid bridge;
- `compensating_error`: a second error cancels the first at the outcome level;
- `discarded_branch`: the corrupted state lies on a branch later discarded; or
- `independent_recomputation`: the answer is recomputed correctly from unaffected
  information.

The benchmark contains both one-error and multi-error items. `first_error_step` is
the primary localization target; `erroneous_steps` preserves all adjudicated error
locations.

## 3. Data composition and splits

### Frozen release target

The core set contains **240 base problems and 720 matched candidate solutions**:

| stratum | base problems | dev | test | candidate solutions |
|---|---:|---:|---:|---:|
| fresh controlled generation | 120 | 24 | 96 | 360 |
| MathQA program lift | 80 | 16 | 64 | 240 |
| curated MATH algebra/calculus | 40 | 8 | 32 | 120 |
| **core total** | **240** | **48** | **192** | **720** |

Every base problem has one clean, one outcome-masked, and one wrong-answer
candidate. A separate **100-item naturalistic challenge set** contains model-written
solutions selected by answer correctness and double-annotated for process validity,
onset, and state component. It is not forced into artificial triplets and is never
pooled with the controlled headline result.

The current 250-problem seed-2026 dataset is development/pilot material only. It is
not the frozen MathNLP test set and its existing model outputs are not final paper
evidence.

### Split policy

- There is **no training split and no fine-tuning in the paper**.
- Development labels may be used only to finalize prompt format and parsers.
- Test labels remain unread until prompts, model settings, and analysis code freeze.
- Split base problems before creating candidate variants; all members of a triplet
  remain in one split.
- External data respect upstream train/test boundaries: upstream train material may
  enter DriftMath dev; upstream test material enters DriftMath test.
- `original_id`, normalized problem text, and declared `leakage_group` may occur in
  only one split.
- For generated items, `leakage_group` represents the surface template and parameter
  regime. Test parameters and surface realizations must be held out from dev.
- Report every result by source stratum; never hide external contamination risk in a
  pooled number.

### Provenance policy

Each item records source name, source mode, license, upstream ID, upstream split,
contamination risk, generation seed, generator revision, and leakage group.

Allowed source modes are:

- `synthetic`: problem and trace generated from an original controlled template;
- `program_lift`: external natural-language problem with an executable, verified
  operation program;
- `human_curated`: external problem/solution whose relevant steps were independently
  annotated and adjudicated; and
- `model_generated`: a model-written naturalistic candidate solution.

Redistribution is allowed only after the exact upstream dataset version and license
have been recorded. Unsupported subsets are referenced by ID rather than copied.

## 4. Verification protocol

### What the CAS does

SymPy is retained because exact symbolic comparison is valuable within a declared
operator language. It may verify exact arithmetic, algebraic equivalence, finite
candidate sets, explicit constraints, derivatives supported by the generator, and
state transitions produced by typed operations.

### What the CAS does not do

It does not judge prose entailment, whether an omitted explanation is acceptable,
informal proof validity, diagrams, arbitrary theorem use, or every alternative
derivation. Those cases cannot receive an automatic “verified” label.

### Verification stack

1. **Exact construction:** use integers/rationals and explicit domains wherever
   possible; avoid floating-point gold labels.
2. **Operator replay:** independently replay each formal action and check its
   post-state rather than merely comparing against a stored final answer.
3. **Independent property check:** test identities on at least 25 valid sampled
   assignments or use a second algorithmic calculation. This is a bug detector, not
   a proof, and is reported separately.
4. **Minimal-pair audit:** assert that clean and corrupted candidates differ only at
   declared steps and that the outcome-masked candidate retains the reference final
   answer.
5. **Human validation:** two annotators independently label every curated MATH and
   naturalistic item; disagreements are adjudicated. At least 20% of controlled and
   program-lifted items receive the same blind audit.
6. **Freeze audit:** randomly re-audit 10% after dataset freeze and report exact
   onset agreement plus a chance-corrected agreement statistic for categorical
   labels.
7. **Exclusion:** ambiguous items, unsupported notation, unverifiable source answers,
   and valid alternative derivations without a unique local verdict are excluded
   with a recorded reason.

Every item stores verification evidence. An item is release-eligible only if it has
either successful symbolic-plus-property verification or double/adjudicated human
verification. Drifted items additionally require `mutation_verified=true`.

## 5. Difficulty dimensions

Difficulty is defined before model evaluation:

- number of solution steps;
- peak live-state width;
- dependency depth;
- maximum dependency fan-in;
- maximum live span of a binding;
- number of accumulated constraints;
- number of active branches/candidates;
- number of symbol reuses;
- drift onset depth;
- propagation length; and
- recovery distance.

Headline analyses use predeclared low/medium/high bins computed from development
quantiles and then applied unchanged to test.

## 6. Primary metrics

### Critic benchmark

1. **Outcome-controlled balanced accuracy:** clean versus outcome-masked drift,
   excluding all wrong-final-answer items.
2. **Matched-triplet accuracy:** proportion of base problems for which all three
   candidates are classified correctly.
3. **Pairwise discrimination:** proportion of clean/outcome-masked pairs ranked in
   the correct order by invalidity confidence.
4. **First-error exact accuracy** and **within-one accuracy**.
5. **Component macro-F1** and **drift-type macro-F1**.
6. **Calibration:** Brier score and expected calibration error.
7. Results stratified by source, recovery mode, error count, and difficulty.

Wrong-answer controls are reported separately and are never allowed to inflate the
outcome-controlled headline metric.

### Solver/ledger study

- final-answer accuracy;
- hidden-drift rate among final-correct runs;
- strict gold-normalized state fidelity;
- first drift depth;
- propagation length;
- constraint and candidate-set fidelity;
- emitted-step coverage; and
- protocol/verification failure rate.

All C-versus-D comparisons are paired by model and problem. Report paired bootstrap
95% confidence intervals and the number of complete/failed units. The ledger study
is a mitigation analysis, not the definition of the benchmark.

## 7. Automated release gates

The validator must reject a release containing any of the following:

- duplicate item IDs;
- a base problem, upstream ID, leakage group, or normalized problem text crossing
  splits;
- a matched group missing any of its three required labels;
- mismatched problem text/reference answer inside a matched triplet;
- a clean item with error labels, or a drifted item without an onset;
- an outcome-masked item whose final answer differs from the reference answer;
- a wrong-answer control whose final answer equals the reference answer;
- non-contiguous step indices or out-of-range annotations;
- a recovery step at or before drift onset;
- undeclared changes between a clean candidate and its paired corruption;
- a broken formal state chain outside a labelled state-reset error;
- an inconsistent reference trace;
- missing license, source mode, upstream split, or contamination label;
- insufficient verification evidence;
- manifest count or SHA-256 mismatch; or
- a trace outside the declared 3–12 step core range.

Warnings, which must be counted in the paper artifact, include single-annotated
external items, missing optional formal states, or unsupported independent numerical
checks. No release may contain validator errors.

## 8. Paper structure for eight pages

1. Introduction and failure example — 0.9 page
2. Related work and distinction from ProcessBench/process supervision — 0.7 page
3. State-drift formulation and taxonomy — 1.0 page
4. Benchmark construction and verification — 1.7 pages
5. Experimental setup and metrics — 0.8 page
6. Results: critic benchmark — 1.2 pages
7. Results: solver/ledger mitigation and analysis — 0.8 page
8. Limitations, ethics, and conclusion — 0.9 page

The StateScope UI is supplementary and may receive at most a small qualitative
figure in the main paper. The paper's central artifact is the outcome-controlled
benchmark and its validation evidence.
