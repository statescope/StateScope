"""StateScope demo server (Python standard-library HTTP layer).

The demo exposes every committed model configuration through explicit execution
routes: local vLLM for open weights, native vendor APIs for proprietary models, and
OpenRouter where configured. Hosted credentials stay in server-side environment
variables. The deterministic mock remains a clearly labeled offline/debug mode.

Run:
    python -m apps.statescope.server --host 127.0.0.1 --port 8000

Endpoints (JSON):
    GET  /api/health                  -> {ok, mode}
    GET  /api/models                  -> {ok, models:[{key, model_id, access, routes, ...}]}
    GET  /api/examples                -> [{id, title, difficulty, blurb, family, ...}]
    GET  /api/ops                     -> {ok, ops: {family: [{op, description, example_args, ...}]}}
    GET  /api/export?run=&system=&scope=run|whatif&fmt=json|md  -> downloadable session
    POST /api/run        {example_id, model_key, route_key?, base_url?, api_key?, execution_mode?, system:"both"|"c"|"d", drift_step?}
    POST /api/continue   {run_id, system:"c"|"d", step, base_branch_id?, op?, args?, claimed_state?, api_key?, mode:"model"|"replay"}
    POST /api/live/start {example_id, model_key, route_key?, base_url?, api_key?, execution_mode?, drift_step?} -> {live_id}
    POST /api/live/step  {live_id} -> {systems:{c,d}, done, run_id?}   # one model turn per call
    POST /api/live/stop  {live_id}
    POST /api/regenerate {example_id, seed?} -> {example}              # fresh-parameter reinstantiation
    POST /api/replay     {example_id, system:"c"|"d", step, formula?|args?}

/api/run executes the selected model live and remembers the session under run_id.
/api/continue is the interactive core: edit any step of that just-produced trace --
the operation (both systems) or, for System C, the *claimed state* itself (C's model
owns the state; D's ledger refuses state edits by design) -- and either let the
*same model* continue solving from the post-edit state ("model") or re-derive
downstream deterministically ("replay"; mock runs always use this).
/api/live/* steps a paired run one model turn per call, so an audience can watch
state evolve; the finished run is remembered like any /api/run. Each example
carries a SymPy-verified gold trace; the CAS is the correctness judge -- drift
explanations are generated deterministically from the state diff, never by an LLM.
"""

from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from apps.statescope.backend.export import export_session_json, export_session_markdown
from apps.statescope.backend.intervene import intervene_and_continue
from apps.statescope.backend.replay import counterfactual_replay
from apps.statescope.backend.session import Session, build_session, run_session
from apps.statescope.backend.stepper import SYSTEM_NAMES, LiveSolver
from apps.statescope.examples import fresh_variant, get_problem, list_examples
from apps.statescope import model_routes
from driftmath.adapters.runner_adapter import OperationAdapter
from driftmath.runtime import op_specs
from driftmath.runtime.tool_api import Ledger, apply_op_verified
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger

_WEB = Path(__file__).resolve().parent / "web"

_SYSTEMS = {"c": SystemCToolsText, "d": SystemDLedger}

MOCK_KEY = model_routes.MOCK_KEY

# Completed runs kept in memory so /api/continue and /api/export can address them.
_RUNS: OrderedDict[str, dict] = OrderedDict()
_MAX_RUNS = 32

# In-flight step-through runs (one model turn per /api/live/step call).
_LIVE: OrderedDict[str, dict] = OrderedDict()
_MAX_LIVE = 8
_MAX_BRANCHES_PER_SYSTEM = 128
_EXECUTION_MODES = {"controlled", "autonomous"}

# The static research UI is published on GitHub Pages while the Python/SymPy
# runtime is hosted separately.  Keep this allow-list narrow: request bodies can
# contain a user-supplied, request-scoped API key.
_ALLOWED_ORIGINS = {
    "https://statescope.github.io",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
}
_ALLOWED_ORIGINS.update(
    origin.strip().rstrip("/")
    for origin in os.environ.get("STATESCOPE_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
)


def _remember_run(entry: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    _RUNS[run_id] = entry
    while len(_RUNS) > _MAX_RUNS:
        _RUNS.popitem(last=False)
    return run_id

_COMPONENT_LABELS = {
    "bindings": "variable bindings",
    "constraints": "the constraint set",
    "current_equation": "the current equation",
    "current_expr": "the current expression",
    "candidates": "the candidate solutions",
    "final_answer": "the final answer",
}


def _verifications(trace) -> dict[int, str]:
    """Per-step CAS status by replaying the ops through a fresh ledger."""
    ledger = Ledger()
    out: dict[int, str] = {}
    for st in trace.steps:
        try:
            res = apply_op_verified(ledger, {"op": st.op, "args": st.args})
            out[st.index] = res.verification.get("status", "skipped")
        except Exception:
            out[st.index] = "skipped"
    return out


def build_drift_explanation(session: Session, verifs: dict[int, str]) -> dict:
    """Deterministic drift explanation from the state diff + CAS verdicts (no LLM judging).

    Distinguishes two very different situations that both show as index-aligned drift:
    a *stale carried state* (same operation as the oracle, different state -- a real
    bookkeeping error) and a *schedule divergence* (the model chose a different
    operation than the oracle at that index -- possibly a valid alternative
    derivation, which the strict index-by-index comparison penalizes by design).
    """
    diff_by_step = {d.step: d.diff for d in session.state_diffs}
    all_diverged = sorted({c for d in session.state_diffs for c in d.diff})
    premature_final_steps = [
        st.index
        for st in session.trace.steps
        if st.after_state.final_answer is not None and not op_specs.is_terminal_op(st.op)
    ]
    cod = session.cod
    if cod is None:
        explanation = "No drift detected: every aligned step's state matches the CAS oracle."
        if premature_final_steps:
            explanation += (
                f" Step {premature_final_steps[0]} contains a provisional answer before the explicit "
                "terminal operation, so the fixed schedule correctly continues."
            )
        return {
            "first_drift_step": None,
            "first_drift_op": None,
            "gold_op_at_drift": None,
            "schedule_divergence": False,
            "diverged_components": [],
            "all_diverged_components": all_diverged,
            "cas_status_at_drift": None,
            "premature_final_steps": premature_final_steps,
            "explanation": explanation,
        }

    comps = diff_by_step.get(cod, [])
    step = next((s for s in session.trace.steps if s.index == cod), None)
    gold_step = next((s for s in session.gold_trace.steps if s.index == cod), None)
    op = step.op if step else "?"
    gold_op = gold_step.op if gold_step else None
    cas = verifs.get(cod, "skipped")
    schedule_divergence = step is not None and gold_op is not None and op != gold_op
    labels = ", ".join(_COMPONENT_LABELS.get(c, c) for c in comps) or "the recorded state"

    if schedule_divergence:
        head = (
            f"First divergence at step {cod}: the model executed '{op}' where the oracle derivation performs "
            f"'{gold_op}', so from here the run follows a different operation schedule than the reference. "
            "States are compared index-by-index, which penalizes an alternative derivation even when every "
            "executed step is mathematically sound; "
        )
        if cas == "ok":
            head += "this step itself passed CAS verification, so treat this as a schedule difference to inspect, not a proven error."
        elif cas == "failed":
            head += "and this step also failed CAS verification, so the alternative path starts with an invalid operation."
        else:
            head += "this step was not CAS-checkable."
        parts = [head]
    else:
        if cas == "ok":
            cas_clause = (
                "the executed operation passed CAS verification, so this is a state-tracking error "
                "(the carried state went stale), not a computation error"
            )
        elif cas == "failed":
            cas_clause = "the operation also failed CAS verification, so the step itself was mathematically invalid"
        else:
            cas_clause = "the operation was not CAS-checkable at this step"
        parts = [f"First divergence at step {cod} (op '{op}'): the state differs from the oracle in {labels}; {cas_clause}."]

    if session.metrics.recovered:
        parts.append("A later step returned to the oracle state (recovered).")
    if premature_final_steps:
        parts.append(
            f"Step {premature_final_steps[0]} contains a provisional answer claim before the explicit "
            "terminal operation; it is recorded as drift and does not stop the controlled schedule."
        )
    parts.append("The final answer is correct." if session.metrics.final_correct else "The final answer is incorrect.")

    return {
        "first_drift_step": cod,
        "first_drift_op": op,
        "gold_op_at_drift": gold_op,
        "schedule_divergence": schedule_divergence,
        "diverged_components": comps,
        "all_diverged_components": all_diverged,
        "cas_status_at_drift": cas,
        "premature_final_steps": premature_final_steps,
        "explanation": " ".join(parts),
    }


def _payload(session: Session) -> dict:
    verifs = _verifications(session.trace)
    diff_by_step = {d.step: d.diff for d in session.state_diffs}
    return {
        "system": session.system,
        "condition": session.condition,
        "problem_text": session.problem.problem_text,
        "family": session.problem.family,
        "gold_answer": session.problem.gold_answer,
        "final_answer": session.trace.final_answer,
        "metrics": session.metrics.model_dump(),
        "cod": session.cod,
        "failure_events": session.failure_events,
        "adapter_log": session.adapter_log,
        "drift_explanation": build_drift_explanation(session, verifs),
        "steps": [
            {
                "index": st.index,
                "op": st.op,
                "args": st.args,
                "after_state": st.after_state.model_dump(),
                "diff": diff_by_step.get(st.index, []),
                "verify": verifs.get(st.index, "skipped"),
                "terminal": op_specs.is_terminal_op(st.op),
                "premature_final": (
                    st.after_state.final_answer is not None and not op_specs.is_terminal_op(st.op)
                ),
            }
            for st in session.trace.steps
        ],
        "gold_steps": [
            {
                "index": s.index,
                "op": s.op,
                "args": s.args,
                "after_state": s.after_state.model_dump(),
                "terminal": op_specs.is_terminal_op(s.op),
            }
            for s in session.gold_trace.steps
        ],
    }


def models_endpoint() -> dict:
    """Every configured model with its valid local, native, and hosted routes."""
    return {"ok": True, "models": model_routes.catalog()}


def _make_model(
    model_key: str,
    route_key: str | None,
    base_url: str | None,
    execution_mode: str = "controlled",
    api_key: str | None = None,
):
    """Instantiate a catalog model and return its adapter and export-safe provenance."""
    if execution_mode not in _EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {sorted(_EXECUTION_MODES)}, got {execution_mode!r}")
    model, model_info, route = model_routes.make_model(model_key, route_key, base_url, api_key=api_key)
    adapter = None if model_key == MOCK_KEY else OperationAdapter(
        mode="text_json",
        repair_budget=2,
        controlled_schedule=execution_mode == "controlled",
    )
    provenance = model_routes.provenance(model_info, route, base_url)
    provenance["execution_mode"] = execution_mode
    return model, adapter, provenance


def _record_provenance(session: Session, provenance: dict) -> None:
    metadata = dict(session.trace.metadata or {})
    metadata["model_provenance"] = provenance
    session.trace.metadata = metadata


def _friendly_model_error(e: Exception, secret: str | None = None) -> str:
    """Surface backend errors with an actionable hint (no credentials, no payloads)."""
    msg = f"{type(e).__name__}: {e}"
    if secret:
        msg = msg.replace(secret, "[redacted]")
    text = str(e).lower()
    if "connection" in text or "connect" in text or "timed out" in text or "timeout" in text:
        return msg + " — could not reach the endpoint; is the model server running at that URL?"
    if "404" in text or "not found" in text:
        return msg + " — the endpoint answered, but there is no chat-completions API at that URL (wrong port, path, or a non-vLLM server?)"
    if "401" in text or "403" in text or "api key" in text or "credential" in text or "unauthorized" in text:
        return msg + " — credential problem; enter the key in the masked field or configure it on the server"
    return msg


def run_endpoint(body: dict) -> dict:
    problem = get_problem(body["example_id"])
    which = body.get("system", "both")
    systems = ["c", "d"] if which == "both" else [which]
    model_key = body.get("model_key") or body.get("model") or MOCK_KEY
    route_key = body.get("route_key") or None
    base_url = body.get("base_url") or None
    api_key = body.get("api_key").strip() if isinstance(body.get("api_key"), str) else None
    drift_step = body.get("drift_step")
    execution_mode = body.get("execution_mode", "controlled")

    out: dict = {
        "ok": True,
        "model_key": model_key,
        "route_key": route_key,
        "execution_mode": execution_mode,
    }
    sessions: dict[str, Session] = {}
    run_provenance: dict | None = None
    for key in systems:
        system = _SYSTEMS[key]()
        try:
            model, adapter, run_provenance = _make_model(
                model_key, route_key, base_url, execution_mode, api_key=api_key
            )
            if adapter is not None:
                session = run_session(problem, system, model, condition="clean", adapter=adapter)
            else:
                condition = f"natural_mock_drift:{int(drift_step)}" if drift_step is not None else "clean"
                session = run_session(problem, system, model, condition=condition)
            _record_provenance(session, run_provenance)
            sessions[key] = session
            out[key] = _payload(session)
        except KeyError as e:
            return {"ok": False, "error": f"{e}; see /api/models"}
        except Exception as e:  # surface any model/endpoint/parse error cleanly
            return {"ok": False, "error": _friendly_model_error(e, api_key)}
    out["run_id"] = _remember_run(
        {
            "example_id": body["example_id"],
            "model_key": model_key,
            "route_key": route_key,
            "base_url": base_url,
            "execution_mode": execution_mode,
            "provenance": run_provenance,
            "sessions": sessions,
            "whatif": {},
            "branches": {"c": OrderedDict(), "d": OrderedDict()},
        }
    )
    out["model_provenance"] = run_provenance
    return out


def continue_endpoint(body: dict) -> dict:
    """Edit one step of a remembered run, then continue it.

    mode "model" (default): the run's own model keeps solving live from the
    post-edit state, under the chosen system's state-ownership rules.
    mode "replay": deterministic downstream re-derivation on the ledger, no model
    turns. Mock runs always use replay (there is no live model behind them).
    """
    run = _RUNS.get(str(body.get("run_id") or ""))
    if run is None:
        return {"ok": False, "error": "unknown or expired run_id -- run the problem first"}
    key = body.get("system", "d")
    if key not in _SYSTEMS:
        return {"ok": False, "error": f"unknown system {key!r}"}
    parent_branch_id = str(body.get("base_branch_id") or "") or None
    branches = run.setdefault("branches", {"c": OrderedDict(), "d": OrderedDict()})
    branch_pool = branches.setdefault(key, OrderedDict())
    base_session = branch_pool.get(parent_branch_id) if parent_branch_id else run["sessions"].get(key)
    if base_session is None:
        if parent_branch_id:
            return {"ok": False, "error": f"unknown or expired intervention branch {parent_branch_id!r}"}
        return {"ok": False, "error": f"system {key!r} was not part of this run"}
    problem = get_problem(run["example_id"])
    try:
        step = int(body["step"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "step (int) is required"}
    base_trace = base_session.trace
    retained = len(base_trace.steps)
    can_recover_next = step == retained and step < len(problem.gold_trace.steps)
    if not (0 <= step < retained or can_recover_next):
        hi = max(retained - 1, 0)
        return {
            "ok": False,
            "error": f"step {step} is out of range for this trace (retained 0..{hi}; "
            "only the immediately missing next step can be recovered)",
        }
    source_step = base_trace.steps[step] if step < retained else problem.gold_trace.steps[step]

    new_op = body.get("op") or None
    new_args = body.get("args") if isinstance(body.get("args"), dict) else None
    claimed = body.get("claimed_state") if isinstance(body.get("claimed_state"), dict) else None
    if claimed is not None and key == "d":
        return {"ok": False, "error": "System D's ledger owns the state -- edit the operation instead (that is the point)"}
    mode = body.get("mode", "model")
    if mode not in {"model", "replay"}:
        return {"ok": False, "error": "mode must be 'model' or 'replay'"}
    if run["model_key"] == MOCK_KEY:
        mode = "replay"
    api_key = body.get("api_key").strip() if isinstance(body.get("api_key"), str) else None

    try:
        if mode == "replay":
            session = counterfactual_replay(
                problem,
                (step, {"op": new_op, "args": new_args, "after_state": claimed}),
                _SYSTEMS[key](),
                base_trace=base_trace,
            )
        else:
            model, adapter, run_provenance = _make_model(
                run["model_key"],
                run.get("route_key"),
                body.get("base_url") or run["base_url"],
                run.get("execution_mode", "controlled"),
                api_key=api_key,
            )
            session = intervene_and_continue(
                problem, base_trace, system_key=key, step=step,
                op=new_op, args=new_args, claimed_state=claimed, model=model, adapter=adapter,
            )
    except Exception as e:
        return {"ok": False, "error": _friendly_model_error(e, api_key)}

    intervention = {
        "step": step,
        "op": new_op or source_step.op,
        "args": new_args if new_args is not None else dict(source_step.args),
        "claimed_state_override": claimed is not None,
        "mode": mode,
    }
    history = list((base_trace.metadata or {}).get("interventions", []))
    history.append(intervention)
    metadata = dict(session.trace.metadata or {})
    metadata["interventions"] = history
    session.trace.metadata = metadata
    session.condition = f"interventions:{len(history)}"
    session.trace.metadata["condition"] = session.condition
    _record_provenance(session, run["provenance"])

    branch_id = uuid.uuid4().hex[:12]
    branch_pool[branch_id] = session
    while len(branch_pool) > _MAX_BRANCHES_PER_SYSTEM:
        branch_pool.popitem(last=False)
    run["whatif"][key] = session
    return {
        "ok": True,
        "mode": mode,
        "system": key,
        "step": step,
        "branch_id": branch_id,
        "parent_branch_id": parent_branch_id,
        "intervention_count": len(history),
        "branch_count": len(branch_pool),
        "branch_limit": _MAX_BRANCHES_PER_SYSTEM,
        "edited_op": new_op or source_step.op,
        "edited_claimed_state": claimed is not None,
        "recovered_missing_step": can_recover_next,
        "result": _payload(session),
        "original": {
            "metrics": base_session.metrics.model_dump(),
            "cod": base_session.cod,
            "final_answer": base_trace.final_answer,
        },
    }


def live_start_endpoint(body: dict) -> dict:
    """Begin a paired step-through run: both systems, one model turn per /step call."""
    try:
        problem = get_problem(body["example_id"])
    except KeyError:
        return {"ok": False, "error": f"unknown example {body.get('example_id')!r}"}
    model_key = body.get("model_key") or MOCK_KEY
    route_key = body.get("route_key") or None
    base_url = body.get("base_url") or None
    api_key = body.get("api_key").strip() if isinstance(body.get("api_key"), str) else None
    drift_step = body.get("drift_step")
    execution_mode = body.get("execution_mode", "controlled")

    condition = "clean"
    solvers: dict[str, dict] = {}
    run_provenance: dict | None = None
    try:
        for key in ("c", "d"):
            model, adapter, run_provenance = _make_model(
                model_key, route_key, base_url, execution_mode, api_key=api_key
            )
            if adapter is None:  # mock: prime the script (optionally with planted drift)
                condition = f"natural_mock_drift:{int(drift_step)}" if drift_step is not None else "clean"
                model.reset(problem=problem, condition=condition)
            solvers[key] = {"solver": LiveSolver(key), "model": model, "adapter": adapter}
    except KeyError as e:
        return {"ok": False, "error": f"{e}; see /api/models"}
    except Exception as e:
        return {"ok": False, "error": _friendly_model_error(e, api_key)}

    live_id = uuid.uuid4().hex[:12]
    _LIVE[live_id] = {
        "example_id": body["example_id"],
        "model_key": model_key,
        "route_key": route_key,
        "base_url": base_url,
        "execution_mode": execution_mode,
        "provenance": run_provenance,
        "condition": condition,
        "solvers": solvers,
        "budget": len(problem.gold_trace.steps) + 2,
    }
    while len(_LIVE) > _MAX_LIVE:
        _LIVE.popitem(last=False)
    return {
        "ok": True,
        "live_id": live_id,
        "budget": _LIVE[live_id]["budget"],
        "model_provenance": run_provenance,
        "execution_mode": execution_mode,
    }


def live_step_endpoint(body: dict) -> dict:
    """Advance every unfinished system by exactly one model turn; return partial payloads."""
    live_id = str(body.get("live_id") or "")
    entry = _LIVE.get(live_id)
    if entry is None:
        return {"ok": False, "error": "unknown or expired live_id -- start a step-through run first"}
    problem = get_problem(entry["example_id"])

    sessions: dict[str, Session] = {}
    out_systems: dict[str, dict] = {}
    for key, cell in entry["solvers"].items():
        solver: LiveSolver = cell["solver"]
        if not solver.done and solver.next_index < entry["budget"]:
            try:
                solver.turn(problem, cell["model"], cell["adapter"])
            except Exception as e:  # dead endpoint / parse crash: drop the live run cleanly
                _LIVE.pop(live_id, None)
                return {"ok": False, "error": _friendly_model_error(e)}
        if solver.next_index >= entry["budget"]:
            solver.done = True
        session = build_session(
            problem, solver.trace(problem, entry["condition"]), system=SYSTEM_NAMES[key], condition=entry["condition"]
        )
        _record_provenance(session, entry["provenance"])
        sessions[key] = session
        out_systems[key] = {**_payload(session), "done": solver.done}

    all_done = all(cell["solver"].done for cell in entry["solvers"].values())
    out: dict = {"ok": True, "live_id": live_id, "done": all_done, "systems": out_systems}
    if all_done:
        out["run_id"] = _remember_run(
            {
                "example_id": entry["example_id"],
                "model_key": entry["model_key"],
                "route_key": entry["route_key"],
                "base_url": entry["base_url"],
                "execution_mode": entry["execution_mode"],
                "provenance": entry["provenance"],
                "sessions": sessions,
                "whatif": {},
                "branches": {"c": OrderedDict(), "d": OrderedDict()},
            }
        )
        _LIVE.pop(live_id, None)
    return out


def live_stop_endpoint(body: dict) -> dict:
    _LIVE.pop(str(body.get("live_id") or ""), None)
    return {"ok": True}


def regenerate_endpoint(body: dict) -> dict:
    """Reinstantiate the selected trap with fresh parameters (dynamic generation, live)."""
    try:
        example = fresh_variant(str(body.get("example_id") or ""), body.get("seed"))
    except KeyError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "example": example}


def ops_endpoint() -> dict:
    """Per-family operation vocabulary, derived from the op-spec table (for the editor)."""
    out: dict[str, list[dict]] = {}
    for fam in ("family_a", "family_b", "family_c", "family_d"):
        specs = [op_specs.get_spec(name) for name in sorted(op_specs.ops_for_family(fam))]
        out[fam] = [
            {"op": s.name, "description": s.description, "example_args": s.example_args, "cas_verified": s.cas_verified}
            for s in specs
            if s is not None
        ]
    return {"ok": True, "ops": out}


def export_endpoint(query: dict) -> tuple[bytes, str, str] | dict:
    """Resolve an export request; returns (payload, content_type, filename) or an error dict."""
    run = _RUNS.get(str(query.get("run") or ""))
    if run is None:
        return {"ok": False, "error": "unknown or expired run"}
    key = query.get("system", "d")
    scope = query.get("scope", "run")
    branch_id = str(query.get("branch") or "") or None
    if branch_id:
        session = run.get("branches", {}).get(key, {}).get(branch_id)
        scope = "branch"
    else:
        pool = run["whatif"] if scope == "whatif" else run["sessions"]
        session = pool.get(key)
    if session is None:
        return {"ok": False, "error": f"no {scope} session for system {key!r}"}
    if query.get("fmt") == "md":
        text, ctype, ext = export_session_markdown(session), "text/markdown; charset=utf-8", "md"
    else:
        text, ctype, ext = export_session_json(session), "application/json", "json"
    return text.encode("utf-8"), ctype, f"statescope_{run['example_id']}_{scope}_{key}.{ext}"


def replay_endpoint(body: dict) -> dict:
    problem = get_problem(body["example_id"])
    key = body.get("system", "d")
    step = int(body["step"])
    base = problem.gold_trace
    if step >= len(base.steps):
        return {"ok": False, "error": "step out of range"}
    new_args = dict(base.steps[step].args)
    if body.get("args"):
        new_args.update(body["args"])
    if "formula" in body:
        new_args["formula"] = body["formula"]
    try:
        session = counterfactual_replay(problem, (step, {"args": new_args}), _SYSTEMS[key]())
        return {"ok": True, "edited_step": step, "result": _payload(session)}
    except Exception as e:
        return {"ok": False, "error": _friendly_model_error(e)}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        return

    def _send_cors_headers(self) -> None:
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if origin in _ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self):
        """Answer the JSON POST preflight used by the GitHub Pages client."""
        origin = (self.headers.get("Origin") or "").rstrip("/")
        if origin not in _ALLOWED_ORIGINS:
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _send(self, obj: dict, code: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json",
        }.get(path.suffix, "application/octet-stream")
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self._send_cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send_file(_WEB / "index.html")
        if self.path == "/api/health":
            return self._send({"ok": True, "mode": "statescope-research"})
        if self.path == "/api/models":
            return self._send(models_endpoint())
        if self.path == "/api/examples":
            return self._send({"ok": True, "examples": list_examples()})
        if self.path == "/api/ops":
            return self._send(ops_endpoint())
        if self.path.startswith("/api/export"):
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            res = export_endpoint(query)
            if isinstance(res, dict):
                return self._send(res, 404)
            data, ctype, filename = res
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self._send_cors_headers()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if not self.path.startswith("/api/"):
            asset = (_WEB / self.path.lstrip("/")).resolve()
            if _WEB in asset.parents and asset.is_file():
                return self._send_file(asset)
        return self._send({"ok": False, "error": "not found"}, 404)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_POST(self):
        try:
            body = self._read_body()
        except Exception as e:
            return self._send({"ok": False, "error": f"bad request: {e}"}, 400)
        if self.path == "/api/run":
            return self._send(run_endpoint(body))
        if self.path == "/api/continue":
            return self._send(continue_endpoint(body))
        if self.path == "/api/live/start":
            return self._send(live_start_endpoint(body))
        if self.path == "/api/live/step":
            return self._send(live_step_endpoint(body))
        if self.path == "/api/live/stop":
            return self._send(live_stop_endpoint(body))
        if self.path == "/api/regenerate":
            return self._send(regenerate_endpoint(body))
        if self.path == "/api/replay":
            return self._send(replay_endpoint(body))
        return self._send({"ok": False, "error": "not found"}, 404)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="StateScope demo server (provider-aware research catalog).")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    n_models = len(model_routes.catalog()) - 1
    print(f"StateScope -> http://{args.host}:{args.port}   ({n_models} models configured; mock = offline debug)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
