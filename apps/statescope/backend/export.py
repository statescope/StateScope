"""Export a StateScope session to JSON or Markdown (for the future UI / sharing)."""

from __future__ import annotations

from apps.statescope.backend.session import Session


def export_session_json(session: Session) -> str:
    return session.model_dump_json(indent=2)


def export_session_markdown(session: Session) -> str:
    m = session.metrics
    provenance = (session.trace.metadata or {}).get("model_provenance", {})
    lines = [
        f"# StateScope session: {session.problem.id}",
        "",
        f"- system: {session.system}",
        f"- condition: {session.condition}",
        f"- family: {session.problem.family}",
        f"- model: {provenance.get('model_id', 'not recorded')}",
        f"- execution route: {provenance.get('route', 'not recorded')}",
        f"- backend: {provenance.get('backend', 'not recorded')}",
        f"- SF: {m.sf:.3f}",
        f"- COD (first drift point): {session.cod}",
        f"- PL: {m.pl}",
        f"- final_correct: {m.final_correct}",
        f"- recovered: {m.recovered}",
        "",
        "## First drift point",
    ]
    if session.cod is None:
        lines.append("No drift detected (COD = None).")
    else:
        sd = next((d for d in session.state_diffs if d.step == session.cod), None)
        comps = ", ".join(sd.diff) if sd and sd.diff else "?"
        lines.append(f"Step {session.cod} first diverges from gold in: {comps}")

    lines += ["", "## Per-step state diffs"]
    for d in session.state_diffs:
        lines.append(f"- step {d.step}: {', '.join(d.diff) if d.diff else 'matches gold'}")

    if session.failure_events:
        lines += ["", "## Agentic failure events"]
        for fe in session.failure_events:
            lines.append(f"- step {fe.get('step')}: {fe.get('kind')} op={fe.get('op')!r} -- {fe.get('error')}")

    if session.adapter_log:
        lines += ["", "## Adapter log (parse/repair)"]
        for entry in session.adapter_log:
            lines.append(
                f"- step {entry.get('step')}: mode={entry.get('mode')} "
                f"repairs={entry.get('repair_attempts')} error={entry.get('parse_error')}"
            )

    return "\n".join(lines)
