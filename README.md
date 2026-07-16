# StateScope

StateScope is an interactive system for detecting, localizing, explaining, and
repairing solution-state drift in multi-step mathematical reasoning. It compares
two systems under a shared typed-operation protocol:

- **System C** carries a model-owned textual state that can drift.
- **System D** re-derives a typed state in an external ledger.

Every transition is checked against a SymPy oracle. Retained steps can be edited
repeatedly, branched, replayed, and continued with the same model.

## Public demo

- UI: <https://statescope.github.io/StateScope/>
- Backend: <https://statescope-aacl-demo.onrender.com/>
- Health check: <https://statescope-aacl-demo.onrender.com/api/health>

GitHub Pages serves the same `index.html` as the Python application. The hosted
Python/SymPy runtime supplies the API. On localhost, the UI uses the same-origin
server automatically.

## Credential handling

Hosted provider keys are entered into a masked field and attached only to the
selected run or continuation request. They are not written to disk, browser
storage, exported traces, server provenance, or application logs. No credential
is committed to this repository.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[closed,dev]"
.\.venv\Scripts\python.exe -m apps.statescope.server
```

Open <http://127.0.0.1:8000>. The deterministic mock route works without an API
key. Proprietary routes accept a request-scoped key in the UI.

## Focused verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_statescope_server.py tests/test_statescope_backend.py tests/test_statescope_intervene.py
```

The evaluation and paper instructions are in
[`README_AACL_DEMO.md`](README_AACL_DEMO.md).
