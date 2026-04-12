"""FastAPI entrypoint for Ops Workbench."""

from __future__ import annotations

from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "OpenEnv must be installed to run the server. Install dependencies from pyproject.toml."
    ) from exc

from models import WorkbenchAction, WorkbenchObservation
from server.workbench_environment import WorkbenchEnvironment

app = create_app(
    WorkbenchEnvironment,
    WorkbenchAction,
    WorkbenchObservation,
    env_name="ops_workbench_env",
    max_concurrent_envs=4,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> HTMLResponse:
    """Landing page shown in the HF Space iframe."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ops Workbench — Data Cleaning Environment</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #e6edf3; min-height: 100vh;
         display: flex; flex-direction: column; align-items: center;
         justify-content: center; padding: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
          padding: 40px; max-width: 600px; width: 100%; text-align: center; }
  .badge { display: inline-block; background: #1a7f37; color: #fff;
           border-radius: 20px; padding: 4px 14px; font-size: 13px;
           font-weight: 600; margin-bottom: 20px; }
  h1 { font-size: 24px; font-weight: 700; margin-bottom: 10px; }
  p  { color: #8b949e; font-size: 14px; line-height: 1.6; margin-bottom: 24px; }
  .endpoints { text-align: left; background: #0d1117; border-radius: 8px;
               padding: 16px; font-family: monospace; font-size: 13px; }
  .endpoints div { padding: 4px 0; color: #58a6ff; }
  .endpoints span { color: #8b949e; }
  .links { margin-top: 24px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
  a { background: #21262d; border: 1px solid #30363d; color: #e6edf3;
      padding: 8px 18px; border-radius: 8px; text-decoration: none;
      font-size: 13px; font-weight: 500; }
  a:hover { border-color: #58a6ff; color: #58a6ff; }
</style>
</head>
<body>
<div class="card">
  <div class="badge">● Running</div>
  <h1>🧹 Ops Workbench</h1>
  <p>Data cleaning RL environment. An agent inspects and repairs a messy
     customer dataset — resolving nulls, fixing types, removing duplicates,
     standardising values and renaming columns to snake_case.</p>
  <div class="endpoints">
    <div><span>POST </span>/reset  — start a new episode</div>
    <div><span>POST </span>/step   — take one cleaning action</div>
    <div><span>GET  </span>/state  — current episode state</div>
    <div><span>GET  </span>/health — server health check</div>
    <div><span>GET  </span>/schema — action / observation schemas</div>
  </div>
  <div class="links">
    <a href="/docs">📖 API Docs</a>
    <a href="/health">❤️ Health</a>
    <a href="/schema">📐 Schema</a>
    <a href="/metadata">ℹ️ Metadata</a>
  </div>
</div>
</body>
</html>""")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
