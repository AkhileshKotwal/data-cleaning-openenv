"""FastAPI entrypoint for Ops Workbench."""

from __future__ import annotations

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


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
