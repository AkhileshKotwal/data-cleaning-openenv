"""OpenEnv server adapter for the local tuple-style workbench environment."""

from __future__ import annotations

from typing import Any, Optional

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:  # pragma: no cover
    class Environment:  # type: ignore[override]
        pass

    class EnvironmentMetadata:  # type: ignore[override]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

from environment import OpsWorkbenchEnv
from models import WorkbenchAction, WorkbenchObservation, WorkbenchState


class WorkbenchEnvironment(Environment[WorkbenchAction, WorkbenchObservation, WorkbenchState]):
    """Server-compatible adapter around the local environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task: str = "email_triage") -> None:
        super().__init__()
        self._env = OpsWorkbenchEnv(default_task=default_task)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WorkbenchObservation:
        return self._env.reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(
        self,
        action: WorkbenchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WorkbenchObservation:
        observation, _, _, _ = self._env.step(action, timeout_s=timeout_s, **kwargs)
        return observation

    @property
    def state(self) -> WorkbenchState:
        return self._env.state()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ops_workbench_env",
            description="Realistic operations tasks for email triage, support resolution, and data cleaning.",
            version="0.1.0",
            author="Codex",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
        )

    def close(self) -> None:
        self._env.close()
