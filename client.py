"""Typed OpenEnv client for Ops Workbench."""

from __future__ import annotations

from typing import Generic, TypeVar

try:
    from openenv.core.env_client import EnvClient
except ImportError:
    ActionT = TypeVar("ActionT")
    ObservationT = TypeVar("ObservationT")
    StateT = TypeVar("StateT")

    class EnvClient(Generic[ActionT, ObservationT, StateT]):  # pragma: no cover
        pass

from .models import WorkbenchAction, WorkbenchObservation, WorkbenchState


class OpsWorkbenchClient(EnvClient[WorkbenchAction, WorkbenchObservation, WorkbenchState]):
    """Thin typed client wrapper."""

    pass
