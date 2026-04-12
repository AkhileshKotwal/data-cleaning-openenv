"""
client.py — HTTP client for the Data Cleaning OpenEnv environment.
Provides a clean Python interface to the FastAPI server.

Usage:
    with DataCleaningEnv(base_url="http://localhost:7860", task_id="fix_nulls") as env:
        obs = env.reset()
        session_id = obs["session_id"]
        result = env.step(DataCleaningAction(action_type="fill_nulls",
                                             column="age", strategy="mean"))
        print(result.reward, result.done)
"""

import httpx
from typing import Any, Dict, Optional

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)


class StepResult:
    """Holds the result of a step() call."""

    def __init__(self, observation: DataCleaningObservation,
                 reward: float, done: bool, info: Dict):
        self.observation = observation
        self.reward      = reward
        self.done        = done
        self.info        = info

    def __repr__(self):
        return (
            f"StepResult(reward={self.reward:.4f}, done={self.done}, "
            f"score={self.observation.score_so_far:.4f})"
        )


class DataCleaningEnv:
    """HTTP client for the Data Cleaning OpenEnv environment."""

    def __init__(self, base_url: str = "http://localhost:7860",
                 task_id: str = "fix_nulls",
                 adversarial: bool = False,
                 adversary_difficulty: float = 0.3,
                 seed: int = 42,
                 timeout: float = 30.0):
        self.base_url            = base_url.rstrip("/")
        self.task_id             = task_id
        self.adversarial         = adversarial
        self.adversary_difficulty = adversary_difficulty
        self.seed                = seed
        self._client             = httpx.Client(timeout=timeout)
        self._session_id: Optional[str] = None

    def reset(self) -> DataCleaningObservation:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={
                "task_id":             self.task_id,
                "adversarial":         self.adversarial,
                "adversary_difficulty": self.adversary_difficulty,
                "seed":                self.seed,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data.get("session_id")
        return DataCleaningObservation(**{
            k: v for k, v in data.items() if k != "session_id"
        })

    def step(self, action: DataCleaningAction) -> StepResult:
        assert self._session_id, "Call reset() before step()."
        payload = action.model_dump()
        payload["session_id"] = self._session_id
        payload["task_id"]    = self.task_id
        resp = self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data   = resp.json()
        reward = data.pop("reward", 0.0)
        done   = data.pop("done",   False)
        info   = data.pop("info",   {})
        data.pop("session_id", None)
        obs    = DataCleaningObservation(**data)
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> DataCleaningState:
        assert self._session_id, "Call reset() before state()."
        resp = self._client.get(
            f"{self.base_url}/state",
            params={"session_id": self._session_id},
        )
        resp.raise_for_status()
        return DataCleaningState(**resp.json())

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def from_docker_image(cls, image_name: str, task_id: str = "fix_nulls",
                          host_port: int = 7860) -> "DataCleaningEnv":
        """Start a local Docker container and return a connected client."""
        import subprocess, time
        subprocess.check_output([
            "docker", "run", "-d", "--rm",
            "-p", f"{host_port}:7860",
            image_name,
        ])
        time.sleep(3)
        return cls(base_url=f"http://localhost:{host_port}", task_id=task_id)
