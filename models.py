"""Typed models for the Ops Workbench data cleaning environment."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid")
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid")
        done: bool = False
        reward: float | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow")
        episode_id: Optional[str] = None
        step_count: int = 0


class WorkbenchAction(Action):
    """Single tool invocation for the data cleaning environment."""

    action_type: Literal[
        "inspect",
        "fill_nulls",
        "rename_column",
        "cast_type",
        "remove_duplicates",
        "fix_value",
        "normalize",
        "drop_column",
        "submit",
    ]
    column: Optional[str] = None
    strategy: Optional[Literal["mean", "median", "mode", "drop"]] = None
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    dtype: Optional[Literal["int", "float", "str", "datetime"]] = None
    find: Optional[str] = None
    replace: Optional[str] = None


class WorkbenchObservation(Observation):
    """Agent-visible observation."""

    benchmark: str = "ops_workbench"
    task_name: Literal["data_cleaning"] = "data_cleaning"
    difficulty: Literal["hard"] = "hard"
    objective: str
    tools: List[str]
    grading_criteria: Dict[str, str]
    dataset_columns: List[str]
    dataset_preview: List[Dict[str, str]]
    current_dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    duplicate_rows: int
    tool_result: Dict[str, Any] = Field(default_factory=dict)
    inspected_columns: List[str] = Field(default_factory=list)
    progress: float = Field(default=0.01, gt=0.0, lt=1.0)
    last_action_error: Optional[str] = None


class WorkbenchState(State):
    """Internal environment state."""

    current_task: str = "data_cleaning"
    max_steps: int = 30
    submitted: bool = False
    consecutive_errors: int = 0
    progress_score: float = 0.0
    last_action_error: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
